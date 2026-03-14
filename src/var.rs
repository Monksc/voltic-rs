use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    AddOp, ConstantOp, Context, DivOp, ExpOp, GeluOp, MatMulOp, MseOp, MulOp, PermuteOp, ReLUOp,
    ReshapeOp, SigmoidOp, SoftmaxOp, SubOp, TanhOp, VolticError, buffer_kind, errors::Result,
};

#[derive(Copy, Debug, Clone, PartialEq, Hash, Eq)]
pub struct ID(u64);

pub static VARIABLES_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

impl std::fmt::Display for ID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl ID {
    pub fn next() -> Self {
        Self(VARIABLES_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Copy, Debug, Clone)]
pub struct Var(ID);

impl Var {
    pub fn new() -> Self {
        Self(ID::next())
    }

    pub fn with_shape(shape: Vec<u32>) -> Self {
        let id = ID::next();
        Context::insert_shape(id, shape);
        Self(id)
    }

    pub fn shape(&self) -> Vec<u32> {
        Context::shape(self.id()).unwrap()
    }

    pub fn load(&self, data: Vec<Vec<f32>>) -> Result<()> {
        Context::load(self.0, data)
    }

    pub fn make_scalar(scalar: f32) -> Result<Self> {
        let v = Self::with_shape(vec![1]);
        v.load(vec![vec![scalar]])?;
        Context::push_operation(Box::new(ConstantOp::new(v.0)));
        Ok(v)
    }

    pub fn load_causal_mask(&self) -> Result<()> {
        let shape = Context::shape(self.0).ok_or(VolticError::EmptyShape)?;
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(VolticError::IncompatibleShapes {
                lhs: shape,
                rhs: vec![],
                op: "load_causal_mask: expected square 2D tensor",
            });
        }
        let seq_len = shape[0] as usize;
        let data: Vec<Vec<f32>> = (0..seq_len)
            .map(|row| {
                (0..seq_len)
                    .map(|col| if col <= row { 0.0 } else { f32::NEG_INFINITY })
                    .collect()
            })
            .collect();
        self.load(data)
    }

    pub fn to_cpu(&self) -> Result<Vec<f32>> {
        Context::read((self.0, ""))
    }

    pub fn grad(&self) -> Result<Vec<f32>> {
        Context::read((self.0, buffer_kind::GRAD))
    }

    pub fn id(&self) -> ID {
        self.0
    }

    pub fn mse(&self, y_true: Var) -> Result<Var> {
        let pred_shape = Context::shape(self.0).ok_or(VolticError::EmptyShape)?;
        let true_shape = Context::shape(y_true.0).ok_or(VolticError::EmptyShape)?;

        if pred_shape != true_shape {
            return Err(VolticError::IncompatibleShapes {
                lhs: pred_shape.clone(),
                rhs: true_shape,
                op: "mse",
            });
        }

        let n: u32 = pred_shape.iter().product();
        let output = Var::new();
        Context::insert_shape(output.0, vec![n]);
        Context::push_operation(Box::new(MseOp::new(self.0, y_true.0, output.0, n)));
        Ok(output)
    }

    fn apply_activation(
        &self,
        op: impl FnOnce(ID, ID, u32) -> Box<dyn crate::ops::Op>,
    ) -> Result<Var> {
        let shape = Context::shape(self.0).ok_or(VolticError::EmptyShape)?;
        let n: u32 = shape.iter().product();
        let output = Var::new();
        Context::insert_shape(output.0, shape);
        Context::push_operation(op(self.0, output.0, n));
        Ok(output)
    }

    pub fn tanh(&self) -> Result<Var> {
        self.apply_activation(|i, o, n| Box::new(TanhOp::new(i, o, n)))
    }

    pub fn relu(&self) -> Result<Var> {
        self.apply_activation(|i, o, n| Box::new(ReLUOp::new(i, o, n)))
    }

    pub fn sigmoid(&self) -> Result<Var> {
        self.apply_activation(|i, o, n| Box::new(SigmoidOp::new(i, o, n)))
    }

    pub fn gelu(&self) -> Result<Var> {
        self.apply_activation(|i, o, n| Box::new(GeluOp::new(i, o, n)))
    }

    pub fn exp(&self) -> Result<Var> {
        self.apply_activation(|i, o, n| Box::new(ExpOp::new(i, o, n)))
    }

    pub fn softmax(&self, axis: usize) -> Result<Var> {
        let shape = Context::shape(self.0).ok_or(VolticError::EmptyShape)?;
        if axis >= shape.len() {
            return Err(VolticError::InvalidDimension {
                dim: axis,
                ndim: shape.len(),
            });
        }
        let outer: u32 = shape[..axis].iter().product();
        let reduce: u32 = shape[axis];
        let inner: u32 = shape[axis + 1..].iter().product();
        let output = Var::new();
        Context::insert_shape(output.0, shape);
        Context::push_operation(Box::new(SoftmaxOp::new(
            self.0, output.0, outer, reduce, inner,
        )));
        Ok(output)
    }

    pub fn prev(&self) -> Self {
        Self(ID(self.0.0 - 1))
    }
    pub fn next(&self) -> Self {
        Self(ID(self.0.0 + 1))
    }

    pub fn permute(&self, perm: &[usize]) -> Result<Self> {
        let shape = Context::shape(self.0).ok_or(VolticError::EmptyShape)?;

        if perm.len() != shape.len() {
            return Err(VolticError::InvalidDimension {
                dim: perm.len(),
                ndim: shape.len(),
            });
        }

        // Validate permutation — must be a valid permutation of 0..rank
        let mut seen = vec![false; perm.len()];
        for &p in perm {
            if p >= perm.len() || seen[p] {
                return Err(VolticError::InvalidDimension {
                    dim: p,
                    ndim: perm.len(),
                });
            }
            seen[p] = true;
        }

        let out_shape: Vec<u32> = perm.iter().map(|&p| shape[p]).collect();
        let output = Self::new();
        Context::insert_shape(output.0, out_shape);
        Context::push_operation(Box::new(PermuteOp::new(
            self.0,
            output.0,
            shape,
            perm.to_vec(),
        )));
        Ok(output)
    }

    /// Convenience wrapper — swap two axes
    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<Self> {
        let shape = Context::shape(self.0).ok_or(VolticError::EmptyShape)?;
        let rank = shape.len();

        if axis_a >= rank || axis_b >= rank {
            return Err(VolticError::InvalidDimension {
                dim: axis_a.max(axis_b),
                ndim: rank,
            });
        }

        let mut perm: Vec<usize> = (0..rank).collect();
        perm.swap(axis_a, axis_b);
        self.permute(&perm)
    }

    pub fn mat_mul(&self, rhs: Var) -> Result<Self> {
        let lhs_shape = Context::shape(self.0).ok_or(VolticError::EmptyShape)?;
        let rhs_shape = Context::shape(rhs.0).ok_or(VolticError::EmptyShape)?;
        let lhs_rank = lhs_shape.len();
        let rhs_rank = rhs_shape.len();

        if lhs_rank < 2 || rhs_rank < 2 {
            return Err(VolticError::InvalidDimension {
                dim: lhs_rank.min(rhs_rank),
                ndim: 2,
            });
        }

        // Last two dims: lhs [..., M, K] @ rhs [..., K, N]
        let m = lhs_shape[lhs_rank - 2];
        let k = lhs_shape[lhs_rank - 1];
        let k2 = rhs_shape[rhs_rank - 2];
        let n = rhs_shape[rhs_rank - 1];

        if k != k2 {
            return Err(VolticError::IncompatibleShapes {
                lhs: lhs_shape.clone(),
                rhs: rhs_shape.clone(),
                op: "mat_mul: K mismatch",
            });
        }

        let lhs_batch = &lhs_shape[..lhs_rank - 2];
        let rhs_batch = &rhs_shape[..rhs_rank - 2];

        // rhs is allowed to be 2D (no batch dims) — acts as shared weights
        if rhs_rank > 2 && lhs_batch != rhs_batch {
            return Err(VolticError::IncompatibleShapes {
                lhs: lhs_shape.clone(),
                rhs: rhs_shape.clone(),
                op: "mat_mul: batch dims mismatch",
            });
        }

        let batch: u32 = lhs_batch.iter().product::<u32>().max(1);

        // Output shape: [...lhs_batch, M, N]
        let mut out_shape = lhs_batch.to_vec();
        out_shape.push(m);
        out_shape.push(n);

        let output = Self::new();
        Context::insert_shape(output.0, out_shape);
        Context::push_operation(Box::new(MatMulOp::new(
            self.0, rhs.0, output.0, batch, m, k, n,
        )));
        Ok(output)
    }

    pub fn reshape(&self, new_shape: Vec<u32>) -> Result<Self> {
        let shape = Context::shape(self.0).ok_or(VolticError::EmptyShape)?;

        let old_n: u32 = shape.iter().product();
        let new_n: u32 = new_shape.iter().product();

        if old_n != new_n {
            return Err(VolticError::IncompatibleShapes {
                lhs: shape,
                rhs: new_shape,
                op: "reshape: element count mismatch",
            });
        }

        let output = Self::new();
        Context::insert_shape(output.0, new_shape.clone());
        Context::push_operation(Box::new(ReshapeOp::new(self.0, output.0, new_shape)));
        Ok(output)
    }
}

macro_rules! impl_binary_op {
    ($method:ident, $safe:ident, $bc:ident, $Op:ty, $trait:ident, $trait_fn:ident) => {
        impl Var {
            // LCS broadcast — core path
            pub fn $safe(self, rhs: Var) -> Result<Var> {
                let lhs_shape = Context::shape(self.0).ok_or(VolticError::EmptyShape)?;
                let rhs_shape = Context::shape(rhs.0).ok_or(VolticError::EmptyShape)?;
                let info = crate::BroadcastShape::infer(&lhs_shape, &rhs_shape).map_err(|_| {
                    VolticError::IncompatibleShapes {
                        lhs: lhs_shape,
                        rhs: rhs_shape,
                        op: stringify!($method),
                    }
                })?;
                let out_shape = info.out_shape[..info.rank].to_vec();
                let output = Var::new();
                Context::insert_shape(output.0, out_shape);
                Context::push_operation(Box::new(<$Op>::new(self.0, rhs.0, output.0, info)));
                Ok(output)
            }

            // Panic wrapper
            pub fn $method(self, rhs: Var) -> Var {
                self.$safe(rhs)
                    .expect(concat!("Var::", stringify!($method), " failed"))
            }

            // Explicit broadcast dims — skips LCS
            pub fn $bc(self, rhs: &Var, dims: &[usize]) -> Var {
                let lhs_shape = Context::shape(self.0).expect("no lhs shape");
                let rhs_shape = Context::shape(rhs.0).expect("no rhs shape");
                let info = crate::BroadcastShape::with_dims(&lhs_shape, &rhs_shape, dims)
                    .expect(concat!("Var::", stringify!($bc), " incompatible shapes"));
                let out_shape = info.out_shape[..info.rank].to_vec();
                let output = Var::new();
                Context::insert_shape(output.0, out_shape);
                Context::push_operation(Box::new(<$Op>::new(self.0, rhs.0, output.0, info)));
                output
            }
        }

        // Var OP Var (all 4 ref/value combos)
        impl std::ops::$trait<Var> for Var {
            type Output = Var;
            fn $trait_fn(self, rhs: Var) -> Var {
                self.$safe(rhs).unwrap()
            }
        }
        impl std::ops::$trait<&Var> for Var {
            type Output = Var;
            fn $trait_fn(self, rhs: &Var) -> Var {
                self.$safe(*rhs).unwrap()
            }
        }
        impl std::ops::$trait<Var> for &Var {
            type Output = Var;
            fn $trait_fn(self, rhs: Var) -> Var {
                (*self).$safe(rhs).unwrap()
            }
        }
        impl std::ops::$trait<&Var> for &Var {
            type Output = Var;
            fn $trait_fn(self, rhs: &Var) -> Var {
                (*self).$safe(*rhs).unwrap()
            }
        }

        // Var OP f32 (scalar constant)
        impl std::ops::$trait<f32> for Var {
            type Output = Var;
            fn $trait_fn(self, rhs: f32) -> Var {
                self.$safe(Self::make_scalar(rhs).unwrap()).unwrap()
            }
        }
        impl std::ops::$trait<f32> for &Var {
            type Output = Var;
            fn $trait_fn(self, rhs: f32) -> Var {
                (*self).$safe(Var::make_scalar(rhs).unwrap()).unwrap()
            }
        }
    };
}

impl_binary_op!(add, add_safe, add_bc, AddOp, Add, add);
impl_binary_op!(sub, sub_safe, sub_bc, SubOp, Sub, sub);
impl_binary_op!(mul, mul_safe, mul_bc, MulOp, Mul, mul);
impl_binary_op!(div, div_safe, div_bc, DivOp, Div, div);

impl Var {
    pub fn bias_add(self, rhs: Var) -> Var {
        let ndim = Context::shape(self.0).unwrap().len();
        self.add_bc(&rhs, &(0..ndim - 1).collect::<Vec<_>>())
    }
    pub fn bias_sub(self, rhs: Var) -> Var {
        let ndim = Context::shape(self.0).unwrap().len();
        self.sub_bc(&rhs, &(0..ndim - 1).collect::<Vec<_>>())
    }
    pub fn bias_mul(self, rhs: Var) -> Var {
        let ndim = Context::shape(self.0).unwrap().len();
        self.mul_bc(&rhs, &(0..ndim - 1).collect::<Vec<_>>())
    }
    pub fn bias_div(self, rhs: Var) -> Var {
        let ndim = Context::shape(self.0).unwrap().len();
        self.div_bc(&rhs, &(0..ndim - 1).collect::<Vec<_>>())
    }
}
