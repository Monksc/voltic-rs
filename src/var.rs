use std::{
    ops::Add,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{
    BiasAddOp, BiasDivOp, BiasMulOp, BiasSubOp, BroadcastAddOp, BroadcastDivOp, BroadcastMulOp,
    BroadcastSubOp, Context, ExpOp, GeluOp, MatMulOp, MseOp, Op, PermuteOp, ReLUOp, ReduceMaxOp,
    ReduceMeanOp, ReduceSumOp, ScalarAddOp, ScalarDivOp, ScalarMulOp, ScalarSubOp, SigmoidOp,
    SoftmaxOp, TanhOp, VolticError, buffer_kind, errors::Result,
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

    pub fn load(&self, data: Vec<Vec<f32>>) -> Result<()> {
        Context::load(self.0, data)
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

    pub fn mat_mul(self, rhs: Var) -> Result<Self> {
        let lhs_shape = Context::shape(self.0).ok_or(VolticError::EmptyShape)?;
        let rhs_shape = Context::shape(rhs.0).ok_or(VolticError::EmptyShape)?;

        if lhs_shape.len() != 2 {
            return Err(VolticError::InvalidDimension {
                dim: lhs_shape.len(),
                ndim: 2,
            });
        }

        if rhs_shape.len() != 2 {
            return Err(VolticError::InvalidDimension {
                dim: rhs_shape.len(),
                ndim: 2,
            });
        }

        if lhs_shape[1] != rhs_shape[0] {
            return Err(VolticError::MatrixMultiplyMismatch {
                lhs: (lhs_shape[0], lhs_shape[1]),
                rhs: (rhs_shape[0], rhs_shape[1]),
            });
        }

        let output = Self::new();
        Context::insert_shape(output.0, vec![lhs_shape[0], rhs_shape[1]]);
        Context::push_operation(Box::new(MatMulOp::new(
            self.0,
            rhs.0,
            output.0,
            lhs_shape[0],
            lhs_shape[1],
            rhs_shape[1],
        )));
        Ok(output)
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
}

macro_rules! impl_scalar_op {
    ($method:ident, $safe_method:ident, $Op:ty) => {
        impl Var {
            pub fn $safe_method(&self, scalar: f32) -> Result<Self> {
                let shape = Context::shape(self.0).ok_or(VolticError::EmptyShape)?;
                let n: u32 = shape.iter().product();
                let output = Self::new();
                Context::insert_shape(output.0, shape);
                Context::push_operation(Box::new(<$Op>::new(self.0, output.0, n, scalar)));
                Ok(output)
            }

            pub fn $method(&self, scalar: f32) -> Self {
                Self::$safe_method(self, scalar).expect(concat!(
                    "Var::",
                    stringify!($method),
                    ": shape error"
                ))
            }
        }
    };
}

// Scalar ops
impl_scalar_op!(scalar_add, scalar_add_safe, ScalarAddOp);
impl_scalar_op!(scalar_sub, scalar_sub_safe, ScalarSubOp);
impl_scalar_op!(scalar_mul, scalar_mul_safe, ScalarMulOp);
impl_scalar_op!(scalar_div, scalar_div_safe, ScalarDivOp);

// Replace the impl_bias_op! macro and bias methods in var.rs with this.
// Also update imports to use BroadcastAddOp, BroadcastSubOp, BroadcastMulOp, BroadcastDivOp
// instead of BiasAddOp, BiasSubOp, BiasMulOp, BiasDivOp

macro_rules! impl_broadcast_method {
    ($method:ident, $safe_method:ident, $Op:ty) => {
        impl Var {
            pub fn $safe_method(&self, rhs: Var, axis: usize) -> Result<Self> {
                let shape = Context::shape(self.0).ok_or(VolticError::EmptyShape)?;
                let rhs_shape = Context::shape(rhs.0).ok_or(VolticError::EmptyShape)?;

                if axis >= shape.len() {
                    return Err(VolticError::InvalidDimension {
                        dim: axis,
                        ndim: shape.len(),
                    });
                }

                // rhs must match shape with the broadcast axis removed
                let expected_rhs: Vec<u32> = shape
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != axis)
                    .map(|(_, &d)| d)
                    .collect();

                if rhs_shape != expected_rhs {
                    println!("Axis: {}", axis);
                    return Err(VolticError::IncompatibleShapes {
                        lhs: shape.clone(),
                        rhs: rhs_shape,
                        op: stringify!($method),
                    });
                }

                let outer: u32 = shape[..axis].iter().product();
                let reduce: u32 = shape[axis];
                let inner: u32 = shape[axis + 1..].iter().product();

                let output = Self::new();
                Context::insert_shape(output.0, shape);
                Context::push_operation(Box::new(<$Op>::new(
                    self.0, rhs.0, output.0, outer, reduce, inner,
                )));
                Ok(output)
            }

            pub fn $method(&self, rhs: Var, axis: usize) -> Self {
                Self::$safe_method(self, rhs, axis).expect(concat!(
                    "Var::",
                    stringify!($method),
                    ": shape error"
                ))
            }
        }
    };
}

impl_broadcast_method!(broadcast_add, broadcast_add_safe, BroadcastAddOp);
impl_broadcast_method!(broadcast_sub, broadcast_sub_safe, BroadcastSubOp);
impl_broadcast_method!(broadcast_mul, broadcast_mul_safe, BroadcastMulOp);
impl_broadcast_method!(broadcast_div, broadcast_div_safe, BroadcastDivOp);

// Bias methods kept for backwards compatibility — wrap broadcast on last axis
impl Var {
    pub fn bias_add_safe(&self, bias: Var) -> Result<Self> {
        let axis = Context::shape(self.0).ok_or(VolticError::EmptyShape)?.len() - 1;
        self.broadcast_add_safe(bias, 0)
    }
    pub fn bias_sub_safe(&self, bias: Var) -> Result<Self> {
        let axis = Context::shape(self.0).ok_or(VolticError::EmptyShape)?.len() - 1;
        self.broadcast_sub_safe(bias, 0)
    }
    pub fn bias_mul_safe(&self, bias: Var) -> Result<Self> {
        let axis = Context::shape(self.0).ok_or(VolticError::EmptyShape)?.len() - 1;
        self.broadcast_mul_safe(bias, 0)
    }
    pub fn bias_div_safe(&self, bias: Var) -> Result<Self> {
        let axis = Context::shape(self.0).ok_or(VolticError::EmptyShape)?.len() - 1;
        self.broadcast_div_safe(bias, 0)
    }
    pub fn bias_add(&self, bias: Var) -> Self {
        self.bias_add_safe(bias)
            .expect("Var::bias_add: shape error")
    }
    pub fn bias_sub(&self, bias: Var) -> Self {
        self.bias_sub_safe(bias)
            .expect("Var::bias_sub: shape error")
    }
    pub fn bias_mul(&self, bias: Var) -> Self {
        self.bias_mul_safe(bias)
            .expect("Var::bias_mul: shape error")
    }
    pub fn bias_div(&self, bias: Var) -> Self {
        self.bias_div_safe(bias)
            .expect("Var::bias_div: shape error")
    }
}

macro_rules! impl_reduce_method {
    ($method:ident, $Op:ty) => {
        pub fn $method(&self, axis: usize) -> Result<Self> {
            let shape = Context::shape(self.0).ok_or(VolticError::EmptyShape)?;
            if axis >= shape.len() {
                return Err(VolticError::InvalidDimension {
                    dim: axis,
                    ndim: shape.len(),
                });
            }
            let (outer, reduce, inner) = <$Op>::compute_dims(&shape, axis);
            let output = Self::new();
            Context::insert_shape(output.0, <$Op>::infer_output_shape(&shape, axis));
            Context::push_operation(Box::new(<$Op>::new(self.0, output.0, outer, reduce, inner)));
            Ok(output)
        }
    };
}

impl Var {
    impl_reduce_method!(reduce_sum, ReduceSumOp);
    impl_reduce_method!(reduce_max, ReduceMaxOp);
    impl_reduce_method!(reduce_mean, ReduceMeanOp);
}
