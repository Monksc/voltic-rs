use std::{
    ops::Add,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{
    BiasAddOp, BiasDivOp, BiasMulOp, BiasSubOp, Context, GeluOp, MatMulOp, MseOp, Op, ReLUOp,
    ScalarAddOp, ScalarDivOp, ScalarMulOp, ScalarSubOp, SigmoidOp, TanhOp, VolticError,
    buffer_kind, errors::Result,
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
}

macro_rules! impl_bias_op {
    ($method:ident, $safe_method:ident, $Op:ty) => {
        impl Var {
            pub fn $safe_method(&self, bias: Var) -> Result<Self> {
                let shape = Context::shape(self.0).ok_or(VolticError::EmptyShape)?;
                let bias_shape = Context::shape(bias.0).ok_or(VolticError::EmptyShape)?;

                // bias must match cols
                if bias_shape != vec![shape[1]] {
                    return Err(VolticError::IncompatibleShapes {
                        lhs: shape.clone(),
                        rhs: bias_shape,
                        op: stringify!($method),
                    });
                }

                let rows = shape[0];
                let cols = shape[1];
                let output = Self::new();
                Context::insert_shape(output.0, shape);
                Context::push_operation(Box::new(<$Op>::new(self.0, bias.0, output.0, rows, cols)));
                Ok(output)
            }

            pub fn $method(&self, bias: Var) -> Self {
                Self::$safe_method(self, bias).expect(concat!(
                    "Var::",
                    stringify!($method),
                    ": shape error"
                ))
            }
        }
    };
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

// Bias ops
impl_bias_op!(bias_add, bias_add_safe, BiasAddOp);
impl_bias_op!(bias_sub, bias_sub_safe, BiasSubOp);
impl_bias_op!(bias_mul, bias_mul_safe, BiasMulOp);
impl_bias_op!(bias_div, bias_div_safe, BiasDivOp);

// Scalar ops
impl_scalar_op!(scalar_add, scalar_add_safe, ScalarAddOp);
impl_scalar_op!(scalar_sub, scalar_sub_safe, ScalarSubOp);
impl_scalar_op!(scalar_mul, scalar_mul_safe, ScalarMulOp);
impl_scalar_op!(scalar_div, scalar_div_safe, ScalarDivOp);
