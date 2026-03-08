use std::fmt;

pub type Result<T> = std::result::Result<T, VolticError>;

#[derive(Debug, Clone)]
pub enum VolticError {
    // Shape / Dimension errors
    ShapeMismatch {
        expected: Vec<u32>,
        got: Vec<u32>,
    },
    IncompatibleShapes {
        lhs: Vec<u32>,
        rhs: Vec<u32>,
        op: &'static str,
    },
    InvalidDimension {
        dim: usize,
        ndim: usize,
    },
    EmptyShape,

    // Matrix errors
    MatrixNotSquare {
        rows: u32,
        cols: u32,
    },
    MatrixMultiplyMismatch {
        lhs: (u32, u32),
        rhs: (u32, u32),
    },

    // Index / Bounds errors
    IndexOutOfBounds {
        index: Vec<u32>,
        shape: Vec<u32>,
    },

    // GPU / WGPU errors
    GpuNotAvailable,
    GpuBufferError(String),
    ShaderCompileError(String),

    // General
    NotImplemented(&'static str),
    Internal(String),
}

impl fmt::Display for VolticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {expected:?}, got {got:?}")
            }

            Self::IncompatibleShapes { lhs, rhs, op } => {
                write!(f, "Incompatible shapes for '{op}': {lhs:?} and {rhs:?}")
            }

            Self::InvalidDimension { dim, ndim } => write!(
                f,
                "Dimension {dim} out of range for tensor with {ndim} dims"
            ),

            Self::EmptyShape => write!(f, "Shape cannot be empty"),

            Self::MatrixNotSquare { rows, cols } => {
                write!(f, "Matrix must be square, got ({rows}x{cols})")
            }

            Self::MatrixMultiplyMismatch { lhs, rhs } => write!(
                f,
                "Cannot multiply ({}x{}) by ({}x{}): inner dimensions must match",
                lhs.0, lhs.1, rhs.0, rhs.1
            ),

            Self::IndexOutOfBounds { index, shape } => {
                write!(f, "Index {index:?} out of bounds for shape {shape:?}")
            }

            Self::GpuNotAvailable => write!(f, "No compatible GPU adapter found"),

            Self::GpuBufferError(msg) => write!(f, "GPU buffer error: {msg}"),

            Self::ShaderCompileError(msg) => write!(f, "Shader compilation failed: {msg}"),

            Self::NotImplemented(feature) => write!(f, "Not yet implemented: {feature}"),

            Self::Internal(msg) => write!(f, "Internal error: {msg}"),
        }
    }
}

impl std::error::Error for VolticError {}

// Convenience From impls
impl From<wgpu::Error> for VolticError {
    fn from(e: wgpu::Error) -> Self {
        Self::GpuBufferError(e.to_string())
    }
}

impl From<String> for VolticError {
    fn from(s: String) -> Self {
        Self::Internal(s)
    }
}
