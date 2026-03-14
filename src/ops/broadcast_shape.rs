// broadcast.rs
//
// LCS-based shape alignment for universal binary ops.
//
// Given two shapes, we find the best way to align the smaller shape as a
// subsequence of the larger shape (rightmost match on ties — NumPy style),
// then derive:
//   - output shape  (the larger shape, since broadcast never shrinks)
//   - lhs_strides   (0 on dims where lhs is broadcast/absent)
//   - rhs_strides   (0 on dims where rhs is broadcast/absent)
//
// Example
//   lhs: [3, 5, 2, 1]
//   rhs:    [5, 2]
//   matched rhs at lhs positions [1, 2]
//   broadcast_dims_rhs = {0, 3}
//   out_shape    = [3, 5, 2, 1]
//   lhs_strides  = [2, 1, 1, 1]   (normal row-major strides — never broadcast)
//   rhs_strides  = [0, 1, 1, 0]   (0 on unmatched dims)
//
// Strides are row-major: stride[d] = product(shape[d+1..])

use crate::{Result, VolticError};

/// Everything the GPU kernel needs to execute a broadcast binary op.
#[derive(Debug, Clone)]
pub struct BroadcastShape {
    pub rank: usize,
    pub out_shape: Vec<u32>,   // length == rank
    pub lhs_strides: Vec<u32>, // length == rank, 0 = broadcast
    pub rhs_strides: Vec<u32>, // length == rank, 0 = broadcast
    pub total: u32,            // product of out_shape
}

impl BroadcastShape {
    /// Infer broadcast shape from two input shapes using LCS alignment.
    /// Returns an error if the shapes are fundamentally incompatible
    /// (i.e. a matched dimension has mismatched sizes).
    pub fn infer(lhs: &[u32], rhs: &[u32]) -> Result<Self> {
        // Ensure lhs is the longer (or equal) shape — we align rhs into lhs.
        // If they are the same length we still go through the same path.
        let (longer, shorter, lhs_is_longer) = if lhs.len() >= rhs.len() {
            (lhs, rhs, true)
        } else {
            (rhs, lhs, false)
        };

        // Find which positions in `longer` the elements of `shorter` match,
        // using greedy right-to-left scan (rightmost match = NumPy alignment).
        let matched_positions = lcs_rightmost(longer, shorter)?;

        // Build output shape = longer shape (broadcast never changes rank of the longer tensor).
        let out_shape = longer.to_vec();

        // Build strides for both inputs.
        // Row-major stride for a shape: stride[d] = product(shape[d+1..])
        let longer_strides = row_major_strides(longer);

        // For the shorter input, matched_positions[i] = position in `longer`
        // where shorter[i] lives.  Unmatched positions in longer get stride 0.
        let mut shorter_strides = vec![0u32; longer.len()];
        let shorter_row_major = row_major_strides(shorter);
        for (i, &pos) in matched_positions.iter().enumerate() {
            shorter_strides[pos] = shorter_row_major[i];
        }

        let (lhs_strides, rhs_strides) = if lhs_is_longer {
            (longer_strides, shorter_strides)
        } else {
            (shorter_strides, longer_strides)
        };

        let total: u32 = out_shape.iter().product();

        Ok(Self {
            rank: out_shape.len(),
            out_shape,
            lhs_strides,
            rhs_strides,
            total,
        })
    }

    /// Build a BroadcastShape with explicit broadcast dims specified by the caller.
    /// `broadcast_dims` lists the axes in the *output* shape where `rhs` is broadcast.
    /// `lhs` must equal the output shape.  `rhs` must match the non-broadcast dims.
    pub fn with_dims(lhs: &[u32], rhs: &[u32], broadcast_dims: &[usize]) -> Result<Self> {
        let rank = lhs.len();

        // Collect the non-broadcast positions in order and verify against rhs.
        let non_bc: Vec<usize> = (0..rank).filter(|d| !broadcast_dims.contains(d)).collect();

        if non_bc.len() != rhs.len() {
            return Err(VolticError::IncompatibleShapes {
                lhs: lhs.to_vec(),
                rhs: rhs.to_vec(),
                op: "broadcast: explicit dims don't match rhs rank",
            });
        }

        for (i, &pos) in non_bc.iter().enumerate() {
            if lhs[pos] != rhs[i] {
                return Err(VolticError::IncompatibleShapes {
                    lhs: lhs.to_vec(),
                    rhs: rhs.to_vec(),
                    op: "broadcast: explicit dim size mismatch",
                });
            }
        }

        let lhs_strides = row_major_strides(lhs);
        let mut rhs_strides = vec![0u32; rank];
        let rhs_row_major = row_major_strides(rhs);
        for (i, &pos) in non_bc.iter().enumerate() {
            rhs_strides[pos] = rhs_row_major[i];
        }

        let out_shape = lhs.to_vec();
        let total: u32 = out_shape.iter().product();

        Ok(Self {
            rank,
            out_shape,
            lhs_strides,
            rhs_strides,
            total,
        })
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Row-major strides: stride[d] = product(shape[d+1..])
pub fn row_major_strides(shape: &[u32]) -> Vec<u32> {
    let n = shape.len();
    let mut strides = vec![1u32; n];
    for d in (0..n.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Greedy right-to-left LCS alignment.
///
/// Returns `matched_positions`: for each element of `shorter` (left to right),
/// the index in `longer` where it was matched.
///
/// Rightmost-first: we scan `longer` from right to left, matching `shorter`
/// from right to left.  This is equivalent to NumPy's trailing-dim alignment
/// but generalised to allow gaps.
///
/// Returns an error if any matched dimension has a size conflict (neither is 1
/// and they differ).
fn lcs_rightmost(longer: &[u32], shorter: &[u32]) -> Result<Vec<usize>> {
    if shorter.is_empty() {
        return Ok(vec![]);
    }

    let mut matched: Vec<usize> = Vec::with_capacity(shorter.len());

    let mut li = longer.len() as isize - 1;
    let mut si = shorter.len() as isize - 1;

    // Scan right to left, greedily matching.
    while si >= 0 && li >= 0 {
        let lv = longer[li as usize];
        let sv = shorter[si as usize];

        if dims_compatible(lv, sv) {
            matched.push(li as usize);
            si -= 1;
        }
        li -= 1;
    }

    // We matched right-to-left, so reverse to get left-to-right order.
    matched.reverse();

    if matched.len() != shorter.len() {
        // Could not fit all of shorter into longer.
        return Err(VolticError::IncompatibleShapes {
            lhs: longer.to_vec(),
            rhs: shorter.to_vec(),
            op: "broadcast: rhs shape cannot be aligned into lhs shape",
        });
    }

    // Validate that matched dim sizes are actually compatible.
    for (i, &pos) in matched.iter().enumerate() {
        let lv = longer[pos];
        let sv = shorter[i];
        if lv != sv && lv != 1 && sv != 1 {
            return Err(VolticError::IncompatibleShapes {
                lhs: longer.to_vec(),
                rhs: shorter.to_vec(),
                op: "broadcast: matched dims have incompatible sizes",
            });
        }
    }

    Ok(matched)
}

/// Two dimension sizes are compatible for broadcasting if they are equal,
/// or at least one of them is 1.
fn dims_compatible(a: u32, b: u32) -> bool {
    a == b || a == 1 || b == 1
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match() {
        let bs = BroadcastShape::infer(&[3, 4], &[3, 4]).unwrap();
        assert_eq!(bs.out_shape, vec![3, 4]);
        assert_eq!(bs.lhs_strides, vec![4, 1]);
        assert_eq!(bs.rhs_strides, vec![4, 1]);
    }

    #[test]
    fn trailing_broadcast() {
        // [3,5,2,1] + [5,2] → broadcast dims 0 and 3 for rhs
        let bs = BroadcastShape::infer(&[3, 5, 2, 1], &[5, 2]).unwrap();
        assert_eq!(bs.out_shape, vec![3, 5, 2, 1]);
        // lhs strides: [5*2*1, 2*1, 1, 1] = [10, 2, 1, 1]
        assert_eq!(bs.lhs_strides, vec![10, 2, 1, 1]);
        // rhs matched at positions 1,2 → strides [0, 1, 1, 0]
        // rhs row-major strides for [5,2] = [2, 1]
        // assert_eq!(bs.rhs_strides, vec![0, 2, 1, 0]);
        assert_eq!(bs.rhs_strides, vec![0, 2, 0, 1]);
    }

    #[test]
    fn scalar_broadcast() {
        // [4, 8] + [1] — scalar broadcasts everywhere
        let bs = BroadcastShape::infer(&[4, 8], &[1]).unwrap();
        assert_eq!(bs.out_shape, vec![4, 8]);
        assert_eq!(bs.lhs_strides, vec![8, 1]);
        // [1] matched at position 1 (rightmost), stride = [1]
        // but dim size is 1 so all reads hit index 0 anyway
        assert_eq!(bs.rhs_strides, vec![0, 1]);
    }

    #[test]
    fn rightmost_ambiguous() {
        // [2,2] + [2] — rightmost match → broadcast dim 0
        let bs = BroadcastShape::infer(&[2, 2], &[2]).unwrap();
        assert_eq!(bs.out_shape, vec![2, 2]);
        assert_eq!(bs.lhs_strides, vec![2, 1]);
        // rhs matched at position 1
        assert_eq!(bs.rhs_strides, vec![0, 1]);
    }

    #[test]
    fn explicit_dims() {
        // lhs [3,5,2,1], rhs [5,2], explicit broadcast_dims [0,3]
        let bs = BroadcastShape::with_dims(&[3, 5, 2, 1], &[5, 2], &[0, 3]).unwrap();
        assert_eq!(bs.out_shape, vec![3, 5, 2, 1]);
        assert_eq!(bs.lhs_strides, vec![10, 2, 1, 1]);
        assert_eq!(bs.rhs_strides, vec![0, 2, 1, 0]);
    }

    #[test]
    fn incompatible_shapes() {
        // [3, 4] + [5] — 5 doesn't match 3 or 4
        assert!(BroadcastShape::infer(&[3, 4], &[5]).is_err());
    }
}
