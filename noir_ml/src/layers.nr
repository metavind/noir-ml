//! Layer Operations for Neural Networks in Noir
//!
//! This module provides implementations for different neural network layers.
//! Currently, the available layers are:
//! - Fully connected (fc)

/// Fully Connected (fc) Layer.
///
/// # Example
/// ```
/// let input = [1, 2];
/// let weights = [2, 3, 4, 5];
/// let biases = [1, 2];
/// let output = fc(input, weights, biases);
/// assert_eq!(output, [9, 16]);
/// ```
fn fc<N_IN, N_IN_OUT, N_OUT>(x: [ Field; N_IN], w: [Field; N_IN_OUT], b: [Field; N_OUT]) -> [Field; N_OUT] {
  assert(N_IN_OUT == N_IN * N_OUT);
  
  let mut result = [ 0; N_OUT ];

  for i in 0..N_OUT {
    for j in 0..N_IN {
      result[i] += w[i*N_IN + j] * x[j];
    }
    result[i] += b[i];
  }
  result
}

////////////////////
//     TESTS      //
////////////////////

#[test]
fn test_fc() {
  assert(fc([ 1, 2, 3 ], [ 4, 5, 6, 7, 8, 9 ], [ 10, 11 ]) == [ 42, 61 ]);
}