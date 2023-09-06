//! Activation Functions for Neural Networks in Noir
//!
//! This module contains activation functions commonly used in neural networks.
//! The available functions are:
//! - ReLU

use crate::utils::is_positive;

/// ReLU (Rectified Linear Unit) Activation Function.
/// y = max(0, x)
///
/// # Example
/// ```
/// let values = [-5, 2, 0];
/// let activated = relu(values);
/// assert_eq!(activated, [0, 2, 0]);
/// ```
fn relu<N>(values: [ Field; N ]) -> [ Field; N ] {
  let mut result = [0; N];

  for i in 0..N {
    if is_positive(values[i]) {
      result[i] = values[i];
    }
  }
  result
}

fn poly<N>(values: [ Field; N ]) -> [ Field; N ] {
  let mut result = [0; N];

  for i in 0..N {
    result[i] = values[i] * values[i] + values[i];
  }
  result
}

////////////////////
//     TESTS      //
////////////////////

#[test]
fn test_relu() {
  assert(relu([ -1, 0, 1]) == [ 0, 0, 1 ]);
  let comp_constant = 10944121435919637611123202872628637544274182200208017171849102093287904247808;
  assert(relu([ comp_constant - 1, comp_constant, comp_constant + 1 ]) == [ comp_constant - 1, comp_constant, 0]);
}

#[test]
fn test_poly() {
  assert(poly([ -1, 0, 1, 2]) == [ 0, 0, 2, 6 ]);
}