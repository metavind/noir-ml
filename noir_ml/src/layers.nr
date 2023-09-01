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

fn conv1d<N_CIN_IN, N_COUT_CIN_K, N_COUT, N_COUT_OUT>(
  x: [Field; N_CIN_IN],
  w: [Field; N_COUT_CIN_K],
  b: [Field; N_COUT],
  in_size: Field,
  in_channels: Field,
  out_channels: Field,
  kernel_size: Field,
  stride: Field) -> [Field; N_COUT_OUT] {

  assert(N_CIN_IN == in_channels * in_size);
  assert(N_COUT_CIN_K == kernel_size * in_channels * out_channels);
  assert(N_COUT == out_channels);

  let out_size = (in_size - kernel_size) / stride + 1;

  assert(N_COUT_OUT == out_size * out_channels);

  let mut result = [ 0; N_COUT_OUT ];

  for i in 0..out_channels {
    for j in 0..out_size {
      let mut sum = b[i];
      for k in 0..in_channels {
        for l in 0..kernel_size {
          sum += w[i * in_channels * kernel_size + k * kernel_size + l] * x[(j * stride + l) + k * in_size];
        }
      }
      result[i * out_size + j] = sum;
    }
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

#[test]
fn test_conv1d() {
  let x: [Field; 10] = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ];
  let w: [Field; 6] = [ 2, 3, 1, 2, 0, 3 ];
  let b: [Field; 1] = [1];

  let inp_size = 5;
  let in_channels = 2;
  let out_channels = 1;
  let kernel_size = 3;
  let stride = 1;

  let output = conv1d(x, w, b, inp_size, in_channels, out_channels, kernel_size, stride);

  assert(output == [ 48, 59, 70 ]);
}