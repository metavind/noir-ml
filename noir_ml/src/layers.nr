//! Layer Operations for Neural Networks in Noir
//!
//! This module provides implementations for different neural network layers.
//! Currently, the available layers are:
//! - Fully connected (fc)

use crate::utils::is_greater_than_signed;

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
  assert(N_COUT_CIN_K == out_channels * in_channels * kernel_size);
  assert(N_COUT == out_channels);

  let out_size = (in_size - kernel_size) / stride + 1;

  assert(N_COUT_OUT == out_channels * out_size);

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

fn conv2d_nonsq<N_CIN_HIN_WIN, N_COUT_CIN_HK_WK, N_COUT, N_COUT_HOUT_WOUT>(
  x: [Field; N_CIN_HIN_WIN],
  w: [Field; N_COUT_CIN_HK_WK],
  b: [Field; N_COUT],
  in_height: Field,
  in_width: Field,
  in_channels: Field,
  out_channels: Field,
  kernel_height: Field,
  kernel_width: Field,
  stride: Field) -> [Field; N_COUT_HOUT_WOUT] {

  assert(N_CIN_HIN_WIN == in_channels * in_height * in_width);
  assert(N_COUT_CIN_HK_WK == out_channels * in_channels * kernel_height * kernel_width);
  assert(N_COUT == out_channels);

  let out_height = (in_height - kernel_height) / stride + 1;
  let out_width = (in_width - kernel_width) / stride + 1;

  assert(N_COUT_HOUT_WOUT == out_channels * out_height * out_width);

  let mut result = [ 0; N_COUT_HOUT_WOUT ];

  for i in 0..out_channels {
    for j in 0..out_height {
      for k in 0..out_width {
        let mut sum = b[i];
        for l in 0..in_channels {
          for m in 0..kernel_height {
            for n in 0..kernel_width {
              sum += w[i * in_channels * kernel_height * kernel_width + l * kernel_height * kernel_width + m * kernel_width + n] * x[(j * stride + m) * in_width + k * stride + n + l * in_height * in_width];
            }
          }
        }
        result[i * out_height * out_width + j * out_width + k] = sum;
      }
    }
  }
  result
}

fn conv2d<N_CIN_IN, N_COUT_CIN_K, N_COUT, N_COUT_OUT>(
  x: [Field; N_CIN_IN],
  w: [Field; N_COUT_CIN_K],
  b: [Field; N_COUT],
  in_size: Field,
  in_channels: Field,
  out_channels: Field,
  kernel_size: Field,
  stride: Field) -> [Field; N_COUT_OUT] {
    conv2d_nonsq(x, w, b, in_size, in_size, in_channels, out_channels, kernel_size, kernel_size, stride)
  }

fn max_pool_1d<N_CIN_IN, N_COUT_OUT>(
    x: [Field; N_CIN_IN],
    in_size: Field,
    in_channels: Field,
    kernel_size: Field) -> [Field; N_COUT_OUT] {

    assert(N_CIN_IN == in_channels * in_size);

    let out_size = in_size / kernel_size;

    assert(N_COUT_OUT == in_channels * out_size);

    let mut result = [0; N_COUT_OUT];

    for i in 0..in_channels {
        for j in 0..out_size {
            let mut max_val = x[i * in_size + j * kernel_size];
            for k in 1..kernel_size {
                let val = x[i * in_size + j * kernel_size + k];
                if is_greater_than_signed(val, max_val) {
                    max_val = val;
                }
            }
            result[i * out_size + j] = max_val;
        }
    }
    result
}

fn avg_pool_1d<N_CIN_IN, N_COUT_OUT>(
    x: [Field; N_CIN_IN],
    in_size: Field,
    in_channels: Field,
    kernel_size: Field) -> [Field; N_COUT_OUT] {

    assert(N_CIN_IN == in_channels * in_size);

    let out_size = in_size / kernel_size;

    assert(N_COUT_OUT == in_channels * out_size);

    let mut result = [0; N_COUT_OUT];

    for i in 0..in_channels {
        for j in 0..out_size {
            let mut sum_val = 0;
            for k in 0..kernel_size {
                sum_val += x[i * in_size + j * kernel_size + k];
            }
            result[i * out_size + j] = sum_val/kernel_size;
        }
    }
    result
}

fn sum_pool_1d<N_CIN_IN, N_COUT_OUT>(
    x: [Field; N_CIN_IN],
    in_size: Field,
    in_channels: Field,
    kernel_size: Field) -> [Field; N_COUT_OUT] {

    assert(N_CIN_IN == in_channels * in_size);

    let out_size = in_size / kernel_size;

    assert(N_COUT_OUT == in_channels * out_size);

    let mut result = [0; N_COUT_OUT];

    for i in 0..in_channels {
        for j in 0..out_size {
            let mut sum_val = 0;
            for k in 0..kernel_size {
                sum_val += x[i * in_size + j * kernel_size + k];
            }
            result[i * out_size + j] = sum_val;
        }
    }
    result
}

fn max_pool_2d_nonsq<N_CIN_HIN_WIN, N_COUT_HOUT_WOUT>(
    x: [Field; N_CIN_HIN_WIN],
    in_height: Field,
    in_width: Field,
    in_channels: Field,
    kernel_height: Field,
    kernel_width: Field) -> [Field; N_COUT_HOUT_WOUT] {

    assert(N_CIN_HIN_WIN == in_channels * in_height * in_width);

    let out_height = in_height / kernel_height;
    let out_width = in_width / kernel_width;

    assert(N_COUT_HOUT_WOUT == in_channels * out_height * out_width);

    let mut result = [0; N_COUT_HOUT_WOUT];

    for i in 0..in_channels {
        for j in 0..out_height {
            for k in 0..out_width {
                let mut max_val = x[i * in_height * in_width + j * kernel_height * in_width + k * kernel_width];
                for l in 0..kernel_height {
                    for m in 0..kernel_width {
                        let val = x[i * in_height * in_width + (j * kernel_height + l) * in_width + k * kernel_width + m];
                        if is_greater_than_signed(val, max_val) {
                            max_val = val;
                        }
                    }
                }
                result[i * out_height * out_width + j * out_width + k] = max_val;
            }
        }
    }
    result
}

fn max_pool_2d<N_CIN_IN, N_COUT_OUT>(
    x: [Field; N_CIN_IN],
    in_size: Field,
    in_channels: Field,
    kernel_size: Field) -> [Field; N_COUT_OUT] {

    max_pool_2d_nonsq(x, in_size, in_size, in_channels, kernel_size, kernel_size)
}

fn global_max_pool_2d_nonsq<N_CIN_HIN_WIN, N_COUT>(
    x: [Field; N_CIN_HIN_WIN],
    in_height: Field,
    in_width: Field,
    in_channels: Field) -> [Field; N_COUT] {

    assert(N_CIN_HIN_WIN == in_channels * in_height * in_width);

    let mut result = [0; N_COUT];

    for i in 0..in_channels {
        let mut max_val = x[i * in_height * in_width];
        for j in 0..in_height {
            for k in 0..in_width {
                let val = x[i * in_height * in_width + j * in_width + k];
                if is_greater_than_signed(val, max_val) {
                    max_val = val;
                }
            }
        }
        result[i] = max_val;
    }
    result
}

fn global_max_pool_2d<N_CIN_IN, N_COUT>(
    x: [Field; N_CIN_IN],
    in_size: Field,
    in_channels: Field) -> [Field; N_COUT] {

    global_max_pool_2d_nonsq(x, in_size, in_size, in_channels)
}

fn avg_pool_2d_nonsq<N_CIN_HIN_WIN, N_COUT_HOUT_WOUT>(
    x: [Field; N_CIN_HIN_WIN],
    in_height: Field,
    in_width: Field,
    in_channels: Field,
    kernel_height: Field,
    kernel_width: Field) -> [Field; N_COUT_HOUT_WOUT] {

    assert(N_CIN_HIN_WIN == in_channels * in_height * in_width);

    let out_height = in_height / kernel_height;
    let out_width = in_width / kernel_width;

    assert(N_COUT_HOUT_WOUT == in_channels * out_height * out_width);

    let mut result = [0; N_COUT_HOUT_WOUT];

    for i in 0..in_channels {
        for j in 0..out_height {
            for k in 0..out_width {
                let mut sum_val = 0;
                for l in 0..kernel_height {
                    for m in 0..kernel_width {
                        sum_val += x[i * in_height * in_width + (j * kernel_height + l) * in_width + k * kernel_width + m];
                    }
                }
                result[i * out_height * out_width + j * out_width + k] = sum_val / (kernel_height * kernel_width);
            }
        }
    }
    result
}

fn avg_pool_2d<N_CIN_IN, N_COUT_OUT>(
    x: [Field; N_CIN_IN],
    in_size: Field,
    in_channels: Field,
    kernel_size: Field) -> [Field; N_COUT_OUT] {

    avg_pool_2d_nonsq(x, in_size, in_size, in_channels, kernel_size, kernel_size)
}

fn sum_pool_2d_nonsq<N_CIN_HIN_WIN, N_COUT_HOUT_WOUT>(
    x: [Field; N_CIN_HIN_WIN],
    in_height: Field,
    in_width: Field,
    in_channels: Field,
    kernel_height: Field,
    kernel_width: Field) -> [Field; N_COUT_HOUT_WOUT] {

    assert(N_CIN_HIN_WIN == in_channels * in_height * in_width);

    let out_height = in_height / kernel_height;
    let out_width = in_width / kernel_width;

    assert(N_COUT_HOUT_WOUT == in_channels * out_height * out_width);

    let mut result = [0; N_COUT_HOUT_WOUT];

    for i in 0..in_channels {
        for j in 0..out_height {
            for k in 0..out_width {
                let mut sum_val = 0;
                for l in 0..kernel_height {
                    for m in 0..kernel_width {
                        sum_val += x[i * in_height * in_width + (j * kernel_height + l) * in_width + k * kernel_width + m];
                    }
                }
                result[i * out_height * out_width + j * out_width + k] = sum_val;
            }
        }
    }
    result
}

fn sum_pool_2d<N_CIN_IN, N_COUT_OUT>(
    x: [Field; N_CIN_IN],
    in_size: Field,
    in_channels: Field,
    kernel_size: Field) -> [Field; N_COUT_OUT] {

    sum_pool_2d_nonsq(x, in_size, in_size, in_channels, kernel_size, kernel_size)
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

#[test]
fn test_conv2d() {
  let x: [Field; 9] = [
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
];
  let w: [Field; 4] = [
    1, 2,
    3, 4
];
  let b: [Field; 1] = [1];

  let in_size = 3;
  let in_channels = 1;
  let out_channels = 1;
  let kernel_size = 2;
  let stride = 1;

  let output = conv2d(x, w, b, in_size, in_channels, out_channels, kernel_size, stride);

  assert(output == [ 38, 48, 68, 78 ]);
}

#[test]
fn test_conv2d_nonsq() {
  let x: [Field; 40] = [
    1, 2, 3, 4, 5,
    4, 5, 6, 7, 8,
    3, 4, 5, 6, 7,
    2, 3, 4, 5, 6,

    5, 4, 3, 2, 1,
    8, 7, 6, 5, 4,
    7, 6, 5, 4, 3,
    6, 5, 4, 3, 2
];
  let w: [Field; 36] = [
    1, 0,
    0, 1,
    0, 1,

    1, 0,
    0, 1,
    1, 0,

    1, 0,
    0, 1,
    1, 1,

    1, 1,
    0, 1,
    0, 1,

    1, 1,
    0, 1,
    1, 0,

    1, 1,
    1, 0,
    1, 0,
];
  let b: [Field; 3] = [1, 2, 3];

  let in_height = 4;
  let in_width = 5;
  let in_channels = 2;
  let out_channels = 3;
  let kernel_height = 3;
  let kernel_width = 2;
  let stride = 2;

  let output = conv2d_nonsq(x, w, b, in_height, in_width, in_channels, out_channels, kernel_height, kernel_width, stride);

  assert(output == [ 30, 30, 37, 37, 38, 38 ]);
}

#[test]
fn test_max_pool_1d() {
    let x = [1, 2, 3, 4, 5, 6, 7, 8];
    let output = max_pool_1d(x, 4, 2, 2);
    assert(output == [2, 4, 6, 8]);
}

#[test]
fn test_avg_pool_1d() {
    let x = [1, 3, 2, 4, 5, 7, 6, 8];
    let output = avg_pool_1d(x, 4, 2, 2);
    assert(output == [2, 3, 6, 7]);
}

#[test]
fn test_sum_pool_1d() {
    let x = [1, 3, 2, 4, 5, 7, 6, 8];
    let output = sum_pool_1d(x, 4, 2, 2);
    assert(output == [4, 6, 12, 14]);
}

#[test]
fn test_max_pool_2d() {
    let x = [
        1, 2,
        3, 4,

        5, 6,
        7, 8
];

    let output = max_pool_2d(x, 2, 2, 2);

    assert(output == [4, 8]);
}

#[test]
fn test_max_pool_2d_nonsq() {
    let x = [
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,

        0, 2, 4,
        6, 8, 10,
        12, 14, 16,
        18, 20, 22
];

    let output = max_pool_2d_nonsq(x, 4, 3, 2, 2, 3);

    assert(output == [6, 12, 10, 22]);
}

#[test]
fn test_global_max_pool_2d() {
    let x = [
        1, 2,
        3, 4,

        5, 6,
        7, 8
];

    let output = global_max_pool_2d(x, 2, 2);

    assert(output == [4, 8]);
}

#[test]
fn test_global_max_pool_2d_nonsq() {
    let x = [
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,

        0, 2, 4,
        6, 8, 10,
        12, 14, 16,
        18, 20, 22
];

    let output = global_max_pool_2d_nonsq(x, 4, 3, 2);

    assert(output == [12, 22]);
}

#[test]
fn test_avg_pool_2d() {
    let x = [
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,

        0, 2, 4,
        4, 2, 0,
        2, 0, 4
];

    let output = avg_pool_2d(x, 3, 2, 3);

    assert(output == [5, 2]);
}

#[test]
fn test_avg_pool_2d_nonsq() {
    let x = [
        1, 2, 3,
        2, 4, 6,
        7, 8, 9,
        9, 8, 7,

        0, 2, 4,
        6, 8, 10,
        12, 14, 16,
        18, 20, 22
];

    let output = avg_pool_2d_nonsq(x, 4, 3, 2, 2, 3);
    assert(output == [3, 8, 5, 17]);
}

#[test]
fn test_sum_pool_2d() {
    let x = [
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,

        0, 2, 4,
        4, 2, 0,
        2, 0, 4
];

    let output = sum_pool_2d(x, 3, 2, 3);

    assert(output == [45, 18]);
}

#[test]
fn test_sum_pool_2d_nonsq() {
    let x = [
        1, 2, 3,
        2, 4, 6,
        7, 8, 9,
        9, 8, 7,

        0, 2, 4,
        6, 8, 10,
        12, 14, 16,
        18, 20, 22
];

    let output = sum_pool_2d_nonsq(x, 4, 3, 2, 2, 3);
    assert(output == [18, 48, 30, 102]);
}