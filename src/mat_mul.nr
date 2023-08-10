// Ideally, there we would support for const generic and we can write a funtion like this:
//fn mat_mul<N, M, L>(x : [ Field; N*M ], y : [ Field; M*L ]) -> [ Field; N*L ] {
//
//  let mut res = [ 0; N*L ];
//
//  for i in 0..N {
//    for j in 0..L {
//      for k in 0..M {
//        res[i*L + j] += x[i*M + k] * y[k*L + j];
//      }
//    }
//  }
//  res
//}

// Possible workarounds involve using different functions for different matrix sizes, or using 
// a struct with a large fixed size array and extra fields to indicate the actual rows and 
// columns of the matrix.

// The following code uses the second approach. It is not ideal, but it works.

use crate::utils_arr::{pad_arr, prune_arr};

global MAX_LENGTH: comptime Field = 10;

struct Matrix {
  rows: Field,
  cols: Field,
  data: [ Field; MAX_LENGTH ],
}

fn mat_mul(x: Matrix, y: Matrix) -> Matrix {
  assert(x.cols == y.rows);

  let mut result = Matrix { rows: x.rows, cols: y.cols, data: [ 0; MAX_LENGTH ] };
// Ideally, we would like to use the i, j, k for loop variables till x.rows, y.cols, x.cols respectively.
// But, since the compiler does not support const generics, the loop range must be known at compile time. 
// We therefore use MAX_LENGTH as the upper bound.
  for i in 0..MAX_LENGTH {
    if i as u64 < x.rows as u64 {
        for j in 0..MAX_LENGTH {
            if j as u64 < y.cols as u64 {
                for k in 0..MAX_LENGTH {
                    if k as u64 < x.cols as u64 {
                        result.data[i*y.cols + j] += x.data[i*x.cols + k] * y.data[k*y.cols + j];
                    }
                }
            }
        }
    }
  }
  result
}

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_mat_mul() {
  let x = Matrix { rows: 2, cols: 2, data: pad_arr([ 1, 2, 3, 4 ]) };
  let y = Matrix { rows: 2, cols: 2, data: pad_arr([ 5, 6, 7, 8 ]) };

  let res = mat_mul(x, y);

  assert(res.rows == 2);
  assert(res.cols == 2);
  assert(prune_arr(res.data) == [ 19, 22, 43, 50 ]);

}