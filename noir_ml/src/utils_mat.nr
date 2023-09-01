fn mat_vec_mul<N_IN, N_IN_OUT, N_OUT>(m: [ Field; N_IN_OUT], v: [Field; N_IN]) -> [Field; N_OUT] {
  assert(N_IN_OUT == N_IN * N_OUT);
  
  let mut result = [ 0; N_OUT ];

  for i in 0..N_OUT {
    for j in 0..N_IN {
      result[i] += m[i*N_IN + j] * v[j];
    }
  }
  result
}

use crate::utils_arr::{pad_arr, prune_arr};
global MAX_LENGTH: Field = 10;

struct Matrix {
  rows: Field,
  cols: Field,
  data: [ Field; MAX_LENGTH ],
}

fn mat_mul(x: Matrix, y: Matrix) -> Matrix {
  assert(x.cols == y.rows);

  let mut result = Matrix { rows: x.rows, cols: y.cols, data: [ 0; MAX_LENGTH ] };

  for i in 0..x.rows {
    for j in 0..y.cols {
      for k in 0..x.cols {
        result.data[i*y.cols + j] += x.data[i*x.cols + k] * y.data[k*y.cols + j];
      }
    }
  }
  result
}



////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_mat_vec_mul() {
  assert(mat_vec_mul([ 1 ], [ 2 ]) == [ 2 ]);
  assert(mat_vec_mul([ 1, 2, 3, 4 ], [ 5, 6]) == [ 17, 39 ]);
  assert(mat_vec_mul([ 1, 2, 3, 4, 5, 6, 7, 8, 9 ], [ 2, 1, 3 ]) == [ 13, 31, 49 ]);
}

#[test]
fn test_mat_mul() {
  let x = Matrix { rows: 2, cols: 2, data: pad_arr([ 1, 2, 3, 4 ]) };
  let y = Matrix { rows: 2, cols: 2, data: pad_arr([ 5, 6, 7, 8 ]) };

  let res = mat_mul(x, y);

  assert(res.rows == 2);
  assert(res.cols == 2);
  assert(prune_arr(res.data) == [ 19, 22, 43, 50 ]);

}