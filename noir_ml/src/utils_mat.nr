fn mat_vec_mul<N_M, M, N>(m: [ Field; N_M], v: [Field; M]) -> [Field; N] {

  assert(N_M == N * M);
  
  let mut result = [ 0; N ];

  for i in 0..N {
    for j in 0..M {
      result[i] += m[i*M + j] * v[j];
    }
  }
  result
}

struct Matrix<N> {
  rows: Field,
  cols: Field,
  data: [ Field; N ],
}

fn mat_mul<N, M, L>(x: Matrix<N>, y: Matrix<M>) -> Matrix<L> {

  assert(x.cols == y.rows);

  let mut result = [ 0; L ];

  for i in 0..x.rows {
    for j in 0..y.cols {
      for k in 0..x.cols {
        result[i*y.cols + j] += x.data[i*x.cols + k] * y.data[k*y.cols + j];
      }
    }
  }
  Matrix { rows: x.rows, cols: y.cols, data: result }
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
fn test_mat_mul_1() {
  let x = Matrix { rows: 2, cols: 2, data: [ 1, 2, 3, 4 ] };
  let y = Matrix { rows: 2, cols: 2, data: [ 5, 6, 7, 8 ] };

  let res = mat_mul(x, y);

  assert(res.rows == 2);
  assert(res.cols == 2);
  assert(res.data == [ 19, 22, 43, 50 ]);
}

#[test]
fn test_mat_mul_2() {
  let x = Matrix { rows: 5, cols: 2, data: [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ] };
  let y = Matrix { rows: 2, cols: 5, data: [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ] };

  let res = mat_mul(x, y);

  assert(res.rows == 5);
  assert(res.cols == 5);
  assert(res.data == [ 13, 16, 19, 22, 25, 27, 34, 41, 48, 55, 41, 52, 63, 74, 85, 55, 70, 85, 100, 115, 69, 88, 107, 126, 145 ]);
}

#[test]
fn test_mat_mul_3() {
  let x = Matrix { rows: 2, cols: 5, data: [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ] };
  let y = Matrix { rows: 5, cols: 2, data: [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ] };

  let res = mat_mul(x, y);

  assert(res.rows == 2);
  assert(res.cols == 2);
  assert(res.data == [ 95, 110, 220, 260 ]);
}