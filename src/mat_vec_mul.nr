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

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_mat_vec_mul() {
  assert(mat_vec_mul([ 1 ], [ 2 ]) == [ 2 ]);
  assert(mat_vec_mul([ 1, 2, 3, 4 ], [ 5, 6]) == [ 17, 39 ]);
  assert(mat_vec_mul([ 1, 2, 3, 4, 5, 6, 7, 8, 9 ], [ 2, 1, 3 ]) == [ 13, 31, 49 ]);
}