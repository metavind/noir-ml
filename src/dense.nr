fn dense<N_IN, N_IN_OUT, N_OUT>(x: [ Field; N_IN], w: [Field; N_IN_OUT], b: [Field; N_OUT]) -> [Field; N_OUT] {
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
fn test_dense() {
  assert(dense([ 1, 2, 3 ], [ 4, 5, 6, 7, 8, 9 ], [ 10, 11 ]) == [ 42, 61 ]);
}