fn dot_prod<N>(x : [ Field; N ], y : [ Field; N ]) -> Field {
  let mut dot_prod = 0;

  for i in 0..N {
    dot_prod += x[i] * y[i];
  }
  dot_prod
}

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_dot_prod() {
  assert(dot_prod([ 1 ], [ 2 ]) == 2); 
  assert(dot_prod([ 1, 2 ], [ 3, 4 ]) == 11); 
  assert(dot_prod([ 1, 2, 3 ], [ 4, 5, 6 ]) == 32); 
  assert(dot_prod([ 1, 2, 3, 4 ], [ 5, 6, 7, 8 ]) == 70); 
}