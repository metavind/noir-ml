fn vec_add<N>(x : [ Field; N ], y : [ Field; N ]) -> [ Field; N ] {
  let mut result = [0; N];

  for i in 0..N {
    result[i] = x[i] + y[i];
  }
  result
}

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_vec_add() {
  assert(vec_add([ 1 ], [ 2 ]) == [ 3 ]); 
  assert(vec_add([ 1, 2 ], [ 3, 4 ]) == [ 4, 6 ]); 
  assert(vec_add([ 1, 2, 3 ], [ 4, 5, 6 ]) == [ 5, 7, 9 ]); 
}