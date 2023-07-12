fn pad_arr<MAX_LENGTH>(x : [ Field ]) -> [ Field; MAX_LENGTH] {

  let n = x.len();
  assert(n <= MAX_LENGTH);

  let mut result = [ 0; MAX_LENGTH ];

  for i in 0..n {
    result[i] = x[i];
  }
  result
}

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_pad_arr() {
  assert(pad_arr([ 1 ]) == [ 1 ]);
  assert(pad_arr([ 1 ]) == [ 1, 0 ]);
  assert(pad_arr([ 1, 2 ]) == [ 1, 2 ]);
  assert(pad_arr([ 1 ]) == [ 1, 0, 0, 0 ,0 ]);
  assert(pad_arr([ 1, 2, 3, 4, 5 ]) == [ 1, 2, 3, 4, 5 ]);
}