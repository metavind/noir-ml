fn prune_arr<N>(x : [ Field ]) -> [ Field; N ] {
  let mut result = [ 0; N ];

  for i in 0..N {
    result[i] = x[i];
  }
  result
}

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_prune_arr() {
  assert(prune_arr([ 1, 2, 3, 4, 5 ]) == [ 1 ]);
  assert(prune_arr([ 1, 2, 3, 4, 5 ]) == [ 1, 2 ]);
  assert(prune_arr([ 1, 2, 3, 4, 5 ]) == [ 1, 2, 3 ]);
  assert(prune_arr([ 1, 2, 3, 4, 5 ]) == [ 1, 2, 3, 4 ]);
  assert(prune_arr([ 1, 2, 3, 4, 5 ]) == [ 1, 2, 3, 4, 5 ]);
}