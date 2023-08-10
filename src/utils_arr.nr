fn arr_add<N>(x : [ Field; N ], y : [ Field; N ]) -> [ Field; N ] {
  let mut result = [0; N];

  for i in 0..N {
    result[i] = x[i] + y[i];
  }
  result
}


fn dot_prod<N>(x : [ Field; N ], y : [ Field; N ]) -> Field {
  let mut dot_prod = 0;

  for i in 0..N {
    dot_prod += x[i] * y[i];
  }
  dot_prod
}


fn prune_arr<N>(x : [ Field ]) -> [ Field; N ] {
  let mut result = [ 0; N ];

  for i in 0..N {
    result[i] = x[i];
  }
  result
}


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
fn test_arr_add() {
  assert(arr_add([ 1 ], [ 2 ]) == [ 3 ]); 
  assert(arr_add([ 1, 2 ], [ 3, 4 ]) == [ 4, 6 ]); 
  assert(arr_add([ 1, 2, 3 ], [ 4, 5, 6 ]) == [ 5, 7, 9 ]);
  
}

#[test]
fn test_dot_prod() {
  assert(dot_prod([ 1 ], [ 2 ]) == 2); 
  assert(dot_prod([ 1, 2 ], [ 3, 4 ]) == 11); 
  assert(dot_prod([ 1, 2, 3 ], [ 4, 5, 6 ]) == 32); 
  assert(dot_prod([ 1, 2, 3, 4 ], [ 5, 6, 7, 8 ]) == 70); 
}

#[test]
fn test_prune_arr() {
  assert(prune_arr([ 1, 2, 3, 4, 5 ]) == [ 1 ]);
  assert(prune_arr([ 1, 2, 3, 4, 5 ]) == [ 1, 2 ]);
  assert(prune_arr([ 1, 2, 3, 4, 5 ]) == [ 1, 2, 3 ]);
  assert(prune_arr([ 1, 2, 3, 4, 5 ]) == [ 1, 2, 3, 4 ]);
  assert(prune_arr([ 1, 2, 3, 4, 5 ]) == [ 1, 2, 3, 4, 5 ]);
}

#[test]
fn test_pad_arr() {
  assert(pad_arr([ 1 ]) == [ 1 ]);
  assert(pad_arr([ 1 ]) == [ 1, 0 ]);
  assert(pad_arr([ 1, 2 ]) == [ 1, 2 ]);
  assert(pad_arr([ 1 ]) == [ 1, 0, 0, 0 ,0 ]);
  assert(pad_arr([ 1, 2, 3, 4, 5 ]) == [ 1, 2, 3, 4, 5 ]);
}
