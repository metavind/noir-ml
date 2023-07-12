fn arg_min(values: [ Field ]) -> Field {
  assert(values.len() as u64 > 0);

  let mut min_index = 0;
  let mut min_value = values[0];

  for i in 1..values.len() {
    let value = values[i];
    if value as u64 < min_value as u64 {
      min_value = value;
      min_index = i;
    }
  }
  min_index
}

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_arg_min() {
  assert(arg_min([ 1 ]) == 0); 
  assert(arg_min([ 1, 2 ]) == 0); 
  assert(arg_min([ 2, 1]) == 1); 
  assert(arg_min([ 3, 2, 5, 1, 4 ]) == 3); 
}