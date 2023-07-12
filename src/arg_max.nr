use crate::is_greater_than_signed;

fn arg_max(values: [ Field ]) -> Field {
  assert(values.len() as u64 > 0);

  let mut max_index = 0;
  let mut max_value = values[0];

  for i in 1..values.len() {
    let value = values[i];
    if is_greater_than_signed::is_greater_than_signed(value, max_value) {
      max_value = value;
      max_index = i;
    }
  }
  max_index
}

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_arg_max() {
    assert(arg_max([ 1 ]) == 0); 
    assert(arg_max([ 1, 2 ]) == 1); 
    assert(arg_max([ 2, 1]) == 0); 
    assert(arg_max([ 3, 2, 5, 1, 4 ]) == 2); 
}