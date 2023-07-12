fn is_greater_than(x: Field, value: Field) -> bool {
  let x_bytes = x.to_le_bytes(32);
  let value_bytes = value.to_le_bytes(32);

  let mut result: bool = false;
  let mut done: bool = false;

  for i in 0..32 {
    if !done {
      let byte1 = x_bytes[31 - i] as u8; 
      let byte2 = value_bytes[31 - i] as u8; 
      if byte1 != byte2 {
        done = true;
        result = byte1 > byte2;    
      }
    }
  }
  result
}

#[test]
fn test_is_greater_than() {
  assert(is_greater_than(2, 0));
  assert(is_greater_than(-1, 2));
  assert(!is_greater_than(2, 2));
  assert(!is_greater_than(2, 3));
}