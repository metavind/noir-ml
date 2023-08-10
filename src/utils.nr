//check if a field element is positive by comparing it with the constant (p-1)/2
//comp_constant = 10944121435919637611123202872628637544274182200208017171849102093287904247808;
global comp_constant_bytes = [0x00, 0x00, 0x00, 0xf8, 0xc9, 0xfa, 0xf0, 0xa1, 0x48, 0xb8, 0xdc, 0x3c, 0x24, 0xf4, 0x19, 0x94, 0x2e, 0xac, 0xc0, 0x40, 0xdb, 0x22, 0x28, 0xdc, 0x14, 0xd0, 0x98, 0x70, 0x39, 0x27, 0x32, 0x18];

fn is_positive(value: Field) -> bool {
  let value_bytes = value.to_le_bytes(32);

  let mut result: bool = true;
  let mut done: bool = false;

  for i in 0..32 {
    if !done {
      let byte1 = value_bytes[31 - i] as u8; 
      let byte2 = comp_constant_bytes[31 - i] as u8; 
      if byte1 != byte2 {
        done = true;
        result = byte1 < byte2;    
      }
    }
  }
  result
}


fn is_negative(value: Field) -> bool {
  let value_bytes = value.to_le_bytes(32);

  let mut result: bool = false;
  let mut done: bool = false;

  for i in 0..32 {
    if !done {
      let byte1 = value_bytes[31 - i] as u8; 
      let byte2 = comp_constant_bytes[31 - i] as u8; 
      if byte1 != byte2 {
        done = true;
        result = byte1 > byte2;    
      }
    }
  }
  result
}

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


fn is_greater_than_signed(x: Field, value: Field) -> bool {
  let x_is_positive = is_positive(x);
  let value_is_positive = is_positive(value);

  if (x_is_positive == value_is_positive) {
     is_greater_than(x, value)
  } else {
    x_is_positive > value_is_positive
  }
}


fn arg_max<N>(values: [ Field; N ]) -> Field {
  assert(N as u64 > 0);

  let mut max_index = 0;
  let mut max_value = values[0];

  for i in 1..N {
    let value = values[i];
    if is_greater_than_signed(value, max_value) {
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
fn test_is_positive() {
  assert(is_positive(0) == true); //0 is positive
  assert(is_positive(1) == true);
  assert(is_positive(-1) == false);
  let comp_constant = 10944121435919637611123202872628637544274182200208017171849102093287904247808;
  assert(is_positive(comp_constant) == true);
  assert(is_positive(comp_constant - 1) == true);
  assert(is_positive(comp_constant + 1) == false);
}

#[test]
fn test_is_negative() {
  assert(is_negative(0) == false);
  assert(is_negative(1) == false);
  assert(is_negative(-1) == true);
  let comp_constant = 10944121435919637611123202872628637544274182200208017171849102093287904247808;
  assert(is_negative(comp_constant) == false);
  assert(is_negative(comp_constant - 1) == false);
  assert(is_negative(comp_constant + 1) == true);
}

#[test]
fn test_is_greater_than() {
  assert(is_greater_than(2, 0));
  assert(is_greater_than(-1, 2));
  assert(!is_greater_than(2, 2));
  assert(!is_greater_than(2, 3));
}

#[test]
fn test_is_greater_than_signed() {
  assert(is_greater_than_signed(2, 1));
  assert(is_greater_than_signed(1, -1));
  assert(is_greater_than_signed(-10, -11));
  assert(!is_greater_than_signed(-1, 1));
  assert(!is_greater_than_signed(-3, -2));
}

#[test]
fn test_arg_max() {
    assert(arg_max([ 1 ]) == 0); 
    assert(arg_max([ 1, 2 ]) == 1); 
    assert(arg_max([ 2, 1]) == 0); 
    assert(arg_max([ 3, 2, 5, 1, 4 ]) == 2);
    assert(arg_max([ -1, 1]) == 1);
    assert(arg_max([ 1, -1]) == 0);
    assert(arg_max([ -3, -2, -5, -1, -4]) == 3);
}