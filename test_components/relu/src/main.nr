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


fn main(x : [ Field; 2 ]) -> pub [ Field; 2 ] {
  let mut result = [0; 2];

  for i in 0..2 {
    if is_positive(x[i]) {
      result[i] = x[i];
    }
  }
  result
}

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_main() {
  assert(main([ -1, 1 ]) == [ 0, 1 ]);
}