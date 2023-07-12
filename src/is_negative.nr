//check if a field element is negative by comparing it with the constant (p-1)/2
//comp_constant = 10944121435919637611123202872628637544274182200208017171849102093287904247808;
global comp_constant_bytes = [0x00, 0x00, 0x00, 0xf8, 0xc9, 0xfa, 0xf0, 0xa1, 0x48, 0xb8, 0xdc, 0x3c, 0x24, 0xf4, 0x19, 0x94, 0x2e, 0xac, 0xc0, 0x40, 0xdb, 0x22, 0x28, 0xdc, 0x14, 0xd0, 0x98, 0x70, 0x39, 0x27, 0x32, 0x18];

fn is_negative(value: Field) -> bool {
    let value_bytes = value.to_le_bytes(32);

    let mut result: bool = false;
    let mut done: bool = false;

    for i in 0..32 {
        if !done & (value_bytes[31 - i] != comp_constant_bytes[31 - i]) {
            done = true;
            result = (value_bytes[31 - i] > comp_constant_bytes[31 - i]);    
        }
    }
    result
}

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_is_negative() {
  assert(is_negative(0)==false);
  assert(is_negative(1)==false);
  assert(is_negative(-1)==true);
  let comp_constant = 10944121435919637611123202872628637544274182200208017171849102093287904247808;
  assert(is_negative(comp_constant)==false);
  assert(is_negative(comp_constant+1)==true);
}