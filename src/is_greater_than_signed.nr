use crate::is_positive;
use crate::is_negative;
use crate::is_greater_than;

fn is_greater_than_signed(x: Field, value: Field) -> bool {
  let x_is_positive = is_positive::is_positive(x);
  let value_is_positive = is_positive::is_positive(value);

  if (x_is_positive & !value_is_positive) {
    true
  } else if (!x_is_positive & value_is_positive) {
    false
  } else {
    is_greater_than::is_greater_than(x, value)
  }
}

#[test]
fn test_is_greater_than_signed() {
  assert(is_greater_than_signed(2, 1));
  assert(is_greater_than_signed(1, -1));
  assert(is_greater_than_signed(-10, -11));
  assert(!is_greater_than_signed(-1, 1));
  assert(!is_greater_than_signed(-3, -2));
}