use dep::noir_ml::utils::arg_max;

fn main(x: [Field; 2]) -> pub Field {
   arg_max(x)
}

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_main() {
  assert(main([ -1, 1 ]) == 1);
}