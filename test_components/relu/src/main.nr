use dep::noir_ml::activations::relu;

fn main(x: [Field; 2]) -> pub [Field; 2] {
   relu(x)
}

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_main() {
  assert(main([ -1, 1 ]) == [ 0, 1 ]);
}