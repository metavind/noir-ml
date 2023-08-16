use dep::noir_ml::{layers::dense};

global w: [Field; 4] = [ 3, 4, 5, 6 ];
global b: [Field; 2] = [ 7, 8 ];

fn main(x: [Field; 2]) -> pub [Field; 2] {
   dense(x, w, b)
}

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_main() {
  assert(main([ 1, 2 ]) == [ 18, 25 ]);
}