use noir_ml::layers::dense;
use noir_ml::activations::relu;
use noir_ml::utils::arg_max;

global hid_weights: [Field; 20] = [-201799, 253330, -94632, -1679517, -194549, 139218, -158168, 497705, -566381, 1437657, -1098219, -252057, -643523, -291986, -291486, 26194, 1668978, 576082, -1310992, -1006468];
global hid_bias: [Field; 10] = [332004696131, -615841984749, -247557893395, 528358042240, -533230543137, -386732697487, 680972695351, -655961990356, -657531619072, -634221017361];

global out_weights: [Field; 20] = [-167334, 1135549, -176126, -262836, 915547, 590526, -63137, 271829, 1246469, 1245495, 60699, -1024452, 24655, -177994, -1364973, -1049718, 78062, 185411, -1245607, -1103638];
global out_bias: [Field; 2] = [-94743110239505760, 611362576484680192];

fn circle_fc(input: [Field; 2]) -> Field {
  let mut hid = dense(input, hid_weights, hid_bias);
  hid = relu(hid);
  let output = dense(hid, out_weights, out_bias);
  let result = arg_max(output);
  result
}

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_circle_fc_001() {
  let sample = [-181267, 336834];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_002() {
  let sample = [4443, 966478];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_003() {
  let sample = [-342958, -151764];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_004() {
  let sample = [-293178, 962480];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_005() {
  let sample = [192460, -394641];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_006() {
  let sample = [321429, -305944];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_007() {
  let sample = [850645, 552831];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_008() {
  let sample = [999847, 189800];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_009() {
  let sample = [-367555, -173140];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_010() {
  let sample = [55282, 977907];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_011() {
  let sample = [-481136, 796966];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_012() {
  let sample = [-91968, 352019];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_013() {
  let sample = [966399, 359227];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_014() {
  let sample = [-281583, -308476];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_015() {
  let sample = [-919316, -252760];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_016() {
  let sample = [374433, -47643];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_017() {
  let sample = [197512, 332326];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_018() {
  let sample = [-686810, 713735];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_019() {
  let sample = [-284807, 267009];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_020() {
  let sample = [-841971, 465137];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_021() {
  let sample = [8279, 989227];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_022() {
  let sample = [1005008, -212258];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_023() {
  let sample = [-331449, 214292];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_024() {
  let sample = [-841805, 607126];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_025() {
  let sample = [644762, 743086];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_026() {
  let sample = [899288, 366547];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_027() {
  let sample = [-245009, -962478];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_028() {
  let sample = [860271, -518446];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_029() {
  let sample = [-988762, -240288];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_030() {
  let sample = [-206211, -297599];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_031() {
  let sample = [302932, 238234];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_032() {
  let sample = [591095, 803354];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_033() {
  let sample = [-402363, 865038];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_034() {
  let sample = [40283, -407915];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_035() {
  let sample = [-504267, 851776];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_036() {
  let sample = [61945, 389726];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_037() {
  let sample = [-133510, 437193];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_038() {
  let sample = [-328679, -245549];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_039() {
  let sample = [-34611, -416385];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_040() {
  let sample = [-257841, -304329];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_041() {
  let sample = [-359800, 207066];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_042() {
  let sample = [-402526, -65045];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_043() {
  let sample = [188306, 315791];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_044() {
  let sample = [424563, -887778];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_045() {
  let sample = [-400453, 914752];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_046() {
  let sample = [85719, 400918];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_047() {
  let sample = [568556, -770839];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_048() {
  let sample = [-14422, 1022163];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_049() {
  let sample = [6690, 348149];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_050() {
  let sample = [280876, -326192];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_051() {
  let sample = [-311174, 944597];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_052() {
  let sample = [-390345, 941992];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_053() {
  let sample = [898346, -394937];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_054() {
  let sample = [286990, -941758];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_055() {
  let sample = [365011, 143223];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_056() {
  let sample = [-321764, -909114];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_057() {
  let sample = [401513, 88386];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_058() {
  let sample = [-343326, -268442];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_059() {
  let sample = [-413850, 80826];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_060() {
  let sample = [-991448, 139636];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_061() {
  let sample = [241868, -332681];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_062() {
  let sample = [933092, 297936];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_063() {
  let sample = [-951828, 160820];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_064() {
  let sample = [-257683, -320984];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_065() {
  let sample = [-154471, 356415];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_066() {
  let sample = [-28061, 405897];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_067() {
  let sample = [392122, -65502];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_068() {
  let sample = [988178, -53517];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_069() {
  let sample = [206659, 956319];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_070() {
  let sample = [940917, -157621];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_071() {
  let sample = [352570, -205462];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_072() {
  let sample = [295823, 970343];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_073() {
  let sample = [-1002715, 39226];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_074() {
  let sample = [-381519, -65233];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_075() {
  let sample = [426141, 19275];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_076() {
  let sample = [332122, -199235];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_077() {
  let sample = [-33730, -364001];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_078() {
  let sample = [743769, 646839];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_079() {
  let sample = [-581462, 807431];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_080() {
  let sample = [289245, 344664];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_081() {
  let sample = [-852944, 484932];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_082() {
  let sample = [-158523, 324373];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_083() {
  let sample = [941088, 186300];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_084() {
  let sample = [-54190, -399184];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_085() {
  let sample = [-409041, 122957];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_086() {
  let sample = [190315, 380451];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_087() {
  let sample = [672726, -761079];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_088() {
  let sample = [-574909, -791126];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_089() {
  let sample = [132134, -971560];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_090() {
  let sample = [1011023, -194904];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_091() {
  let sample = [-408712, -199850];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_092() {
  let sample = [10265, -385426];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_093() {
  let sample = [-44572, -327246];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_094() {
  let sample = [149966, -378175];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_095() {
  let sample = [-331500, -281734];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_096() {
  let sample = [-677929, 763232];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_097() {
  let sample = [54594, 354398];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_098() {
  let sample = [-431927, -12556];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_099() {
  let sample = [-899622, 497820];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_100() {
  let sample = [80517, -399861];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_101() {
  let sample = [42558, 1029245];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_102() {
  let sample = [-984180, -327878];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_103() {
  let sample = [165543, 368111];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_104() {
  let sample = [398010, 92428];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_105() {
  let sample = [-930273, 304480];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_106() {
  let sample = [18094, -970994];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_107() {
  let sample = [-78485, -390369];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_108() {
  let sample = [765770, 665598];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_109() {
  let sample = [-668440, -682017];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_110() {
  let sample = [346124, -92909];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_111() {
  let sample = [19130, 373351];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_112() {
  let sample = [-256776, 303201];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_113() {
  let sample = [424139, 903133];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_114() {
  let sample = [294383, -299256];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_115() {
  let sample = [453755, 6754];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_116() {
  let sample = [-244799, -934766];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_117() {
  let sample = [-192499, -347785];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_118() {
  let sample = [-773517, -657036];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_119() {
  let sample = [-122863, -389240];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_120() {
  let sample = [-130393, -993221];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_121() {
  let sample = [-372748, 163046];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_122() {
  let sample = [-83355, -966006];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_123() {
  let sample = [880451, -426968];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_124() {
  let sample = [-441032, 100954];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_125() {
  let sample = [431662, -181769];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_126() {
  let sample = [-350669, -953480];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_127() {
  let sample = [-332432, -176015];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_128() {
  let sample = [226715, 322670];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_129() {
  let sample = [-126144, 943683];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_130() {
  let sample = [313120, 249583];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_131() {
  let sample = [-438613, 54621];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_132() {
  let sample = [840822, 523738];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_133() {
  let sample = [-4778, -400692];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_134() {
  let sample = [242795, 948710];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_135() {
  let sample = [373605, 922076];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_136() {
  let sample = [909799, 364011];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_137() {
  let sample = [95969, -396239];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_138() {
  let sample = [-499587, -878335];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_139() {
  let sample = [-944543, 197432];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_140() {
  let sample = [182824, 335599];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_141() {
  let sample = [-831041, -580908];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_142() {
  let sample = [587281, 828014];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_143() {
  let sample = [-791065, 604149];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_144() {
  let sample = [-333011, -172665];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_145() {
  let sample = [-869238, -542824];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_146() {
  let sample = [158954, -942026];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_147() {
  let sample = [371561, 177261];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_148() {
  let sample = [-801488, -597013];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_149() {
  let sample = [215125, 341398];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_150() {
  let sample = [-250084, 956643];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_151() {
  let sample = [374729, -174519];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_152() {
  let sample = [707946, 684340];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_153() {
  let sample = [-398388, -54827];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_154() {
  let sample = [-946849, -240173];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_155() {
  let sample = [380515, -61908];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_156() {
  let sample = [139644, 342184];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_157() {
  let sample = [260088, 308179];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_158() {
  let sample = [-830933, -547113];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_159() {
  let sample = [-773280, -574072];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_160() {
  let sample = [-454456, -66927];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_161() {
  let sample = [-339927, 924667];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_162() {
  let sample = [379388, -42004];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_163() {
  let sample = [-208178, -380182];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_164() {
  let sample = [-695490, 634915];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_165() {
  let sample = [182752, 380588];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_166() {
  let sample = [505672, -805716];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_167() {
  let sample = [957315, 80263];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_168() {
  let sample = [-737271, -589676];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_169() {
  let sample = [746188, -598009];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_170() {
  let sample = [147189, 383858];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_171() {
  let sample = [216210, 368407];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_172() {
  let sample = [156373, 400903];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_173() {
  let sample = [387987, 151172];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_174() {
  let sample = [857803, -379055];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_175() {
  let sample = [868670, 518042];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_176() {
  let sample = [795830, -578743];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_177() {
  let sample = [229159, -952642];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_178() {
  let sample = [959129, 55981];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_179() {
  let sample = [197657, 350627];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_180() {
  let sample = [306151, -975891];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_181() {
  let sample = [550506, -797310];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_182() {
  let sample = [-401149, -28667];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_183() {
  let sample = [582016, 826817];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_184() {
  let sample = [-233108, -347019];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_185() {
  let sample = [-403536, 137955];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_186() {
  let sample = [-929980, 399506];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_187() {
  let sample = [-100094, 1012894];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_188() {
  let sample = [911969, 322399];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_189() {
  let sample = [-61711, 378633];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_190() {
  let sample = [203245, -359836];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_191() {
  let sample = [-1000760, 72362];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_192() {
  let sample = [-328658, -192217];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_193() {
  let sample = [211913, 370847];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_194() {
  let sample = [580638, 822021];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_195() {
  let sample = [-231233, -346617];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_196() {
  let sample = [771740, 596903];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_197() {
  let sample = [-369507, 118307];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_198() {
  let sample = [964070, -70693];
  assert(circle_fc(sample) == 0);
}

#[test]
fn test_circle_fc_199() {
  let sample = [-324710, 227571];
  assert(circle_fc(sample) == 1);
}

#[test]
fn test_circle_fc_200() {
  let sample = [133102, 403879];
  assert(circle_fc(sample) == 1);
}