use crate::dense;
use crate::relu;
use crate::arg_max;

use dep::std;

global hid_weights: [Field; 20] = [-201799, 253330, -94632, -1679517, -194549, 139218, -158168, 497705, -566381, 1437657, -1098219, -252057, -643523, -291986, -291486, 26194, 1668978, 576082, -1310992, -1006468];
global hid_bias: [Field; 10] = [332004696131, -615841984749, -247557893395, 528358042240, -533230543137, -386732697487, 680972695351, -655961990356, -657531619072, -634221017361];

global out_weights: [Field; 20] = [-167334, 1135549, -176126, -262836, 915547, 590526, -63137, 271829, 1246469, 1245495, 60699, -1024452, 24655, -177994, -1364973, -1049718, 78062, 185411, -1245607, -1103638];
global out_bias: [Field; 2] = [-94743110239505760, 611362576484680192];

fn circle(input: [Field; 2]) -> Field {
  let mut hid = dense::dense(input, hid_weights, hid_bias);
  hid = relu::relu(hid);
  let output = dense::dense(hid, out_weights, out_bias);
  let result = arg_max::arg_max(output);
  result
}

////////////////////
//     TESTS      //
////////////////////
#[test]
fn test_circle() {

let sample001 = [-181267, 336834];
assert(circle(sample001) == 1);

let sample002 = [4443, 966478];
assert(circle(sample002) == 0);

let sample003 = [-342958, -151764];
assert(circle(sample003) == 1);

let sample004 = [-293178, 962480];
assert(circle(sample004) == 0);

let sample005 = [192460, -394641];
assert(circle(sample005) == 1);

let sample006 = [321429, -305944];
assert(circle(sample006) == 1);

let sample007 = [850645, 552831];
assert(circle(sample007) == 0);

let sample008 = [999847, 189800];
assert(circle(sample008) == 0);

let sample009 = [-367555, -173140];
assert(circle(sample009) == 1);

let sample010 = [55282, 977907];
assert(circle(sample010) == 0);

let sample011 = [-481136, 796966];
assert(circle(sample011) == 0);

let sample012 = [-91968, 352019];
assert(circle(sample012) == 1);

let sample013 = [966399, 359227];
assert(circle(sample013) == 0);

let sample014 = [-281583, -308476];
assert(circle(sample014) == 1);

let sample015 = [-919316, -252760];
assert(circle(sample015) == 0);

let sample016 = [374433, -47643];
assert(circle(sample016) == 1);

let sample017 = [197512, 332326];
assert(circle(sample017) == 1);

let sample018 = [-686810, 713735];
assert(circle(sample018) == 0);

let sample019 = [-284807, 267009];
assert(circle(sample019) == 1);

let sample020 = [-841971, 465137];
assert(circle(sample020) == 0);

let sample021 = [8279, 989227];
assert(circle(sample021) == 0);

let sample022 = [1005008, -212258];
assert(circle(sample022) == 0);

let sample023 = [-331449, 214292];
assert(circle(sample023) == 1);

let sample024 = [-841805, 607126];
assert(circle(sample024) == 0);

let sample025 = [644762, 743086];
assert(circle(sample025) == 0);

let sample026 = [899288, 366547];
assert(circle(sample026) == 0);

let sample027 = [-245009, -962478];
assert(circle(sample027) == 0);

let sample028 = [860271, -518446];
assert(circle(sample028) == 0);

let sample029 = [-988762, -240288];
assert(circle(sample029) == 0);

let sample030 = [-206211, -297599];
assert(circle(sample030) == 1);

let sample031 = [302932, 238234];
assert(circle(sample031) == 1);

let sample032 = [591095, 803354];
assert(circle(sample032) == 0);

let sample033 = [-402363, 865038];
assert(circle(sample033) == 0);

let sample034 = [40283, -407915];
assert(circle(sample034) == 1);

let sample035 = [-504267, 851776];
assert(circle(sample035) == 0);

let sample036 = [61945, 389726];
assert(circle(sample036) == 1);

let sample037 = [-133510, 437193];
assert(circle(sample037) == 1);

let sample038 = [-328679, -245549];
assert(circle(sample038) == 1);

let sample039 = [-34611, -416385];
assert(circle(sample039) == 1);

let sample040 = [-257841, -304329];
assert(circle(sample040) == 1);

let sample041 = [-359800, 207066];
assert(circle(sample041) == 1);

let sample042 = [-402526, -65045];
assert(circle(sample042) == 1);

let sample043 = [188306, 315791];
assert(circle(sample043) == 1);

let sample044 = [424563, -887778];
assert(circle(sample044) == 0);

let sample045 = [-400453, 914752];
assert(circle(sample045) == 0);

let sample046 = [85719, 400918];
assert(circle(sample046) == 1);

let sample047 = [568556, -770839];
assert(circle(sample047) == 0);

let sample048 = [-14422, 1022163];
assert(circle(sample048) == 0);

let sample049 = [6690, 348149];
assert(circle(sample049) == 1);

let sample050 = [280876, -326192];
assert(circle(sample050) == 1);

let sample051 = [-311174, 944597];
assert(circle(sample051) == 0);

let sample052 = [-390345, 941992];
assert(circle(sample052) == 0);

let sample053 = [898346, -394937];
assert(circle(sample053) == 0);

let sample054 = [286990, -941758];
assert(circle(sample054) == 0);

let sample055 = [365011, 143223];
assert(circle(sample055) == 1);

let sample056 = [-321764, -909114];
assert(circle(sample056) == 0);

let sample057 = [401513, 88386];
assert(circle(sample057) == 1);

let sample058 = [-343326, -268442];
assert(circle(sample058) == 1);

let sample059 = [-413850, 80826];
assert(circle(sample059) == 1);

let sample060 = [-991448, 139636];
assert(circle(sample060) == 0);

let sample061 = [241868, -332681];
assert(circle(sample061) == 1);

let sample062 = [933092, 297936];
assert(circle(sample062) == 0);

let sample063 = [-951828, 160820];
assert(circle(sample063) == 0);

let sample064 = [-257683, -320984];
assert(circle(sample064) == 1);

let sample065 = [-154471, 356415];
assert(circle(sample065) == 1);

let sample066 = [-28061, 405897];
assert(circle(sample066) == 1);

let sample067 = [392122, -65502];
assert(circle(sample067) == 1);

let sample068 = [988178, -53517];
assert(circle(sample068) == 0);

let sample069 = [206659, 956319];
assert(circle(sample069) == 0);

let sample070 = [940917, -157621];
assert(circle(sample070) == 0);

let sample071 = [352570, -205462];
assert(circle(sample071) == 1);

let sample072 = [295823, 970343];
assert(circle(sample072) == 0);

let sample073 = [-1002715, 39226];
assert(circle(sample073) == 0);

let sample074 = [-381519, -65233];
assert(circle(sample074) == 1);

let sample075 = [426141, 19275];
assert(circle(sample075) == 1);

let sample076 = [332122, -199235];
assert(circle(sample076) == 1);

let sample077 = [-33730, -364001];
assert(circle(sample077) == 1);

let sample078 = [743769, 646839];
assert(circle(sample078) == 0);

let sample079 = [-581462, 807431];
assert(circle(sample079) == 0);

let sample080 = [289245, 344664];
assert(circle(sample080) == 1);

let sample081 = [-852944, 484932];
assert(circle(sample081) == 0);

let sample082 = [-158523, 324373];
assert(circle(sample082) == 1);

let sample083 = [941088, 186300];
assert(circle(sample083) == 0);

let sample084 = [-54190, -399184];
assert(circle(sample084) == 1);

let sample085 = [-409041, 122957];
assert(circle(sample085) == 1);

let sample086 = [190315, 380451];
assert(circle(sample086) == 1);

let sample087 = [672726, -761079];
assert(circle(sample087) == 0);

let sample088 = [-574909, -791126];
assert(circle(sample088) == 0);

let sample089 = [132134, -971560];
assert(circle(sample089) == 0);

let sample090 = [1011023, -194904];
assert(circle(sample090) == 0);

let sample091 = [-408712, -199850];
assert(circle(sample091) == 1);

let sample092 = [10265, -385426];
assert(circle(sample092) == 1);

let sample093 = [-44572, -327246];
assert(circle(sample093) == 1);

let sample094 = [149966, -378175];
assert(circle(sample094) == 1);

let sample095 = [-331500, -281734];
assert(circle(sample095) == 1);

let sample096 = [-677929, 763232];
assert(circle(sample096) == 0);

let sample097 = [54594, 354398];
assert(circle(sample097) == 1);

let sample098 = [-431927, -12556];
assert(circle(sample098) == 1);

let sample099 = [-899622, 497820];
assert(circle(sample099) == 0);

let sample100 = [80517, -399861];
assert(circle(sample100) == 1);

let sample101 = [42558, 1029245];
assert(circle(sample101) == 0);

let sample102 = [-984180, -327878];
assert(circle(sample102) == 0);

let sample103 = [165543, 368111];
assert(circle(sample103) == 1);

let sample104 = [398010, 92428];
assert(circle(sample104) == 1);

let sample105 = [-930273, 304480];
assert(circle(sample105) == 0);

let sample106 = [18094, -970994];
assert(circle(sample106) == 0);

let sample107 = [-78485, -390369];
assert(circle(sample107) == 1);

let sample108 = [765770, 665598];
assert(circle(sample108) == 0);

let sample109 = [-668440, -682017];
assert(circle(sample109) == 0);

let sample110 = [346124, -92909];
assert(circle(sample110) == 1);

let sample111 = [19130, 373351];
assert(circle(sample111) == 1);

let sample112 = [-256776, 303201];
assert(circle(sample112) == 1);

let sample113 = [424139, 903133];
assert(circle(sample113) == 0);

let sample114 = [294383, -299256];
assert(circle(sample114) == 1);

let sample115 = [453755, 6754];
assert(circle(sample115) == 1);

let sample116 = [-244799, -934766];
assert(circle(sample116) == 0);

let sample117 = [-192499, -347785];
assert(circle(sample117) == 1);

let sample118 = [-773517, -657036];
assert(circle(sample118) == 0);

let sample119 = [-122863, -389240];
assert(circle(sample119) == 1);

let sample120 = [-130393, -993221];
assert(circle(sample120) == 0);

let sample121 = [-372748, 163046];
assert(circle(sample121) == 1);

let sample122 = [-83355, -966006];
assert(circle(sample122) == 0);

let sample123 = [880451, -426968];
assert(circle(sample123) == 0);

let sample124 = [-441032, 100954];
assert(circle(sample124) == 1);

let sample125 = [431662, -181769];
assert(circle(sample125) == 1);

let sample126 = [-350669, -953480];
assert(circle(sample126) == 0);

let sample127 = [-332432, -176015];
assert(circle(sample127) == 1);

let sample128 = [226715, 322670];
assert(circle(sample128) == 1);

let sample129 = [-126144, 943683];
assert(circle(sample129) == 0);

let sample130 = [313120, 249583];
assert(circle(sample130) == 1);

let sample131 = [-438613, 54621];
assert(circle(sample131) == 1);

let sample132 = [840822, 523738];
assert(circle(sample132) == 0);

let sample133 = [-4778, -400692];
assert(circle(sample133) == 1);

let sample134 = [242795, 948710];
assert(circle(sample134) == 0);

let sample135 = [373605, 922076];
assert(circle(sample135) == 0);

let sample136 = [909799, 364011];
assert(circle(sample136) == 0);

let sample137 = [95969, -396239];
assert(circle(sample137) == 1);

let sample138 = [-499587, -878335];
assert(circle(sample138) == 0);

let sample139 = [-944543, 197432];
assert(circle(sample139) == 0);

let sample140 = [182824, 335599];
assert(circle(sample140) == 1);

let sample141 = [-831041, -580908];
assert(circle(sample141) == 0);

let sample142 = [587281, 828014];
assert(circle(sample142) == 0);

let sample143 = [-791065, 604149];
assert(circle(sample143) == 0);

let sample144 = [-333011, -172665];
assert(circle(sample144) == 1);

let sample145 = [-869238, -542824];
assert(circle(sample145) == 0);

let sample146 = [158954, -942026];
assert(circle(sample146) == 0);

let sample147 = [371561, 177261];
assert(circle(sample147) == 1);

let sample148 = [-801488, -597013];
assert(circle(sample148) == 0);

let sample149 = [215125, 341398];
assert(circle(sample149) == 1);

let sample150 = [-250084, 956643];
assert(circle(sample150) == 0);

let sample151 = [374729, -174519];
assert(circle(sample151) == 1);

let sample152 = [707946, 684340];
assert(circle(sample152) == 0);

let sample153 = [-398388, -54827];
assert(circle(sample153) == 1);

let sample154 = [-946849, -240173];
assert(circle(sample154) == 0);

let sample155 = [380515, -61908];
assert(circle(sample155) == 1);

let sample156 = [139644, 342184];
assert(circle(sample156) == 1);

let sample157 = [260088, 308179];
assert(circle(sample157) == 1);

let sample158 = [-830933, -547113];
assert(circle(sample158) == 0);

let sample159 = [-773280, -574072];
assert(circle(sample159) == 0);

let sample160 = [-454456, -66927];
assert(circle(sample160) == 1);

let sample161 = [-339927, 924667];
assert(circle(sample161) == 0);

let sample162 = [379388, -42004];
assert(circle(sample162) == 1);

let sample163 = [-208178, -380182];
assert(circle(sample163) == 1);

let sample164 = [-695490, 634915];
assert(circle(sample164) == 0);

let sample165 = [182752, 380588];
assert(circle(sample165) == 1);

let sample166 = [505672, -805716];
assert(circle(sample166) == 0);

let sample167 = [957315, 80263];
assert(circle(sample167) == 0);

let sample168 = [-737271, -589676];
assert(circle(sample168) == 0);

let sample169 = [746188, -598009];
assert(circle(sample169) == 0);

let sample170 = [147189, 383858];
assert(circle(sample170) == 1);

let sample171 = [216210, 368407];
assert(circle(sample171) == 1);

let sample172 = [156373, 400903];
assert(circle(sample172) == 1);

let sample173 = [387987, 151172];
assert(circle(sample173) == 1);

let sample174 = [857803, -379055];
assert(circle(sample174) == 0);

let sample175 = [868670, 518042];
assert(circle(sample175) == 0);

let sample176 = [795830, -578743];
assert(circle(sample176) == 0);

let sample177 = [229159, -952642];
assert(circle(sample177) == 0);

let sample178 = [959129, 55981];
assert(circle(sample178) == 0);

let sample179 = [197657, 350627];
assert(circle(sample179) == 1);

let sample180 = [306151, -975891];
assert(circle(sample180) == 0);

let sample181 = [550506, -797310];
assert(circle(sample181) == 0);

let sample182 = [-401149, -28667];
assert(circle(sample182) == 1);

let sample183 = [582016, 826817];
assert(circle(sample183) == 0);

let sample184 = [-233108, -347019];
assert(circle(sample184) == 1);

let sample185 = [-403536, 137955];
assert(circle(sample185) == 1);

let sample186 = [-929980, 399506];
assert(circle(sample186) == 0);

let sample187 = [-100094, 1012894];
assert(circle(sample187) == 0);

let sample188 = [911969, 322399];
assert(circle(sample188) == 0);

let sample189 = [-61711, 378633];
assert(circle(sample189) == 1);

let sample190 = [203245, -359836];
assert(circle(sample190) == 1);

let sample191 = [-1000760, 72362];
assert(circle(sample191) == 0);

let sample192 = [-328658, -192217];
assert(circle(sample192) == 1);

let sample193 = [211913, 370847];
assert(circle(sample193) == 1);

let sample194 = [580638, 822021];
assert(circle(sample194) == 0);

let sample195 = [-231233, -346617];
assert(circle(sample195) == 1);

let sample196 = [771740, 596903];
assert(circle(sample196) == 0);

let sample197 = [-369507, 118307];
assert(circle(sample197) == 1);

let sample198 = [964070, -70693];
assert(circle(sample198) == 0);

let sample199 = [-324710, 227571];
assert(circle(sample199) == 1);

let sample200 = [133102, 403879];
assert(circle(sample200) == 1);

}