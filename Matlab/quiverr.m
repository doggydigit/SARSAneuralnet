clear all
close all

W00 = [0.000101 0.000395 0.000606 0.000326;
0.000228 0.000522 0.000983 0.000670;
0.000186 0.000794 0.001763 0.000844;
0.000045 0.000383 0.000852 0.000325]



W01=[0.000229 0.000232 0.000083 0.000048;
0.000247 0.000386 0.000546 0.000354;
0.000125 0.000817 0.001668 0.000687;
0.000049 0.000542 0.001093 0.000342]
W02 = [0.000056 0.000066 0.000322 0.000459;
0.000163 0.000430 0.001359 0.001266;
0.000154 0.001321 0.003176 0.001459;
0.000060 -13.9992 0.001759 0.000583]

W490 = [0.008192 0.012480 0.011179 0.006874;
0.012981 0.017076 0.019612 0.014755;
0.006612 0.020162 0.037066 0.018435;
0.001354 0.010073 0.020367 0.007757]
W491 = [0.005061 0.008954 0.012281 0.009940;
0.008003 0.013299 0.022433 0.019916;
0.004865 0.019975 0.039166 0.019920;
0.001324 0.010821 0.021208 0.007805;
]
W492 = [0.003486 0.008888 0.019073 0.018719;
0.006460 0.025170 0.073578 0.062783;
0.007255 0.067848 0.170287 0.087247;
0.003298 0.041054 0.092975 0.036244]

W150 = [0.003290 0.005165 0.004724 0.003055;
0.005196 0.007411 0.010236 0.007928;
0.002923 0.011817 0.024321 0.011871;
0.000775 0.007071 0.014767 0.005566]
W151 =[0.001944 0.003272 0.004434 0.003759;
0.003153 0.005990 0.011053 0.009140;
0.002381 0.012315 0.024672 0.011567;
0.000846 0.007423 0.014129 0.004930]
W152 = [0.001219 0.002909 0.006628 0.006789;
0.002130 0.006695 0.020509 0.019662;
0.001992 0.016230 0.042247 0.023709;
0.000808 0.009592 0.022175 0.008970]

W250 = [0.005243 0.007932 0.007503 0.004898;
0.008287 0.011505 0.016120 0.012429;
0.004521 0.017199 0.035545 0.017721;
0.001103 0.009556 0.020067 0.007674]
W251 = [0.002939 0.004901 0.007235 0.006562;
0.004869 0.008970 0.017446 0.015420;
0.003557 0.017865 0.037419 0.018483;
0.001155 0.010405 0.020929 0.007684]
W252 = [0.001966 0.004863 0.009917 0.009904;
0.003259 0.010644 0.031480 0.029587;
0.003184 0.026748 0.068324 0.037128;
0.001356 0.016283 0.037235 0.014682]

W400 = [0.007438 0.011515 0.010289 0.006374;
0.011580 0.015672 0.018840 0.014282;
0.005983 0.019462 0.036876 0.018367;
0.001288 0.009983 0.020346 0.007755]
W401 = [0.004380 0.007307 0.010609 0.009249;
0.007129 0.011861 0.021239 0.019308;
0.004511 0.019428 0.038924 0.019831;
0.001277 0.010724 0.021171 0.007801]
W402 = [0.002826 0.007502 0.015956 0.015425;
0.005134 0.019716 0.059054 0.051068;
0.005450 0.051191 0.134017 0.070217;
0.002361 0.030499 0.072285 0.028853]


W10 = [0.000423 0.000862 0.001029 0.000730;
0.000651 0.001178 0.002290 0.002057;
0.000433 0.002068 0.004881 0.002607;
0.000127 0.001169 0.002678 0.001056]
W11 = [0.000396 0.000547 0.000460 0.000198;
0.000499 0.000814 0.001325 0.000808;
0.000287 0.001615 0.003598 0.001639;
0.000098 0.000968 0.002105 0.000762]
W12 = [0.000076 0.000118 0.000654 0.000957;
0.000289 0.000623 0.002023 0.002136;
0.000287 0.001603 0.003950 0.002088;
0.000083 0.000892 0.001999 0.000728]

W20 =[0.000531 0.001037 0.001122 0.000760;
0.000879 0.001671 0.002914 0.002305;
0.000650 0.003427 0.007247 0.003338;
0.000215 0.002117 0.004376 0.001506];
W21 = [0.000532 0.000807 0.000954 0.000591;
0.000633 0.001156 0.002186 0.001561;
0.000340 0.002039 0.004718 0.002247;
0.000111 0.001135 0.002527 0.000932];
W22 = [0.000146 0.000276 0.000914 0.001208;
0.000463 0.001122 0.003255 0.003059;
0.000450 0.003115 0.007195 0.003384;
0.000162 0.001896 0.003915 0.001282;
];

W30 =[0.000604 0.001321 0.001422 0.000819;
0.001065 0.002014 0.003315 0.002475;
0.000768 0.003682 0.007798 0.003664;
0.000229 0.002172 0.004544 0.001608];
W31 = [0.000864 0.001199 0.001143 0.000766;
0.001040 0.001708 0.002961 0.002161;
0.000468 0.002791 0.006673 0.003204;
0.000139 0.001539 0.003603 0.001374;];
W32 = [0.000160 0.000402 0.001550 0.001874;
0.000563 0.001636 0.005080 0.004822;
0.000617 0.004442 0.010772 0.005368;
0.000221 0.002667 0.005926 0.002094];

W40 = [0.000976 0.001885 0.001674 0.000913;
0.001458 0.002557 0.003829 0.002919;
0.000889 0.004113 0.008764 0.004222;
0.000249 0.002362 0.005047 0.001826];
W41 = [0.000928 0.001289 0.001490 0.001224;
0.001290 0.002155 0.004105 0.003294;
0.000696 0.003616 0.008651 0.004510;
0.000191 0.001926 0.004559 0.001823
];
W42 = [0.000205 0.000619 0.002121 0.002334;
0.000687 0.002053 0.006399 0.005966;
0.000708 0.005327 0.013604 0.006849;
0.000246 0.003120 0.007330 0.002718];

W50 = [0.001130 0.002198 0.001859 0.000970;
0.001729 0.003033 0.004457 0.003292;
0.001039 0.004778 0.010238 0.004929;
0.000284 0.002764 0.005927 0.002135]
W51 = [0.001014 0.001527 0.001814 0.001330;
0.001395 0.002482 0.004643 0.003498;
0.000760 0.004054 0.009684 0.004917;
0.000205 0.002083 0.004968 0.001983];
W52 = [0.000307 0.000752 0.002572 0.002908;
0.000887 0.002642 0.008065 0.007488;
0.000887 0.007081 0.017472 0.008496;
0.000325 0.004206 0.009580 0.003409
];

W100 = [0.002135 0.003641 0.003134 0.001791;
0.003317 0.005064 0.007253 0.005503;
0.001832 0.007420 0.016443 0.008393;
0.000435 0.004107 0.009368 0.003681];
W101 = [0.001516 0.002456 0.003259 0.002735;
0.002380 0.004397 0.008390 0.006853;
0.001554 0.008110 0.018520 0.009058;
0.000444 0.004409 0.010143 0.003912];
W102 = [0.000918 0.002082 0.005013 0.005142;
0.001752 0.005020 0.014953 0.013978;
0.001533 0.011932 0.030523 0.016216;
0.000562 0.006926 0.016117 0.006164];

W70 = [0.001802 0.003010 0.002296 0.001273;
0.002589 0.004028 0.005529 0.004082;
0.001366 0.005812 0.012752 0.006454;
0.000341 0.003269 0.007213 0.002780];
W71 = [0.001119 0.001837 0.002436 0.001988;
0.001734 0.003370 0.006623 0.005316;
0.001141 0.006239 0.014192 0.007030;
0.000341 0.003398 0.007559 0.002862];
W72 = [0.000400 0.001162 0.003568 0.003764;
0.001082 0.003395 0.010290 0.009720;
0.001095 0.008822 0.021684 0.010936;
0.000409 0.005202 0.011821 0.004312];

W80 = [0.001935 0.003303 0.002594 0.001391;
0.002868 0.004378 0.005978 0.004462;
0.001517 0.006233 0.013685 0.006979;
0.000368 0.003550 0.007966 0.003091];
W81 = [0.001234 0.002050 0.002732 0.002294;
0.001989 0.003810 0.007482 0.006138;
0.001337 0.007075 0.016301 0.008143;
0.000385 0.003794 0.008699 0.003383];
W82 = [0.000592 0.001421 0.004057 0.004270;
0.001260 0.003832 0.011704 0.011045;
0.001179 0.009571 0.024156 0.012570;
0.000439 0.005572 0.012889 0.004858
];



figure(1)
R0 = QQ(W00,W01,W02)
% figure(2)
% R15 = QQ(W150,W151,W152)
% figure(3)
% R25 = QQ(W250,W251,W252)
% figure(4)
% R40 = QQ(W400,W401,W402)
figure(50)
RFinal = QQ(W490,W491,W492);
% figure(10)
% R0 = QQ(W10,W11,W12);
% figure(11)
% R15 = QQ(W20,W21,W22);
figure(3)
R25 = QQ(W30,W31,W32);
% figure(13)
% R40 = QQ(W40,W41,W42);
% figure(14)
% 
figure(10)
RR = QQ(W100,W101,W102);