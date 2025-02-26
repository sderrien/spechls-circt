#include<ac_int.h>


struct SCC_0_res {
	int out_0;
};

int main() {
  comb_update_SCC_0();
}
 struct SCC_0_res SCC_0(int v_0, int v_1, bool v_2, bool v_3, int* v_4) {
	int v_6;
	int v_7;
	int v_8;
	bool v_9;
	int v_10;
	ac_int<2,false> v_11;
	ac_int<4,false> v_12;
	int v_13;
	ac_int<3,false> v_14;
	int LUT_op_0[8]={0,3,5,6,8,3334,4564,45}
	int v_15;
	bool v_16;
	int v_17;
	DelayLine <int,2> op_1;

	int v_18;
	bool exit;
		 // Initialisation update

void init_SCC_0() {
	// %c1_i32 = hw.constant 1 : i32
	int v_5= 1;

	// init Mu adfag
	v_18= v_10;
	// update Mu adfag
}
		 // Combinational update

void comb_update_SCC_0() {
		// %0 = comb.add %in_0, %c1_i32 : i32
		v_6=v_0 + v_5;

		// %1 = comb.and %0, %in_1 : i32
		v_7=v_6 & v_1;

		// %2 = comb.xor %1, %in_0 : i32
		v_8=v_7 ^ v_0;

		// %3 = comb.extract %2 from 0 : (i32) -> i1
		v_9= (v_8>>0)&1 ;

		// %4 = comb.mux %3, %2, %0 : i32
		v_10 = v_9?v_8:v_6;

		// %5 = comb.extract %2 from 0 : (i32) -> i2
		v_11= (v_8>>0)&3 ;

		// %6 = comb.extract %2 from 0 : (i32) -> i4
		v_12= (v_8>>0)&15 ;

		switch(v_11) {

		case 0: v_13=v_8; break;
		case 1: v_13=v_6; break;
		case 2: v_13=v_0; break;
		case 3: v_13=v_10; break;
		default : v_13=v_10; break;
		}

		// %8 = comb.extract %2 from 2 : (i32) -> i3
		v_14= (v_8>>2)&7 ;

			v_15=LUT_op_0[v_14];

		// %10 = comb.extract %2 from 5 : (i32) -> i1
		v_16= (v_8>>5)&1 ;

		// %11 = SpecHLS.delay %10 -> %4 by 2(%4) : i32
		op_1.push();
		// %13 = SpecHLS.exit %10
			exit = v_16;

		// hw.output %4 : i32
		return {v_10};
}
		 // Synchronous update

void sync_update_SCC_0() {
		// %11 = SpecHLS.delay %10 -> %4 by 2(%4) : i32
		v_17=op_1.pop();

		v_18= v_8;
}
