module {
	hw.module @gamma_3_6(in %inner_ctrl : i3, in %outer_ctrl : i2, out out_0 : i32) attributes {"#pragma" = "INLINE"} {


	%0 = hw.constant 0 :i32					
	%1 = hw.constant 1 :i32					
	%2 = hw.constant 2 :i32					
	%3 = hw.constant 3 :i32					
	%4 = hw.constant 4 :i32					
	%5 = hw.constant 5 :i32					
	%6 = hw.constant 6 :i32					
	%7 = hw.constant 7 :i32					


    %inner_gamma = SpecHLS.gamma @x %inner_ctrl:i3 ? %1,%2,%3,%4,%5,%6:i32
    %outer_gamma = SpecHLS.gamma @x %outer_ctrl:i2 ? %0,%inner_gamma,%7 :i32
    hw.output %outer_gamma: i32
  }
  

 hw.module @testbench_3_6(
 out out_0_0 : i32,
 out out_0_1 : i32,
 out out_0_2 : i32,
 out out_1_0 : i32,
 out out_1_1 : i32,
 out out_1_2 : i32,
 out out_2_0 : i32,
 out out_2_1 : i32,
 out out_2_2 : i32,
 out out_3_0 : i32,
 out out_3_1 : i32,
 out out_3_2 : i32,
 out out_4_0 : i32,
 out out_4_1 : i32,
 out out_4_2 : i32,
 out out_5_0 : i32,
 out out_5_1 : i32,
 out out_5_2 : i32
 ) {

	%inner0 = hw.constant 0 :i3
	%inner1 = hw.constant 1 :i3
	%inner2 = hw.constant 2 :i3
	%inner3 = hw.constant 3 :i3
	%inner4 = hw.constant 4 :i3
	%inner5 = hw.constant 5 :i3
	%outer0 = hw.constant 0 :i2
	%outer1 = hw.constant 1 :i2
	%outer2 = hw.constant 2 :i2

	%res00 = hw.instance "gamma0_0" @gamma_3_6(inner_ctrl : %inner0 :i3, outer_ctrl: %outer0: i2) -> (out_0: i32)
	%res01 = hw.instance "gamma0_1" @gamma_3_6(inner_ctrl : %inner0 :i3, outer_ctrl: %outer1: i2) -> (out_0: i32)
	%res02 = hw.instance "gamma0_2" @gamma_3_6(inner_ctrl : %inner0 :i3, outer_ctrl: %outer2: i2) -> (out_0: i32)
	%res10 = hw.instance "gamma1_0" @gamma_3_6(inner_ctrl : %inner1 :i3, outer_ctrl: %outer0: i2) -> (out_0: i32)
	%res11 = hw.instance "gamma1_1" @gamma_3_6(inner_ctrl : %inner1 :i3, outer_ctrl: %outer1: i2) -> (out_0: i32)
	%res12 = hw.instance "gamma1_2" @gamma_3_6(inner_ctrl : %inner1 :i3, outer_ctrl: %outer2: i2) -> (out_0: i32)
	%res20 = hw.instance "gamma2_0" @gamma_3_6(inner_ctrl : %inner2 :i3, outer_ctrl: %outer0: i2) -> (out_0: i32)
	%res21 = hw.instance "gamma2_1" @gamma_3_6(inner_ctrl : %inner2 :i3, outer_ctrl: %outer1: i2) -> (out_0: i32)
	%res22 = hw.instance "gamma2_2" @gamma_3_6(inner_ctrl : %inner2 :i3, outer_ctrl: %outer2: i2) -> (out_0: i32)
	%res30 = hw.instance "gamma3_0" @gamma_3_6(inner_ctrl : %inner3 :i3, outer_ctrl: %outer0: i2) -> (out_0: i32)
	%res31 = hw.instance "gamma3_1" @gamma_3_6(inner_ctrl : %inner3 :i3, outer_ctrl: %outer1: i2) -> (out_0: i32)
	%res32 = hw.instance "gamma3_2" @gamma_3_6(inner_ctrl : %inner3 :i3, outer_ctrl: %outer2: i2) -> (out_0: i32)
	%res40 = hw.instance "gamma4_0" @gamma_3_6(inner_ctrl : %inner4 :i3, outer_ctrl: %outer0: i2) -> (out_0: i32)
	%res41 = hw.instance "gamma4_1" @gamma_3_6(inner_ctrl : %inner4 :i3, outer_ctrl: %outer1: i2) -> (out_0: i32)
	%res42 = hw.instance "gamma4_2" @gamma_3_6(inner_ctrl : %inner4 :i3, outer_ctrl: %outer2: i2) -> (out_0: i32)
	%res50 = hw.instance "gamma5_0" @gamma_3_6(inner_ctrl : %inner5 :i3, outer_ctrl: %outer0: i2) -> (out_0: i32)
	%res51 = hw.instance "gamma5_1" @gamma_3_6(inner_ctrl : %inner5 :i3, outer_ctrl: %outer1: i2) -> (out_0: i32)
	%res52 = hw.instance "gamma5_2" @gamma_3_6(inner_ctrl : %inner5 :i3, outer_ctrl: %outer2: i2) -> (out_0: i32)

	hw.output 
	  %res00,  %res01,  %res02,   %res10,  %res11,  %res12,   %res20,  %res21,  %res22,   %res30,  %res31,  %res32,   %res40,  %res41,  %res42,   %res50,  %res51,  %res52  :		  
	  i32 , i32 , i32 ,  i32 , i32 , i32 ,  i32 , i32 , i32 ,  i32 , i32 , i32 ,  i32 , i32 , i32 ,  i32 , i32 , i32 		  

	  }
	  
}  

