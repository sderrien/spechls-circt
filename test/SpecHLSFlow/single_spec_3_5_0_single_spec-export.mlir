module {
hw.module.extern @init_fsmx_x0( out "out0" : !SpecHLS.struct<"statex_x0",i32,ui2,ui1,ui1,ui1,ui32,ui32,ui32,ui1,ui1,ui1,ui32,ui32,ui1,ui32,ui32,ui1,ui32,ui32,ui32:"state","rewindCpt","delayed_commit_0","delayed_commit_1","delayed_commit_2","array_rollback","mu_rollback","rewind","rbwe","commit_x","commit_x0","selSlowPath_x","rollback_x","startStall_x","selSlowPath_x0","rollback_x0","startStall_x0","rewindDepth","slowPath_x","slowPath_x0"> )

hw.module.extern @mispec( in %x : i32,  out "out0" : i1 )

hw.module.extern @slow( in %x : i32,  out "out0" : i32 )

hw.module.extern @fast( in %x : i32,  out "out0" : i32 )

hw.module @SpecSCC_12(out "out_0": i1,out "out_1": i32,out "out_2": i32,out "out_3": i1) {
	%t1 = SpecHLS.init @__guard : i1 
	%t2 = SpecHLS.mu @__guard : %t1,%t3 : i1 
	%t4 = SpecHLS.init @i : i32 
	%t5 = SpecHLS.mu @i : %t4,%t6 : i32 
	%t7 = SpecHLS.init @x : i32 
	%t8 = SpecHLS.mu @x : %t7,%t9 : i32 
	%t10 = "hw.constant"() { value=1:i32} : () -> i32 
	%t11 = "hw.constant"() { value=1024:i32} : () -> i32 
	%t12 = "SpecHLS.cast"(%t11) { } : (i32) -> i32 
	%t13 = "hw.constant"() { value=1:i1} : () -> i1 
	%t14 = SpecHLS.delay %t13 -> %t15 by 3:i32 
	%t16 = "hw.constant"() { value=1:i1} : () -> i1 
	%t17 = SpecHLS.delay %t16 -> %t18 by 2:i1 
	%t19 = "hw.constant"() { value=1:i1} : () -> i1 
	%t20 = SpecHLS.delay %t19 -> %t21 by 2:i1 
	%t22 = "SpecHLS.cast"(%t17) { } : (i1) -> i32 
	%t23 = "SpecHLS.cast"(%t20) { } : (i1) -> i32 
	%t24 = "SpecHLS.custom"(%t22,%t23) { name="pack"} : (i32,i32) -> !SpecHLS.struct<"fsm_mispec_in_x_x0",i32,i32:"x","x0"> 
	%t25 = hw.instance "call_18" @init_fsmx_x0  () -> ( out0 : !SpecHLS.struct<"statex_x0",i32,ui2,ui1,ui1,ui1,ui32,ui32,ui32,ui1,ui1,ui1,ui32,ui32,ui1,ui32,ui32,ui1,ui32,ui32,ui32:"state","rewindCpt","delayed_commit_0","delayed_commit_1","delayed_commit_2","array_rollback","mu_rollback","rewind","rbwe","commit_x","commit_x0","selSlowPath_x","rollback_x","startStall_x","selSlowPath_x0","rollback_x0","startStall_x0","rewindDepth","slowPath_x","slowPath_x0"> ) 
	%t26 = "hw.constant"() { value=1:i1} : () -> i1 
	%t27 = SpecHLS.delay %t26 -> %t28 by 1 (%t25):!SpecHLS.struct<"statex_x0",i32,ui2,ui1,ui1,ui1,ui32,ui32,ui32,ui1,ui1,ui1,ui32,ui32,ui1,ui32,ui32,ui1,ui32,ui32,ui32:"state","rewindCpt","delayed_commit_0","delayed_commit_1","delayed_commit_2","array_rollback","mu_rollback","rewind","rbwe","commit_x","commit_x0","selSlowPath_x","rollback_x","startStall_x","selSlowPath_x0","rollback_x0","startStall_x0","rewindDepth","slowPath_x","slowPath_x0"> 
	%t28 = "SpecHLS.custom"(%t24,%t27) { name="fsm",nbspec=2,pipelineDepth=3} : (!SpecHLS.struct<"fsm_mispec_in_x_x0",i32,i32:"x","x0">,!SpecHLS.struct<"statex_x0",i32,ui2,ui1,ui1,ui1,ui32,ui32,ui32,ui1,ui1,ui1,ui32,ui32,ui1,ui32,ui32,ui1,ui32,ui32,ui32:"state","rewindCpt","delayed_commit_0","delayed_commit_1","delayed_commit_2","array_rollback","mu_rollback","rewind","rbwe","commit_x","commit_x0","selSlowPath_x","rollback_x","startStall_x","selSlowPath_x0","rollback_x0","startStall_x0","rewindDepth","slowPath_x","slowPath_x0">) -> !SpecHLS.struct<"statex_x0",i32,ui2,ui1,ui1,ui1,ui32,ui32,ui32,ui1,ui1,ui1,ui32,ui32,ui1,ui32,ui32,ui1,ui32,ui32,ui32:"state","rewindCpt","delayed_commit_0","delayed_commit_1","delayed_commit_2","array_rollback","mu_rollback","rewind","rbwe","commit_x","commit_x0","selSlowPath_x","rollback_x","startStall_x","selSlowPath_x0","rollback_x0","startStall_x0","rewindDepth","slowPath_x","slowPath_x0"> 
	%t29,%t30,%t31,%t32,%t33,%t34,%t35,%t36,%t37,%t38,%t39,%t40 = "SpecHLS.custom"(%t27) { name="fsmcommand"} : (!SpecHLS.struct<"statex_x0",i32,ui2,ui1,ui1,ui1,ui32,ui32,ui32,ui1,ui1,ui1,ui32,ui32,ui1,ui32,ui32,ui1,ui32,ui32,ui32:"state","rewindCpt","delayed_commit_0","delayed_commit_1","delayed_commit_2","array_rollback","mu_rollback","rewind","rbwe","commit_x","commit_x0","selSlowPath_x","rollback_x","startStall_x","selSlowPath_x0","rollback_x0","startStall_x0","rewindDepth","slowPath_x","slowPath_x0">) -> (i1,i1,i32,i32,i32,i1,i32,i32,i1,i1,i32,i32) 
	%t41 = "hw.constant"() { value=1:i1} : () -> i1 
	%t42 = SpecHLS.delay %t41 -> %t3 by 3:i1 
	%t43 = "SpecHLS.commit"(%t42,%t30) { } : (i1,i1) -> i1 
	%t44 = "hw.constant"() { value=1:i1} : () -> i1 
	%t45 = SpecHLS.delay %t44 -> %t46 by 3:i32 
	%t47 = "SpecHLS.commit"(%t45,%t30) { } : (i32,i1) -> i32 
	%t48 = "hw.constant"() { value=1:i1} : () -> i1 
	%t49 = SpecHLS.delay %t48 -> %t50 by 3:i32 
	%t51 = "SpecHLS.commit"(%t49,%t30) { } : (i32,i1) -> i32 
	%t52 = "hw.constant"() { value=1:i1} : () -> i1 
	%t53 = SpecHLS.delay %t52 -> %t21 by 3:i1 
	%t54 = "SpecHLS.commit"(%t53,%t30) { } : (i1,i1) -> i1 
	%t55 = "SpecHLS.rollback"(%t2,%t31,%t34) { } : (i1,i32,i1) -> i1 
	%t56 = "SpecHLS.cast"(%t55) { } : (i1) -> i32 
	%t57 = "SpecHLS.cast"(%t56) { } : (i32) -> i1 
	%t46 = "SpecHLS.rollback"(%t5,%t31,%t34) { } : (i32,i32,i1) -> i32 
	%t58 = "SpecHLS.cast"(%t46) { } : (i32) -> i32 
	%t59in0 = "SpecHLS.cast"(%t58) { } : (i32) -> i32
	%t59in1 = "SpecHLS.cast"(%t10) { } : (i32) -> i32
	%t59out0 = "comb.add"(%t59in0,%t59in1) { } : (i32,i32) -> i32
	%t59 = "SpecHLS.cast"(%t59out0) { } : (i32) -> i32 
	%t60 = "SpecHLS.cast"(%t59) { } : (i32) -> i32 
	%t61 = "SpecHLS.rollback"(%t8,%t31,%t34) { } : (i32,i32,i1) -> i32 
	%t61_1195615010 = SpecHLS.cast %t61:i32  to i32
	%t62 = hw.instance "call_43" @mispec  ( x :  %t61_1195615010 : i32 ) -> ( out0 : i1 ) 
	%t61_974606690 = SpecHLS.cast %t61:i32  to i32
	%t15 = hw.instance "call_44" @slow  ( x :  %t61_974606690 : i32 ) -> ( out0 : i32 ) 
	%t61_835631769 = SpecHLS.cast %t61:i32  to i32
	%t63 = hw.instance "call_45" @fast  ( x :  %t61_835631769 : i32 ) -> ( out0 : i32 ) 
	%t64 = "SpecHLS.cast"(%t62) { } : (i1) -> i32 
	%t18 = "SpecHLS.cast"(%t64) { } : (i32) -> i1 
	%t65 = "SpecHLS.custom"(%t57,%t18) { name="land"} : (i1,i1) -> i1 
	%t66 = "SpecHLS.custom"(%t18) { name="lnot"} : (i1) -> i1 
	%t67 = "SpecHLS.custom"(%t57,%t66) { name="land"} : (i1,i1) -> i1 
	%t21 = "SpecHLS.custom"(%t65,%t67) { name="lor"} : (i1,i1) -> i1 
	%t68_sel = SpecHLS.cast %t21: i1 to i1
	%t68 = SpecHLS.gamma @i %t68_sel : i1 ? %t46,%t60 :i32
	%t6_c = SpecHLS.cast %t68 : i32 to i32
	%t6 = SpecHLS.def @i %t6_c: i32
	%t69 = "SpecHLS.custom"(%t6,%t12) { name="lt"} : (i32,i32) -> i1 
	%t3_c = SpecHLS.cast %t69 : i1 to i1
	%t3 = SpecHLS.def @__guard %t3_c: i1
	%t70_sel = SpecHLS.cast %t39: i32 to i32
	%t70 = SpecHLS.gamma @x %t70_sel : i32 ? %t63,%t14 :i32
	%t50 = "SpecHLS.rollback"(%t70,%t35,%t34) { } : (i32,i32,i1) -> i32 
	%t71_sel = SpecHLS.cast %t40: i32 to i32
	%t71 = SpecHLS.gamma @x %t71_sel : i32 ? %t61,%t50 :i32
	%t9_c = SpecHLS.cast %t71 : i32 to i32
	%t9 = SpecHLS.def @x %t9_c: i32
	hw.output   %t43 , %t47 , %t51, %t54 : i1, i2,i3,i4 
}


hw.module @SCC_13(in %in_0: i32,in %in_1: i32,in %in_2: i1,in %in_3: i1) {
	%t73 = SpecHLS.init @io_state : i32 
	%t74 = SpecHLS.mu @io_state : %t73,%t75 : i32 
	%t76 = SpecHLS.string "i=%08X, x = %08X\n" : memref<18xi8> 
	%t77 = SpecHLS.ioprintf "String_i=%08X, x = %08X\n" ( %t78 : i32,  %t79 : i32 ) from %t74 when %t80 {Schedule->}
	%t81 = "SpecHLS.cast"(%t77) { } : (i32) -> i32 
	%t82_sel = SpecHLS.cast %t80: i1 to i1
	%t82 = SpecHLS.gamma @io_state %t82_sel : i1 ? %t74,%t81 :i32
	%t75_c = SpecHLS.cast %t82 : i32 to i32
	%t75 = SpecHLS.def @io_state %t75_c: i32
	%t83 = "SpecHLS.custom"(%t72) { name="lnot"} : (i1) -> i1 
	SpecHLS.exit %t83  live     
}

hw.module @IDG_SRC_0() attributes {"#pragma" = "toplevel"} {
	%t72,%t72,%t72,%t72 = hw.instance "inst_SpecSCC_12" @SpecSCC_12  (, out %out0 : i1, out %out1 : i32, out %out2 : i32, out %out3 : i1 ) 
	 = hw.instance "inst_SCC_13" @SCC_13  ( in %in0 : %t78 : i32,  in %in1 : %t79 : i32,  in %in2 : %t80 : i1,  in %in3 : %t72 : i1 , ) 
}
}