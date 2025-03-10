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

	%t29,%t30,%t31,%t32,%t33,%t34,%t35,%t36,%t37,%t38,%t39,%t40 = hw.instance "inst_22" @fsmcommand  (in_0 : %t27 : !SpecHLS.struct<"statex_x0",i32,ui2,ui1,ui1,ui1,ui32,ui32,ui32,ui1,ui1,ui1,ui32,ui32,ui1,ui32,ui32,ui1,ui32,ui32,ui32:"state","rewindCpt","delayed_commit_0","delayed_commit_1","delayed_commit_2","array_rollback","mu_rollback","rewind","rbwe","commit_x","commit_x0","selSlowPath_x","rollback_x","startStall_x","selSlowPath_x0","rollback_x0","startStall_x0","rewindDepth","slowPath_x","slowPath_x0">) -> 
		(out_0 : i1,out_1 : i1,out_2 : i32,out_3 : i32,out_4 : i32,out_5 : i1,out_6 : i32,out_7 : i32,out_8 : i1,out_9 : i1,out_10 : i32,out_11 : i32) 
	
	%t41 = "hw.constant"() { value=1:i1} : () -> i1 
	%t42 = SpecHLS.delay %t41 -> %t3 by 3:i1 
	%t43 = "hw.constant"() { value=1:i1} : () -> i1 
	%t44 = SpecHLS.delay %t43 -> %t45 by 3:i32 
	%t46 = "hw.constant"() { value=1:i1} : () -> i1 
	%t47 = SpecHLS.delay %t46 -> %t48 by 3:i32 
	%t49 = "hw.constant"() { value=1:i1} : () -> i1 
	%t50 = SpecHLS.delay %t49 -> %t21 by 3:i1 
	%t51 = "SpecHLS.rollback"(%t2,%t31,%t34) { depths=[3:i32]} : (i1,i32,i1) -> i1 
	%t52 = "SpecHLS.cast"(%t51) { } : (i1) -> i32 
	%t53 = "SpecHLS.cast"(%t52) { } : (i32) -> i1 
	%t45 = "SpecHLS.rollback"(%t5,%t31,%t34) { depths=[3:i32]} : (i32,i32,i1) -> i32 
	%t54 = "SpecHLS.cast"(%t45) { } : (i32) -> i32 
	%t55in0 = "SpecHLS.cast"(%t54) { } : (i32) -> i32
	%t55in1 = "SpecHLS.cast"(%t10) { } : (i32) -> i32
	%t55out0 = "comb.add"(%t55in0,%t55in1) { } : (i32,i32) -> i32
	%t55 = "SpecHLS.cast"(%t55out0) { } : (i32) -> i32 
	%t56 = "SpecHLS.cast"(%t55) { } : (i32) -> i32 
	%t57 = "SpecHLS.rollback"(%t8,%t31,%t34) { depths=[3:i32]} : (i32,i32,i1) -> i32 
	%t57_1195615010 = SpecHLS.cast %t57:i32  to i32
					%t58 = hw.instance "call_43" @mispec  ( x :  %t57_1195615010 : i32 ) -> ( out0 : i1 ) 
	%t57_974606690 = SpecHLS.cast %t57:i32  to i32
					%t15 = hw.instance "call_44" @slow  ( x :  %t57_974606690 : i32 ) -> ( out0 : i32 ) 
	%t57_835631769 = SpecHLS.cast %t57:i32  to i32
					%t59 = hw.instance "call_45" @fast  ( x :  %t57_835631769 : i32 ) -> ( out0 : i32 ) 
	%t60 = "SpecHLS.cast"(%t58) { } : (i1) -> i32 
	%t18 = "SpecHLS.cast"(%t60) { } : (i32) -> i1 
	%t61 = "SpecHLS.custom"(%t53,%t18) { name="land"} : (i1,i1) -> i1 
	%t62 = "SpecHLS.custom"(%t18) { name="lnot"} : (i1) -> i1 
	%t63 = "SpecHLS.custom"(%t53,%t62) { name="land"} : (i1,i1) -> i1 
	%t21 = "SpecHLS.custom"(%t61,%t63) { name="lor"} : (i1,i1) -> i1 
	%t64_sel = SpecHLS.cast %t21: i1 to i1
	%t64 = SpecHLS.gamma @i %t64_sel : i1 ? %t45,%t56 :i32
	%t6_c = SpecHLS.cast %t64 : i32 to i32
	 		%t6 = SpecHLS.def @i %t6_c: i32
	%t65 = "SpecHLS.custom"(%t6,%t12) { name="lt"} : (i32,i32) -> i1 
	%t3_c = SpecHLS.cast %t65 : i1 to i1
	 		%t3 = SpecHLS.def @__guard %t3_c: i1
	%t66_sel = SpecHLS.cast %t39: i32 to i32
	%t66 = SpecHLS.gamma @x %t66_sel : i32 ? %t59,%t14 :i32
	%t48 = "SpecHLS.rollback"(%t66,%t35,%t34) { depths=[3:i32]} : (i32,i32,i1) -> i32 
	%t67_sel = SpecHLS.cast %t40: i32 to i32
	%t67 = SpecHLS.gamma @x %t67_sel : i32 ? %t57,%t48 :i32
	%t9_c = SpecHLS.cast %t67 : i32 to i32
	 		%t9 = SpecHLS.def @x %t9_c: i32
	//  [%t68, %t69, %t70, %t71] 
	//  [%t72, %t73, %t74, %t75] 
	"SpecHLS.commit"(%t42,%t30) { } : (i1,i1) -> () 
}


hw.module @SCC_13(in %in_0: i32,in %in_1: i32,in %in_2: i1,in %in_3: i1) {
	%t76 = SpecHLS.init @io_state : i32 
	%t77 = SpecHLS.mu @io_state : %t76,%t78 : i32 
	%t79 = SpecHLS.string "i=%08X, x = %08X\n" : memref<18xi8> 
	%t80 = SpecHLS.ioprintf "String_i=%08X, x = %08X\n" ( %t73 : i32,  %t74 : i32 ) from %t77 when %t75 { Schedule="(0, 0.0ns)"}
	%t81 = "SpecHLS.cast"(%t80) { } : (i32) -> i32 
	%t82_sel = SpecHLS.cast %t75: i1 to i1
	%t82 = SpecHLS.gamma @io_state %t82_sel : i1 ? %t77,%t81 :i32
	%t78_c = SpecHLS.cast %t82 : i32 to i32
	 		%t78 = SpecHLS.def @io_state %t78_c: i32
	%t83 = "SpecHLS.custom"(%t72) { name="lnot"} : (i1) -> i1 
	SpecHLS.exit %t83  
	//  [] 
	//  [] 
}

hw.module @IDG_SRC_0() attributes {"#pragma" = "toplevel"} {
	%t72,%t73,%t74,%t75 = hw.instance "inst_SpecSCC_12" @SpecSCC_12  () -> (out_0 : i1,out_1 : i32,out_2 : i32,out_3 : i1) 
	 hw.instance "inst_SCC_13" @SCC_13 (in_0 : %t73 : i32,in_1 : %t74 : i32,in_2 : %t75 : i1,in_3 : %t72 : i1) -> () 
}
