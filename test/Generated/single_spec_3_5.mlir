module {
hw.module.extern @init_fsmx_x0( out "out0" : !SpecHLS.struct<"statex_x0",i32,ui2,ui1,ui1,ui1,ui32,ui32,ui32,ui1,ui1,ui1,ui32,ui32,ui1,ui32,ui32,ui1,ui32,ui32,ui32:"state","rewindCpt","delayed_commit_0","delayed_commit_1","delayed_commit_2","array_rollback","mu_rollback","rewind","rbwe","commit_x","commit_x0","selSlowPath_x","rollback_x","startStall_x","selSlowPath_x0","rollback_x0","startStall_x0","rewindDepth","slowPath_x","slowPath_x0"> )

hw.module.extern @mispec( in %x : i32,  out "out0" : i1 )

hw.module.extern @slow( in %x : i32,  out "out0" : i32 )

hw.module.extern @fast( in %x : i32,  out "out0" : i32 )

SpecHLS.hkernel @IDG_SRC_0 -> {
	%t82:7 = SpecHLS.htask @SpecSCC_13() -> (i1,i32,i32,i32,i32,i1,i1) {
	^body():
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
	%t29,%t30,%t31,%t32,%t33,%t34,%t35,%t36,%t37,%t38,%t39,%t40 = hw.instance "inst_22" @fsmcommand  (in_0 : %t27 : !SpecHLS.struct<"statex_x0",i32,ui2,ui1,ui1,ui1,ui32,ui32,ui32,ui1,ui1,ui1,ui32,ui32,ui1,ui32,ui32,ui1,ui32,ui32,ui32:"state","rewindCpt","delayed_commit_0","delayed_commit_1","delayed_commit_2","array_rollback","mu_rollback","rewind","rbwe","commit_x","commit_x0","selSlowPath_x","rollback_x","startStall_x","selSlowPath_x0","rollback_x0","startStall_x0","rewindDepth","slowPath_x","slowPath_x0">) -> (out_0 : i1,out_1 : i1,out_2 : i32,out_3 : i32,out_4 : i32,out_5 : i1,out_6 : i32,out_7 : i32,out_8 : i1,out_9 : i1,out_10 : i32,out_11 : i32) 
	%t41 = "hw.constant"() { value=1:i1} : () -> i1 
	%t42 = SpecHLS.delay %t41 -> %t3 by 3:i1 
	%t43 = "hw.constant"() { value=1:i1} : () -> i1 
	%t44 = SpecHLS.delay %t43 -> %t6 by 3:i32 
	%t45 = "hw.constant"() { value=1:i1} : () -> i1 
	%t46 = SpecHLS.delay %t45 -> %t9 by 3:i32 
	%t47 = "hw.constant"() { value=1:i1} : () -> i1 
	%t48 = SpecHLS.delay %t47 -> %t49 by 3:i32 
	%t50 = "hw.constant"() { value=1:i1} : () -> i1 
	%t51 = SpecHLS.delay %t50 -> %t52 by 3:i32 
	%t53 = "hw.constant"() { value=1:i1} : () -> i1 
	%t54 = SpecHLS.delay %t53 -> %t21 by 3:i1 
	%t55 = "hw.constant"() { value=1:i1} : () -> i1 
	%t56 = SpecHLS.delay %t55 -> %t57 by 3:i1 
	%t58 = "SpecHLS.rollback"(%t2,%t31,%t34) { depths=[3:i32]} : (i1,i32,i1) -> i1 
	%t59 = "SpecHLS.cast"(%t58) { } : (i1) -> i32 
	%t60 = "SpecHLS.cast"(%t59) { } : (i32) -> i1 
	%t49 = "SpecHLS.rollback"(%t5,%t31,%t34) { depths=[3:i32]} : (i32,i32,i1) -> i32 
	%t61 = "SpecHLS.cast"(%t49) { } : (i32) -> i32 
	%t62in0 = "SpecHLS.cast"(%t61) { } : (i32) -> i32
	%t62in1 = "SpecHLS.cast"(%t10) { } : (i32) -> i32
	%t62out0 = "comb.add"(%t62in0,%t62in1) { } : (i32,i32) -> i32
	%t62 = "SpecHLS.cast"(%t62out0) { } : (i32) -> i32 
	%t63 = "SpecHLS.cast"(%t62) { } : (i32) -> i32 
	%t64 = "SpecHLS.rollback"(%t8,%t31,%t34) { depths=[3:i32]} : (i32,i32,i1) -> i32 
	%t64_1081635795 = SpecHLS.cast %t64:i32  to i32
					%t65 = hw.instance "call_52" @mispec  ( x :  %t64_1081635795 : i32 ) -> ( out0 : i1 ) 
	%t64_1289454852 = SpecHLS.cast %t64:i32  to i32
					%t15 = hw.instance "call_53" @slow  ( x :  %t64_1289454852 : i32 ) -> ( out0 : i32 ) 
	%t64_140110402 = SpecHLS.cast %t64:i32  to i32
					%t66 = hw.instance "call_54" @fast  ( x :  %t64_140110402 : i32 ) -> ( out0 : i32 ) 
	%t67 = "SpecHLS.cast"(%t65) { } : (i1) -> i32 
	%t18 = "SpecHLS.cast"(%t67) { } : (i32) -> i1 
	%t68 = "SpecHLS.custom"(%t60,%t18) { name="land"} : (i1,i1) -> i1 
	%t69 = "SpecHLS.custom"(%t18) { name="lnot"} : (i1) -> i1 
	%t70 = "SpecHLS.custom"(%t60,%t69) { name="land"} : (i1,i1) -> i1 
	%t21 = "SpecHLS.custom"(%t68,%t70) { name="lor"} : (i1,i1) -> i1 
	%t71_sel = SpecHLS.cast %t21: i1 to i1
	%t71 = SpecHLS.gamma @i %t71_sel : i1 ? %t49,%t63 :i32
	%t6_c = SpecHLS.cast %t71 : i32 to i32
	 		%t6 = SpecHLS.def @i %t6_c: i32
	%t72 = "SpecHLS.custom"(%t6,%t12) { name="lt"} : (i32,i32) -> i1 
	%t3_c = SpecHLS.cast %t72 : i1 to i1
	 		%t3 = SpecHLS.def @__guard %t3_c: i1
	%t57 = "SpecHLS.custom"(%t3) { name="lnot"} : (i1) -> i1 
	%t73_sel = SpecHLS.cast %t39: i32 to i32
	%t73 = SpecHLS.gamma @x %t73_sel : i32 ? %t66,%t14 :i32
	%t52 = "SpecHLS.rollback"(%t73,%t35,%t34) { depths=[3:i32]} : (i32,i32,i1) -> i32 
	%t74_sel = SpecHLS.cast %t40: i32 to i32
	%t74 = SpecHLS.gamma @x %t74_sel : i32 ? %t64,%t52 :i32
	%t9_c = SpecHLS.cast %t74 : i32 to i32
	 		%t9 = SpecHLS.def @x %t9_c: i32
	}
	%t96:1 = SpecHLS.htask @SCC_14(%t85:i32,%t86:i32,%t87:i1,%t88:i1,%t82:i1,%t83:i32,%t84:i32 ) -> () {
			^body():
			%t89 = SpecHLS.init @io_state : i32 
			%t90 = SpecHLS.mu @io_state : %t89,%t91 : i32 
			%t92 = SpecHLS.string "i=%08X, x = %08X\n" : memref<18xi8> 
			%t93 = SpecHLS.ioprintf "i=%08X, x = %08X\n" ( %t85 : i32,  %t86 : i32 ) from %t90 when %t87 { Schedule="(0, 0.0ns)"}
			%t94 = "SpecHLS.cast"(%t93) { } : (i32) -> i32 
			%t95_sel = SpecHLS.cast %t87: i1 to i1
			%t95 = SpecHLS.gamma @io_state %t95_sel : i1 ? %t90,%t94 :i32
			%t91_c = SpecHLS.cast %t95 : i32 to i32
	 		%t91 = SpecHLS.def @io_state %t91_c: i32
			SpecHLS.exit %t88 live  %t91:i32,%t82:i1,%t83:i32,%t84:i32   
	}
}
}
