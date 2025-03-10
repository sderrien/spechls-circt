fsm.machine @SpecSCC_348_fsm(%mispec_l_pc: i8,%mispec_l_pc0: i8) -> (i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1) 
	attributes {initialState = "Init0"} {
fsm.state @Proceed output  {

	 %t0 = hw.constant 1 : i1 
	
	 %t1 = hw.constant 1 : i1 
	
	 %t2 = hw.constant 1 : i1 
	
	 %t3 = hw.constant 0 : i8 
	
	 %t4 = hw.constant 1 : i8 
	
	 %t5 = hw.constant 1 : i1 
	
	 %t6 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_l_pc = hw.constant 0 : i8
	
	%default_startStall_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc0 = hw.constant 0 : i8
	
	%default_startStall_l_pc0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t0,%t5,%t6,%t3,%default_rollback_l_pc,%default_startStall_l_pc,%t4,%default_rollback_l_pc0,%default_startStall_l_pc0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_pc1__Rollback guard  {
		 %t7 = hw.constant 1 : i8 
		%t8 = comb.icmp eq %mispec_l_pc,%t7 : i8
	  fsm.return %t8
	} 
	fsm.transition @l_pc00__Rollback guard  {
		 %t9 = hw.constant 0 : i8 
		%t10 = comb.icmp eq %mispec_l_pc0,%t9 : i8
	  fsm.return %t10
	} 
}
fsm.state @l_pc1__Rollback output  {

	 %t11 = hw.constant 1 : i1 
	
	 %t12 = hw.constant 0 : i8 
	
	 %t13 = hw.constant 1 : i8 
	
	 %t14 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc = hw.constant 0 : i8
	
	%default_startStall_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc0 = hw.constant 0 : i8
	
	%default_startStall_l_pc0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t11,%default_commit_l_pc,%t14,%t12,%default_rollback_l_pc,%default_startStall_l_pc,%t13,%default_rollback_l_pc0,%default_startStall_l_pc0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_pc1__Fill0 guard  {
		%t15 = hw.constant 1 : i1
	  fsm.return %t15
	} 
}
fsm.state @l_pc1__Fill0 output  {

	 %t16 = hw.constant 1 : i1 
	
	 %t17 = hw.constant 0 : i8 
	
	 %t18 = hw.constant 1 : i8 
	
	 %t19 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc = hw.constant 0 : i8
	
	%default_startStall_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc0 = hw.constant 0 : i8
	
	%default_startStall_l_pc0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t16,%default_commit_l_pc,%t19,%t17,%default_rollback_l_pc,%default_startStall_l_pc,%t18,%default_rollback_l_pc0,%default_startStall_l_pc0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_pc1__Fill1 guard  {
		%t20 = hw.constant 1 : i1
	  fsm.return %t20
	} 
}
fsm.state @l_pc1__Fill1 output  {

	 %t21 = hw.constant 1 : i1 
	
	 %t22 = hw.constant 0 : i8 
	
	 %t23 = hw.constant 1 : i8 
	
	 %t24 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc = hw.constant 0 : i8
	
	%default_startStall_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc0 = hw.constant 0 : i8
	
	%default_startStall_l_pc0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t21,%default_commit_l_pc,%t24,%t22,%default_rollback_l_pc,%default_startStall_l_pc,%t23,%default_rollback_l_pc0,%default_startStall_l_pc0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_pc1__Fill2 guard  {
		%t25 = hw.constant 1 : i1
	  fsm.return %t25
	} 
}
fsm.state @l_pc1__Fill2 output  {

	 %t26 = hw.constant 1 : i1 
	
	 %t27 = hw.constant 0 : i8 
	
	 %t28 = hw.constant 1 : i8 
	
	 %t29 = hw.constant 1 : i1 
	
	 %t30 = hw.constant 0 : i8 
	
	 %t31 = hw.constant 1 : i8 
	
	 %t32 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_l_pc = hw.constant 0 : i8
	
	%default_startStall_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc0 = hw.constant 0 : i8
	
	%default_startStall_l_pc0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t26,%t29,%t32,%t30,%default_rollback_l_pc,%default_startStall_l_pc,%t31,%default_rollback_l_pc0,%default_startStall_l_pc0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t33 = hw.constant 1 : i1
	  fsm.return %t33
	} 
	fsm.transition @l_pc1_l_pc00__Rollback guard  {
		 %t34 = hw.constant 0 : i8 
		%t35 = comb.icmp eq %mispec_l_pc0,%t34 : i8
	  fsm.return %t35
	} 
}
fsm.state @l_pc00__Rollback output  {

	 %t36 = hw.constant 1 : i1 
	
	 %t37 = hw.constant 1 : i8 
	
	 %t38 = hw.constant 0 : i8 
	
	 %t39 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_pc0 = hw.constant 0 : i1
	
	%default_rollback_l_pc = hw.constant 0 : i8
	
	%default_startStall_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc0 = hw.constant 0 : i8
	
	%default_startStall_l_pc0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t36,%t39,%default_commit_l_pc0,%t38,%default_rollback_l_pc,%default_startStall_l_pc,%t37,%default_rollback_l_pc0,%default_startStall_l_pc0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_pc00__Fill0 guard  {
		%t40 = hw.constant 1 : i1
	  fsm.return %t40
	} 
}
fsm.state @l_pc00__Fill0 output  {

	 %t41 = hw.constant 1 : i1 
	
	 %t42 = hw.constant 1 : i8 
	
	 %t43 = hw.constant 0 : i8 
	
	 %t44 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_pc0 = hw.constant 0 : i1
	
	%default_rollback_l_pc = hw.constant 0 : i8
	
	%default_startStall_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc0 = hw.constant 0 : i8
	
	%default_startStall_l_pc0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t41,%t44,%default_commit_l_pc0,%t43,%default_rollback_l_pc,%default_startStall_l_pc,%t42,%default_rollback_l_pc0,%default_startStall_l_pc0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_pc00__Fill1 guard  {
		%t45 = hw.constant 1 : i1
	  fsm.return %t45
	} 
}
fsm.state @l_pc00__Fill1 output  {

	 %t46 = hw.constant 1 : i1 
	
	 %t47 = hw.constant 1 : i8 
	
	 %t48 = hw.constant 0 : i8 
	
	 %t49 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_pc0 = hw.constant 0 : i1
	
	%default_rollback_l_pc = hw.constant 0 : i8
	
	%default_startStall_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc0 = hw.constant 0 : i8
	
	%default_startStall_l_pc0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t46,%t49,%default_commit_l_pc0,%t48,%default_rollback_l_pc,%default_startStall_l_pc,%t47,%default_rollback_l_pc0,%default_startStall_l_pc0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_pc00__Fill2 guard  {
		%t50 = hw.constant 1 : i1
	  fsm.return %t50
	} 
}
fsm.state @l_pc00__Fill2 output  {

	 %t51 = hw.constant 1 : i1 
	
	 %t52 = hw.constant 1 : i8 
	
	 %t53 = hw.constant 0 : i8 
	
	 %t54 = hw.constant 1 : i1 
	
	 %t55 = hw.constant 0 : i8 
	
	 %t56 = hw.constant 1 : i8 
	
	 %t57 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_l_pc = hw.constant 0 : i8
	
	%default_startStall_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc0 = hw.constant 0 : i8
	
	%default_startStall_l_pc0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t51,%t57,%t54,%t55,%default_rollback_l_pc,%default_startStall_l_pc,%t56,%default_rollback_l_pc0,%default_startStall_l_pc0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t58 = hw.constant 1 : i1
	  fsm.return %t58
	} 
	fsm.transition @l_pc1__Rollback guard  {
		 %t59 = hw.constant 1 : i8 
		%t60 = comb.icmp eq %mispec_l_pc,%t59 : i8
	  fsm.return %t60
	} 
}
fsm.state @l_pc1_l_pc00__Rollback output  {

	 %t61 = hw.constant 1 : i1 
	
	 %t62 = hw.constant 1 : i8 
	
	 %t63 = hw.constant 0 : i8 
	
	 %t64 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_pc0 = hw.constant 0 : i1
	
	%default_rollback_l_pc = hw.constant 0 : i8
	
	%default_startStall_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc0 = hw.constant 0 : i8
	
	%default_startStall_l_pc0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t61,%t64,%default_commit_l_pc0,%t63,%default_rollback_l_pc,%default_startStall_l_pc,%t62,%default_rollback_l_pc0,%default_startStall_l_pc0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_pc1_l_pc00__Fill0 guard  {
		%t65 = hw.constant 1 : i1
	  fsm.return %t65
	} 
}
fsm.state @l_pc1_l_pc00__Fill0 output  {

	 %t66 = hw.constant 1 : i1 
	
	 %t67 = hw.constant 1 : i8 
	
	 %t68 = hw.constant 0 : i8 
	
	 %t69 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_pc0 = hw.constant 0 : i1
	
	%default_rollback_l_pc = hw.constant 0 : i8
	
	%default_startStall_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc0 = hw.constant 0 : i8
	
	%default_startStall_l_pc0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t66,%t69,%default_commit_l_pc0,%t68,%default_rollback_l_pc,%default_startStall_l_pc,%t67,%default_rollback_l_pc0,%default_startStall_l_pc0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_pc1_l_pc00__Fill1 guard  {
		%t70 = hw.constant 1 : i1
	  fsm.return %t70
	} 
}
fsm.state @l_pc1_l_pc00__Fill1 output  {

	 %t71 = hw.constant 1 : i1 
	
	 %t72 = hw.constant 1 : i8 
	
	 %t73 = hw.constant 0 : i8 
	
	 %t74 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_pc0 = hw.constant 0 : i1
	
	%default_rollback_l_pc = hw.constant 0 : i8
	
	%default_startStall_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc0 = hw.constant 0 : i8
	
	%default_startStall_l_pc0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t71,%t74,%default_commit_l_pc0,%t73,%default_rollback_l_pc,%default_startStall_l_pc,%t72,%default_rollback_l_pc0,%default_startStall_l_pc0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_pc1_l_pc00__Fill2 guard  {
		%t75 = hw.constant 1 : i1
	  fsm.return %t75
	} 
}
fsm.state @l_pc1_l_pc00__Fill2 output  {

	 %t76 = hw.constant 1 : i1 
	
	 %t77 = hw.constant 1 : i8 
	
	 %t78 = hw.constant 0 : i8 
	
	 %t79 = hw.constant 1 : i1 
	
	 %t80 = hw.constant 1 : i1 
	
	 %t81 = hw.constant 0 : i8 
	
	 %t82 = hw.constant 1 : i8 
	
	 %t83 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_l_pc = hw.constant 0 : i8
	
	%default_startStall_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc0 = hw.constant 0 : i8
	
	%default_startStall_l_pc0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t76,%t83,%t80,%t81,%default_rollback_l_pc,%default_startStall_l_pc,%t82,%default_rollback_l_pc0,%default_startStall_l_pc0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t84 = hw.constant 1 : i1
	  fsm.return %t84
	} 
}
fsm.state @Init0 output  {

	 %t85 = hw.constant 1 : i1 
	
	 %t86 = hw.constant 0 : i8 
	
	 %t87 = hw.constant 1 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_pc = hw.constant 0 : i1
	
	%default_commit_l_pc0 = hw.constant 0 : i1
	
	%default_rollback_l_pc = hw.constant 0 : i8
	
	%default_startStall_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc0 = hw.constant 0 : i8
	
	%default_startStall_l_pc0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t85,%default_commit_l_pc,%default_commit_l_pc0,%t86,%default_rollback_l_pc,%default_startStall_l_pc,%t87,%default_rollback_l_pc0,%default_startStall_l_pc0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Init1 guard  {
		%t88 = hw.constant 1 : i1
	  fsm.return %t88
	} 
}
fsm.state @Init1 output  {

	 %t89 = hw.constant 1 : i1 
	
	 %t90 = hw.constant 0 : i8 
	
	 %t91 = hw.constant 1 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_pc = hw.constant 0 : i1
	
	%default_commit_l_pc0 = hw.constant 0 : i1
	
	%default_rollback_l_pc = hw.constant 0 : i8
	
	%default_startStall_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc0 = hw.constant 0 : i8
	
	%default_startStall_l_pc0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t89,%default_commit_l_pc,%default_commit_l_pc0,%t90,%default_rollback_l_pc,%default_startStall_l_pc,%t91,%default_rollback_l_pc0,%default_startStall_l_pc0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Init2 guard  {
		%t92 = hw.constant 1 : i1
	  fsm.return %t92
	} 
}
fsm.state @Init2 output  {

	 %t93 = hw.constant 1 : i1 
	
	 %t94 = hw.constant 0 : i8 
	
	 %t95 = hw.constant 1 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_pc = hw.constant 0 : i1
	
	%default_commit_l_pc0 = hw.constant 0 : i1
	
	%default_rollback_l_pc = hw.constant 0 : i8
	
	%default_startStall_l_pc = hw.constant 0 : i1
	
	%default_rollback_l_pc0 = hw.constant 0 : i8
	
	%default_startStall_l_pc0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t93,%default_commit_l_pc,%default_commit_l_pc0,%t94,%default_rollback_l_pc,%default_startStall_l_pc,%t95,%default_rollback_l_pc0,%default_startStall_l_pc0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t96 = hw.constant 1 : i1
	  fsm.return %t96
	} 
}
}