fsm.machine @SpecSCC_87_fsm(%mispec_l_x: i8,%mispec_l_x0: i8) -> (i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1) 
	attributes {initialState = "Init0"} {
fsm.state @Proceed output  {

	 %t0 = hw.constant 1 : i1 
	
	 %t1 = hw.constant 1 : i1 
	
	 %t2 = hw.constant 1 : i1 
	
	 %t3 = hw.constant 1 : i8 
	
	 %t4 = hw.constant 1 : i8 
	
	 %t5 = hw.constant 1 : i1 
	
	 %t6 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t0,%t5,%t6,%t3,%default_rollback_l_x,%default_startStall_l_x,%t4,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x0__Rollback guard  {
		 %t7 = hw.constant 0 : i8 
		%t8 = comb.icmp eq %mispec_l_x,%t7 : i8
	  fsm.return %t8
	} 
	fsm.transition @l_x00__Rollback guard  {
		 %t9 = hw.constant 0 : i8 
		%t10 = comb.icmp eq %mispec_l_x0,%t9 : i8
	  fsm.return %t10
	} 
}
fsm.state @l_x0__Rollback output  {

	 %t11 = hw.constant 1 : i1 
	
	 %t12 = hw.constant 1 : i8 
	
	 %t13 = hw.constant 1 : i8 
	
	 %t14 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t11,%default_commit_l_x,%t14,%t12,%default_rollback_l_x,%default_startStall_l_x,%t13,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x0__Fill0 guard  {
		%t15 = hw.constant 1 : i1
	  fsm.return %t15
	} 
}
fsm.state @l_x0__Fill0 output  {

	 %t16 = hw.constant 1 : i1 
	
	 %t17 = hw.constant 1 : i8 
	
	 %t18 = hw.constant 1 : i8 
	
	 %t19 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t16,%default_commit_l_x,%t19,%t17,%default_rollback_l_x,%default_startStall_l_x,%t18,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x0__Fill1 guard  {
		%t20 = hw.constant 1 : i1
	  fsm.return %t20
	} 
}
fsm.state @l_x0__Fill1 output  {

	 %t21 = hw.constant 1 : i1 
	
	 %t22 = hw.constant 1 : i8 
	
	 %t23 = hw.constant 1 : i8 
	
	 %t24 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t21,%default_commit_l_x,%t24,%t22,%default_rollback_l_x,%default_startStall_l_x,%t23,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x0__Fill2 guard  {
		%t25 = hw.constant 1 : i1
	  fsm.return %t25
	} 
}
fsm.state @l_x0__Fill2 output  {

	 %t26 = hw.constant 1 : i1 
	
	 %t27 = hw.constant 1 : i8 
	
	 %t28 = hw.constant 1 : i8 
	
	 %t29 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t26,%default_commit_l_x,%t29,%t27,%default_rollback_l_x,%default_startStall_l_x,%t28,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x0__Fill3 guard  {
		%t30 = hw.constant 1 : i1
	  fsm.return %t30
	} 
}
fsm.state @l_x0__Fill3 output  {

	 %t31 = hw.constant 1 : i1 
	
	 %t32 = hw.constant 1 : i8 
	
	 %t33 = hw.constant 1 : i8 
	
	 %t34 = hw.constant 1 : i1 
	
	 %t35 = hw.constant 1 : i8 
	
	 %t36 = hw.constant 1 : i8 
	
	 %t37 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t31,%t34,%t37,%t35,%default_rollback_l_x,%default_startStall_l_x,%t36,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t38 = hw.constant 1 : i1
	  fsm.return %t38
	} 
	fsm.transition @l_x0_l_x00__Rollback guard  {
		 %t39 = hw.constant 0 : i8 
		%t40 = comb.icmp eq %mispec_l_x0,%t39 : i8
	  fsm.return %t40
	} 
}
fsm.state @l_x00__Rollback output  {

	 %t41 = hw.constant 1 : i1 
	
	 %t42 = hw.constant 1 : i8 
	
	 %t43 = hw.constant 1 : i8 
	
	 %t44 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t41,%t44,%default_commit_l_x0,%t43,%default_rollback_l_x,%default_startStall_l_x,%t42,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x00__Fill0 guard  {
		%t45 = hw.constant 1 : i1
	  fsm.return %t45
	} 
}
fsm.state @l_x00__Fill0 output  {

	 %t46 = hw.constant 1 : i1 
	
	 %t47 = hw.constant 1 : i8 
	
	 %t48 = hw.constant 1 : i8 
	
	 %t49 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t46,%t49,%default_commit_l_x0,%t48,%default_rollback_l_x,%default_startStall_l_x,%t47,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x00__Fill1 guard  {
		%t50 = hw.constant 1 : i1
	  fsm.return %t50
	} 
}
fsm.state @l_x00__Fill1 output  {

	 %t51 = hw.constant 1 : i1 
	
	 %t52 = hw.constant 1 : i8 
	
	 %t53 = hw.constant 1 : i8 
	
	 %t54 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t51,%t54,%default_commit_l_x0,%t53,%default_rollback_l_x,%default_startStall_l_x,%t52,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x00__Fill2 guard  {
		%t55 = hw.constant 1 : i1
	  fsm.return %t55
	} 
}
fsm.state @l_x00__Fill2 output  {

	 %t56 = hw.constant 1 : i1 
	
	 %t57 = hw.constant 1 : i8 
	
	 %t58 = hw.constant 1 : i8 
	
	 %t59 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t56,%t59,%default_commit_l_x0,%t58,%default_rollback_l_x,%default_startStall_l_x,%t57,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x00__Fill3 guard  {
		%t60 = hw.constant 1 : i1
	  fsm.return %t60
	} 
}
fsm.state @l_x00__Fill3 output  {

	 %t61 = hw.constant 1 : i1 
	
	 %t62 = hw.constant 1 : i8 
	
	 %t63 = hw.constant 1 : i8 
	
	 %t64 = hw.constant 1 : i1 
	
	 %t65 = hw.constant 1 : i8 
	
	 %t66 = hw.constant 1 : i8 
	
	 %t67 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t61,%t67,%t64,%t65,%default_rollback_l_x,%default_startStall_l_x,%t66,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t68 = hw.constant 1 : i1
	  fsm.return %t68
	} 
	fsm.transition @l_x0__Rollback guard  {
		 %t69 = hw.constant 0 : i8 
		%t70 = comb.icmp eq %mispec_l_x,%t69 : i8
	  fsm.return %t70
	} 
}
fsm.state @l_x0_l_x00__Rollback output  {

	 %t71 = hw.constant 1 : i1 
	
	 %t72 = hw.constant 1 : i8 
	
	 %t73 = hw.constant 1 : i8 
	
	 %t74 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t71,%t74,%default_commit_l_x0,%t73,%default_rollback_l_x,%default_startStall_l_x,%t72,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x0_l_x00__Fill0 guard  {
		%t75 = hw.constant 1 : i1
	  fsm.return %t75
	} 
}
fsm.state @l_x0_l_x00__Fill0 output  {

	 %t76 = hw.constant 1 : i1 
	
	 %t77 = hw.constant 1 : i8 
	
	 %t78 = hw.constant 1 : i8 
	
	 %t79 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t76,%t79,%default_commit_l_x0,%t78,%default_rollback_l_x,%default_startStall_l_x,%t77,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x0_l_x00__Fill1 guard  {
		%t80 = hw.constant 1 : i1
	  fsm.return %t80
	} 
}
fsm.state @l_x0_l_x00__Fill1 output  {

	 %t81 = hw.constant 1 : i1 
	
	 %t82 = hw.constant 1 : i8 
	
	 %t83 = hw.constant 1 : i8 
	
	 %t84 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t81,%t84,%default_commit_l_x0,%t83,%default_rollback_l_x,%default_startStall_l_x,%t82,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x0_l_x00__Fill2 guard  {
		%t85 = hw.constant 1 : i1
	  fsm.return %t85
	} 
}
fsm.state @l_x0_l_x00__Fill2 output  {

	 %t86 = hw.constant 1 : i1 
	
	 %t87 = hw.constant 1 : i8 
	
	 %t88 = hw.constant 1 : i8 
	
	 %t89 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t86,%t89,%default_commit_l_x0,%t88,%default_rollback_l_x,%default_startStall_l_x,%t87,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x0_l_x00__Fill3 guard  {
		%t90 = hw.constant 1 : i1
	  fsm.return %t90
	} 
}
fsm.state @l_x0_l_x00__Fill3 output  {

	 %t91 = hw.constant 1 : i1 
	
	 %t92 = hw.constant 1 : i8 
	
	 %t93 = hw.constant 1 : i8 
	
	 %t94 = hw.constant 1 : i1 
	
	 %t95 = hw.constant 1 : i1 
	
	 %t96 = hw.constant 1 : i8 
	
	 %t97 = hw.constant 1 : i8 
	
	 %t98 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t91,%t98,%t95,%t96,%default_rollback_l_x,%default_startStall_l_x,%t97,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t99 = hw.constant 1 : i1
	  fsm.return %t99
	} 
}
fsm.state @Init0 output  {

	 %t100 = hw.constant 1 : i1 
	
	 %t101 = hw.constant 1 : i8 
	
	 %t102 = hw.constant 1 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x = hw.constant 0 : i1
	
	%default_commit_l_x0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t100,%default_commit_l_x,%default_commit_l_x0,%t101,%default_rollback_l_x,%default_startStall_l_x,%t102,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Init1 guard  {
		%t103 = hw.constant 1 : i1
	  fsm.return %t103
	} 
}
fsm.state @Init1 output  {

	 %t104 = hw.constant 1 : i1 
	
	 %t105 = hw.constant 1 : i8 
	
	 %t106 = hw.constant 1 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x = hw.constant 0 : i1
	
	%default_commit_l_x0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t104,%default_commit_l_x,%default_commit_l_x0,%t105,%default_rollback_l_x,%default_startStall_l_x,%t106,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Init2 guard  {
		%t107 = hw.constant 1 : i1
	  fsm.return %t107
	} 
}
fsm.state @Init2 output  {

	 %t108 = hw.constant 1 : i1 
	
	 %t109 = hw.constant 1 : i8 
	
	 %t110 = hw.constant 1 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x = hw.constant 0 : i1
	
	%default_commit_l_x0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t108,%default_commit_l_x,%default_commit_l_x0,%t109,%default_rollback_l_x,%default_startStall_l_x,%t110,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Init3 guard  {
		%t111 = hw.constant 1 : i1
	  fsm.return %t111
	} 
}
fsm.state @Init3 output  {

	 %t112 = hw.constant 1 : i1 
	
	 %t113 = hw.constant 1 : i8 
	
	 %t114 = hw.constant 1 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x = hw.constant 0 : i1
	
	%default_commit_l_x0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x0 = hw.constant 0 : i8
	
	%default_startStall_l_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t112,%default_commit_l_x,%default_commit_l_x0,%t113,%default_rollback_l_x,%default_startStall_l_x,%t114,%default_rollback_l_x0,%default_startStall_l_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t115 = hw.constant 1 : i1
	  fsm.return %t115
	} 
}
}