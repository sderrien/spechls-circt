fsm.machine @SpecSCC_168_fsm(%mispec_x: i8,%mispec_x0: i8) -> (i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1) 
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
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t0,%t5,%t6,%t3,%default_rollback_x,%default_startStall_x,%t4,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @x1__Stall0 guard  {
		 %t7 = hw.constant 1 : i8 
		%t8 = comb.icmp eq %mispec_x,%t7 : i8
	  fsm.return %t8
	} 
	fsm.transition @x00__Rollback guard  {
		 %t9 = hw.constant 0 : i8 
		%t10 = comb.icmp eq %mispec_x0,%t9 : i8
	  fsm.return %t10
	} 
}
fsm.state @x1__Stall0 output  {

	 %t11 = hw.constant 1 : i8 
	
	 %t12 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rbwe = hw.constant 0 : i1
	
	%default_commit_x = hw.constant 0 : i1
	
	%default_selSlowPath_x = hw.constant 0 : i8
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%default_rbwe,%default_commit_x,%t12,%default_selSlowPath_x,%default_rollback_x,%default_startStall_x,%t11,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @x1__Rollback guard  {
		%t13 = hw.constant 1 : i1
	  fsm.return %t13
	} 
}
fsm.state @x1__Rollback output  {

	 %t14 = hw.constant 1 : i1 
	
	 %t15 = hw.constant 0 : i8 
	
	 %t16 = hw.constant 1 : i8 
	
	 %t17 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t14,%default_commit_x,%t17,%t15,%default_rollback_x,%default_startStall_x,%t16,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @x1__Fill0 guard  {
		%t18 = hw.constant 1 : i1
	  fsm.return %t18
	} 
}
fsm.state @x1__Fill0 output  {

	 %t19 = hw.constant 1 : i1 
	
	 %t20 = hw.constant 0 : i8 
	
	 %t21 = hw.constant 1 : i8 
	
	 %t22 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t19,%default_commit_x,%t22,%t20,%default_rollback_x,%default_startStall_x,%t21,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @x1__Fill1 guard  {
		%t23 = hw.constant 1 : i1
	  fsm.return %t23
	} 
}
fsm.state @x1__Fill1 output  {

	 %t24 = hw.constant 1 : i1 
	
	 %t25 = hw.constant 0 : i8 
	
	 %t26 = hw.constant 1 : i8 
	
	 %t27 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t24,%default_commit_x,%t27,%t25,%default_rollback_x,%default_startStall_x,%t26,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @x1__Fill2 guard  {
		%t28 = hw.constant 1 : i1
	  fsm.return %t28
	} 
}
fsm.state @x1__Fill2 output  {

	 %t29 = hw.constant 1 : i1 
	
	 %t30 = hw.constant 0 : i8 
	
	 %t31 = hw.constant 1 : i8 
	
	 %t32 = hw.constant 1 : i1 
	
	 %t33 = hw.constant 0 : i8 
	
	 %t34 = hw.constant 1 : i8 
	
	 %t35 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t29,%t32,%t35,%t33,%default_rollback_x,%default_startStall_x,%t34,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t36 = hw.constant 1 : i1
	  fsm.return %t36
	} 
	fsm.transition @x1_x00__Rollback guard  {
		 %t37 = hw.constant 0 : i8 
		%t38 = comb.icmp eq %mispec_x0,%t37 : i8
	  fsm.return %t38
	} 
}
fsm.state @x00__Rollback output  {

	 %t39 = hw.constant 1 : i1 
	
	 %t40 = hw.constant 1 : i8 
	
	 %t41 = hw.constant 0 : i8 
	
	 %t42 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x0 = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t39,%t42,%default_commit_x0,%t41,%default_rollback_x,%default_startStall_x,%t40,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @x00__Fill0 guard  {
		%t43 = hw.constant 1 : i1
	  fsm.return %t43
	} 
}
fsm.state @x00__Fill0 output  {

	 %t44 = hw.constant 1 : i1 
	
	 %t45 = hw.constant 1 : i8 
	
	 %t46 = hw.constant 0 : i8 
	
	 %t47 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x0 = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t44,%t47,%default_commit_x0,%t46,%default_rollback_x,%default_startStall_x,%t45,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @x00__Fill1 guard  {
		%t48 = hw.constant 1 : i1
	  fsm.return %t48
	} 
}
fsm.state @x00__Fill1 output  {

	 %t49 = hw.constant 1 : i1 
	
	 %t50 = hw.constant 1 : i8 
	
	 %t51 = hw.constant 0 : i8 
	
	 %t52 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x0 = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t49,%t52,%default_commit_x0,%t51,%default_rollback_x,%default_startStall_x,%t50,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @x00__Fill2 guard  {
		%t53 = hw.constant 1 : i1
	  fsm.return %t53
	} 
}
fsm.state @x00__Fill2 output  {

	 %t54 = hw.constant 1 : i1 
	
	 %t55 = hw.constant 1 : i8 
	
	 %t56 = hw.constant 0 : i8 
	
	 %t57 = hw.constant 1 : i1 
	
	 %t58 = hw.constant 0 : i8 
	
	 %t59 = hw.constant 1 : i8 
	
	 %t60 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t54,%t60,%t57,%t58,%default_rollback_x,%default_startStall_x,%t59,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t61 = hw.constant 1 : i1
	  fsm.return %t61
	} 
	fsm.transition @x1__Stall0 guard  {
		 %t62 = hw.constant 1 : i8 
		%t63 = comb.icmp eq %mispec_x,%t62 : i8
	  fsm.return %t63
	} 
}
fsm.state @x1_x00__Rollback output  {

	 %t64 = hw.constant 1 : i1 
	
	 %t65 = hw.constant 1 : i8 
	
	 %t66 = hw.constant 0 : i8 
	
	 %t67 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x0 = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t64,%t67,%default_commit_x0,%t66,%default_rollback_x,%default_startStall_x,%t65,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @x1_x00__Fill0 guard  {
		%t68 = hw.constant 1 : i1
	  fsm.return %t68
	} 
}
fsm.state @x1_x00__Fill0 output  {

	 %t69 = hw.constant 1 : i1 
	
	 %t70 = hw.constant 1 : i8 
	
	 %t71 = hw.constant 0 : i8 
	
	 %t72 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x0 = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t69,%t72,%default_commit_x0,%t71,%default_rollback_x,%default_startStall_x,%t70,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @x1_x00__Fill1 guard  {
		%t73 = hw.constant 1 : i1
	  fsm.return %t73
	} 
}
fsm.state @x1_x00__Fill1 output  {

	 %t74 = hw.constant 1 : i1 
	
	 %t75 = hw.constant 1 : i8 
	
	 %t76 = hw.constant 0 : i8 
	
	 %t77 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x0 = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t74,%t77,%default_commit_x0,%t76,%default_rollback_x,%default_startStall_x,%t75,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @x1_x00__Fill2 guard  {
		%t78 = hw.constant 1 : i1
	  fsm.return %t78
	} 
}
fsm.state @x1_x00__Fill2 output  {

	 %t79 = hw.constant 1 : i1 
	
	 %t80 = hw.constant 1 : i8 
	
	 %t81 = hw.constant 0 : i8 
	
	 %t82 = hw.constant 1 : i1 
	
	 %t83 = hw.constant 1 : i1 
	
	 %t84 = hw.constant 0 : i8 
	
	 %t85 = hw.constant 1 : i8 
	
	 %t86 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t79,%t86,%t83,%t84,%default_rollback_x,%default_startStall_x,%t85,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t87 = hw.constant 1 : i1
	  fsm.return %t87
	} 
}
fsm.state @Init0 output  {

	 %t88 = hw.constant 1 : i1 
	
	 %t89 = hw.constant 0 : i8 
	
	 %t90 = hw.constant 1 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x = hw.constant 0 : i1
	
	%default_commit_x0 = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t88,%default_commit_x,%default_commit_x0,%t89,%default_rollback_x,%default_startStall_x,%t90,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Init1 guard  {
		%t91 = hw.constant 1 : i1
	  fsm.return %t91
	} 
}
fsm.state @Init1 output  {

	 %t92 = hw.constant 1 : i1 
	
	 %t93 = hw.constant 0 : i8 
	
	 %t94 = hw.constant 1 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x = hw.constant 0 : i1
	
	%default_commit_x0 = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t92,%default_commit_x,%default_commit_x0,%t93,%default_rollback_x,%default_startStall_x,%t94,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Init2 guard  {
		%t95 = hw.constant 1 : i1
	  fsm.return %t95
	} 
}
fsm.state @Init2 output  {

	 %t96 = hw.constant 1 : i1 
	
	 %t97 = hw.constant 0 : i8 
	
	 %t98 = hw.constant 1 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x = hw.constant 0 : i1
	
	%default_commit_x0 = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t96,%default_commit_x,%default_commit_x0,%t97,%default_rollback_x,%default_startStall_x,%t98,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t99 = hw.constant 1 : i1
	  fsm.return %t99
	} 
}
}