fsm.machine @SpecSCC_13_fsm(%mispec_x: i8,%mispec_x0: i8) -> (i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1) 
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
	fsm.transition @x1__Rollback guard  {
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
fsm.state @x1__Rollback output  {

	 %t11 = hw.constant 1 : i1 
	
	 %t12 = hw.constant 0 : i8 
	
	 %t13 = hw.constant 1 : i8 
	
	 %t14 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t11,%default_commit_x,%t14,%t12,%default_rollback_x,%default_startStall_x,%t13,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @x1__Fill0 guard  {
		%t15 = hw.constant 1 : i1
	  fsm.return %t15
	} 
}
fsm.state @x1__Fill0 output  {

	 %t16 = hw.constant 1 : i1 
	
	 %t17 = hw.constant 0 : i8 
	
	 %t18 = hw.constant 1 : i8 
	
	 %t19 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t16,%default_commit_x,%t19,%t17,%default_rollback_x,%default_startStall_x,%t18,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @x1__Fill1 guard  {
		%t20 = hw.constant 1 : i1
	  fsm.return %t20
	} 
}
fsm.state @x1__Fill1 output  {

	 %t21 = hw.constant 1 : i1 
	
	 %t22 = hw.constant 0 : i8 
	
	 %t23 = hw.constant 1 : i8 
	
	 %t24 = hw.constant 1 : i1 
	
	 %t25 = hw.constant 0 : i8 
	
	 %t26 = hw.constant 1 : i8 
	
	 %t27 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t21,%t24,%t27,%t25,%default_rollback_x,%default_startStall_x,%t26,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t28 = hw.constant 1 : i1
	  fsm.return %t28
	} 
	fsm.transition @x1_x00__Rollback guard  {
		 %t29 = hw.constant 0 : i8 
		%t30 = comb.icmp eq %mispec_x0,%t29 : i8
	  fsm.return %t30
	} 
}
fsm.state @x00__Rollback output  {

	 %t31 = hw.constant 1 : i1 
	
	 %t32 = hw.constant 1 : i8 
	
	 %t33 = hw.constant 0 : i8 
	
	 %t34 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x0 = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t31,%t34,%default_commit_x0,%t33,%default_rollback_x,%default_startStall_x,%t32,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @x00__Fill0 guard  {
		%t35 = hw.constant 1 : i1
	  fsm.return %t35
	} 
}
fsm.state @x00__Fill0 output  {

	 %t36 = hw.constant 1 : i1 
	
	 %t37 = hw.constant 1 : i8 
	
	 %t38 = hw.constant 0 : i8 
	
	 %t39 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x0 = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t36,%t39,%default_commit_x0,%t38,%default_rollback_x,%default_startStall_x,%t37,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @x00__Fill1 guard  {
		%t40 = hw.constant 1 : i1
	  fsm.return %t40
	} 
}
fsm.state @x00__Fill1 output  {

	 %t41 = hw.constant 1 : i1 
	
	 %t42 = hw.constant 1 : i8 
	
	 %t43 = hw.constant 0 : i8 
	
	 %t44 = hw.constant 1 : i1 
	
	 %t45 = hw.constant 0 : i8 
	
	 %t46 = hw.constant 1 : i8 
	
	 %t47 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t41,%t47,%t44,%t45,%default_rollback_x,%default_startStall_x,%t46,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t48 = hw.constant 1 : i1
	  fsm.return %t48
	} 
	fsm.transition @x1__Rollback guard  {
		 %t49 = hw.constant 1 : i8 
		%t50 = comb.icmp eq %mispec_x,%t49 : i8
	  fsm.return %t50
	} 
}
fsm.state @x1_x00__Rollback output  {

	 %t51 = hw.constant 1 : i1 
	
	 %t52 = hw.constant 1 : i8 
	
	 %t53 = hw.constant 0 : i8 
	
	 %t54 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x0 = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t51,%t54,%default_commit_x0,%t53,%default_rollback_x,%default_startStall_x,%t52,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @x1_x00__Fill0 guard  {
		%t55 = hw.constant 1 : i1
	  fsm.return %t55
	} 
}
fsm.state @x1_x00__Fill0 output  {

	 %t56 = hw.constant 1 : i1 
	
	 %t57 = hw.constant 1 : i8 
	
	 %t58 = hw.constant 0 : i8 
	
	 %t59 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x0 = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t56,%t59,%default_commit_x0,%t58,%default_rollback_x,%default_startStall_x,%t57,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @x1_x00__Fill1 guard  {
		%t60 = hw.constant 1 : i1
	  fsm.return %t60
	} 
}
fsm.state @x1_x00__Fill1 output  {

	 %t61 = hw.constant 1 : i1 
	
	 %t62 = hw.constant 1 : i8 
	
	 %t63 = hw.constant 0 : i8 
	
	 %t64 = hw.constant 1 : i1 
	
	 %t65 = hw.constant 1 : i1 
	
	 %t66 = hw.constant 0 : i8 
	
	 %t67 = hw.constant 1 : i8 
	
	 %t68 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t61,%t68,%t65,%t66,%default_rollback_x,%default_startStall_x,%t67,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t69 = hw.constant 1 : i1
	  fsm.return %t69
	} 
}
fsm.state @Init0 output  {

	 %t70 = hw.constant 1 : i1 
	
	 %t71 = hw.constant 0 : i8 
	
	 %t72 = hw.constant 1 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x = hw.constant 0 : i1
	
	%default_commit_x0 = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t70,%default_commit_x,%default_commit_x0,%t71,%default_rollback_x,%default_startStall_x,%t72,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Init1 guard  {
		%t73 = hw.constant 1 : i1
	  fsm.return %t73
	} 
}
fsm.state @Init1 output  {

	 %t74 = hw.constant 1 : i1 
	
	 %t75 = hw.constant 0 : i8 
	
	 %t76 = hw.constant 1 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_x = hw.constant 0 : i1
	
	%default_commit_x0 = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
	
	%default_rollback_x0 = hw.constant 0 : i8
	
	%default_startStall_x0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t74,%default_commit_x,%default_commit_x0,%t75,%default_rollback_x,%default_startStall_x,%t76,%default_rollback_x0,%default_startStall_x0:i8,i8,i8,i1,i1,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t77 = hw.constant 1 : i1
	  fsm.return %t77
	} 
}
}