fsm.machine @SpecSCC_8_fsm(%mispec_x: i8) -> (i8,i8,i8,i1,i1,i8,i8,i1) 
	attributes {initialState = "Proceed"} {
fsm.state @Proceed output  {

	 %t0 = hw.constant 1 : i1 
	
	 %t1 = hw.constant 1 : i1 
	
	 %t2 = hw.constant 5 : i8 
	
	 %t3 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t0,%t3,%t2,%default_rollback_x,%default_startStall_x:i8,i8,i8,i1,i1,i8,i8,i1
} transitions  {
	fsm.transition @x0__Stall0 guard  {
		 %t4 = hw.constant 0 : i8 
		%t5 = comb.icmp eq %mispec_x,%t4 : i8
	  fsm.return %t5
	} 
	fsm.transition @x1__Stall0 guard  {
		 %t6 = hw.constant 1 : i8 
		%t7 = comb.icmp eq %mispec_x,%t6 : i8
	  fsm.return %t7
	} 
	fsm.transition @x2__Stall0 guard  {
		 %t8 = hw.constant 2 : i8 
		%t9 = comb.icmp eq %mispec_x,%t8 : i8
	  fsm.return %t9
	} 
	fsm.transition @x3__Rollback guard  {
		 %t10 = hw.constant 3 : i8 
		%t11 = comb.icmp eq %mispec_x,%t10 : i8
	  fsm.return %t11
	} 
	fsm.transition @x4__Rollback guard  {
		 %t12 = hw.constant 4 : i8 
		%t13 = comb.icmp eq %mispec_x,%t12 : i8
	  fsm.return %t13
	} 
}
fsm.state @x0__Stall0 output  {

	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rbwe = hw.constant 0 : i1
	
	%default_commit_x = hw.constant 0 : i1
	
	%default_selSlowPath_x = hw.constant 0 : i8
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%default_rbwe,%default_commit_x,%default_selSlowPath_x,%default_rollback_x,%default_startStall_x:i8,i8,i8,i1,i1,i8,i8,i1
} transitions  {
	fsm.transition @x0__Stall1 guard  {
		%t14 = hw.constant 1 : i1
	  fsm.return %t14
	} 
}
fsm.state @x0__Stall1 output  {

	 %t15 = hw.constant 5 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rbwe = hw.constant 0 : i1
	
	%default_commit_x = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%default_rbwe,%default_commit_x,%t15,%default_rollback_x,%default_startStall_x:i8,i8,i8,i1,i1,i8,i8,i1
} transitions  {
	fsm.transition @x0__Stall2 guard  {
		%t16 = hw.constant 1 : i1
	  fsm.return %t16
	} 
}
fsm.state @x0__Stall2 output  {

	 %t17 = hw.constant 5 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rbwe = hw.constant 0 : i1
	
	%default_commit_x = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%default_rbwe,%default_commit_x,%t17,%default_rollback_x,%default_startStall_x:i8,i8,i8,i1,i1,i8,i8,i1
} transitions  {
	fsm.transition @x0__Rollback guard  {
		%t18 = hw.constant 1 : i1
	  fsm.return %t18
	} 
}
fsm.state @x0__Rollback output  {

	 %t19 = hw.constant 1 : i1 
	
	 %t20 = hw.constant 5 : i8 
	
	 %t21 = hw.constant 1 : i1 
	
	 %t22 = hw.constant 5 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t19,%t21,%t22,%default_rollback_x,%default_startStall_x:i8,i8,i8,i1,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t23 = hw.constant 1 : i1
	  fsm.return %t23
	} 
}
fsm.state @x1__Stall0 output  {

	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rbwe = hw.constant 0 : i1
	
	%default_commit_x = hw.constant 0 : i1
	
	%default_selSlowPath_x = hw.constant 0 : i8
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%default_rbwe,%default_commit_x,%default_selSlowPath_x,%default_rollback_x,%default_startStall_x:i8,i8,i8,i1,i1,i8,i8,i1
} transitions  {
	fsm.transition @x1__Stall1 guard  {
		%t24 = hw.constant 1 : i1
	  fsm.return %t24
	} 
}
fsm.state @x1__Stall1 output  {

	 %t25 = hw.constant 5 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rbwe = hw.constant 0 : i1
	
	%default_commit_x = hw.constant 0 : i1
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%default_rbwe,%default_commit_x,%t25,%default_rollback_x,%default_startStall_x:i8,i8,i8,i1,i1,i8,i8,i1
} transitions  {
	fsm.transition @x1__Rollback guard  {
		%t26 = hw.constant 1 : i1
	  fsm.return %t26
	} 
}
fsm.state @x1__Rollback output  {

	 %t27 = hw.constant 1 : i1 
	
	 %t28 = hw.constant 5 : i8 
	
	 %t29 = hw.constant 1 : i1 
	
	 %t30 = hw.constant 5 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t27,%t29,%t30,%default_rollback_x,%default_startStall_x:i8,i8,i8,i1,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t31 = hw.constant 1 : i1
	  fsm.return %t31
	} 
}
fsm.state @x2__Stall0 output  {

	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rbwe = hw.constant 0 : i1
	
	%default_commit_x = hw.constant 0 : i1
	
	%default_selSlowPath_x = hw.constant 0 : i8
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%default_rbwe,%default_commit_x,%default_selSlowPath_x,%default_rollback_x,%default_startStall_x:i8,i8,i8,i1,i1,i8,i8,i1
} transitions  {
	fsm.transition @x2__Rollback guard  {
		%t32 = hw.constant 1 : i1
	  fsm.return %t32
	} 
}
fsm.state @x2__Rollback output  {

	 %t33 = hw.constant 1 : i1 
	
	 %t34 = hw.constant 5 : i8 
	
	 %t35 = hw.constant 1 : i1 
	
	 %t36 = hw.constant 5 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t33,%t35,%t36,%default_rollback_x,%default_startStall_x:i8,i8,i8,i1,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t37 = hw.constant 1 : i1
	  fsm.return %t37
	} 
}
fsm.state @x3__Rollback output  {

	 %t38 = hw.constant 1 : i1 
	
	 %t39 = hw.constant 5 : i8 
	
	 %t40 = hw.constant 1 : i1 
	
	 %t41 = hw.constant 5 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t38,%t40,%t41,%default_rollback_x,%default_startStall_x:i8,i8,i8,i1,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t42 = hw.constant 1 : i1
	  fsm.return %t42
	} 
}
fsm.state @x4__Rollback output  {

	 %t43 = hw.constant 1 : i1 
	
	 %t44 = hw.constant 5 : i8 
	
	 %t45 = hw.constant 1 : i1 
	
	 %t46 = hw.constant 5 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_x = hw.constant 0 : i8
	
	%default_startStall_x = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t43,%t45,%t46,%default_rollback_x,%default_startStall_x:i8,i8,i8,i1,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t47 = hw.constant 1 : i1
	  fsm.return %t47
	} 
}
}