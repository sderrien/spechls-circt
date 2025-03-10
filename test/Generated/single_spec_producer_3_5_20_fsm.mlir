fsm.machine @SpecSCC_215_fsm(%mispec_l_x: i8,%mispec_y: i8,%mispec_y0: i8) -> (i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1) 
	attributes {initialState = "Init0"} {
fsm.state @Proceed output  {

	 %t0 = hw.constant 1 : i1 
	
	 %t1 = hw.constant 1 : i1 
	
	 %t2 = hw.constant 1 : i1 
	
	 %t3 = hw.constant 1 : i1 
	
	 %t4 = hw.constant 1 : i8 
	
	 %t5 = hw.constant 0 : i8 
	
	 %t6 = hw.constant 1 : i8 
	
	 %t7 = hw.constant 1 : i1 
	
	 %t8 = hw.constant 1 : i1 
	
	 %t9 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t0,%t7,%t8,%t9,%t4,%default_rollback_l_x,%default_startStall_l_x,%t5,%default_rollback_y,%default_startStall_y,%t6,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x0__Rollback guard  {
		 %t10 = hw.constant 0 : i8 
		%t11 = comb.icmp eq %mispec_l_x,%t10 : i8
	  fsm.return %t11
	} 
	fsm.transition @y1__Rollback guard  {
		 %t12 = hw.constant 1 : i8 
		%t13 = comb.icmp eq %mispec_y,%t12 : i8
	  fsm.return %t13
	} 
	fsm.transition @y00__Rollback guard  {
		 %t14 = hw.constant 0 : i8 
		%t15 = comb.icmp eq %mispec_y0,%t14 : i8
	  fsm.return %t15
	} 
}
fsm.state @l_x0__Rollback output  {

	 %t16 = hw.constant 1 : i1 
	
	 %t17 = hw.constant 1 : i8 
	
	 %t18 = hw.constant 0 : i8 
	
	 %t19 = hw.constant 1 : i8 
	
	 %t20 = hw.constant 1 : i1 
	
	 %t21 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t16,%default_commit_l_x,%t20,%t21,%t17,%default_rollback_l_x,%default_startStall_l_x,%t18,%default_rollback_y,%default_startStall_y,%t19,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x0__Fill0 guard  {
		%t22 = hw.constant 1 : i1
	  fsm.return %t22
	} 
}
fsm.state @l_x0__Fill0 output  {

	 %t23 = hw.constant 1 : i1 
	
	 %t24 = hw.constant 1 : i8 
	
	 %t25 = hw.constant 0 : i8 
	
	 %t26 = hw.constant 1 : i8 
	
	 %t27 = hw.constant 1 : i1 
	
	 %t28 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t23,%default_commit_l_x,%t27,%t28,%t24,%default_rollback_l_x,%default_startStall_l_x,%t25,%default_rollback_y,%default_startStall_y,%t26,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x0__Fill1 guard  {
		%t29 = hw.constant 1 : i1
	  fsm.return %t29
	} 
}
fsm.state @l_x0__Fill1 output  {

	 %t30 = hw.constant 1 : i1 
	
	 %t31 = hw.constant 1 : i8 
	
	 %t32 = hw.constant 0 : i8 
	
	 %t33 = hw.constant 1 : i8 
	
	 %t34 = hw.constant 1 : i1 
	
	 %t35 = hw.constant 1 : i8 
	
	 %t36 = hw.constant 0 : i8 
	
	 %t37 = hw.constant 1 : i8 
	
	 %t38 = hw.constant 1 : i1 
	
	 %t39 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t30,%t34,%t38,%t39,%t35,%default_rollback_l_x,%default_startStall_l_x,%t36,%default_rollback_y,%default_startStall_y,%t37,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t40 = hw.constant 1 : i1
	  fsm.return %t40
	} 
	fsm.transition @l_x0_y1__Rollback guard  {
		 %t41 = hw.constant 1 : i8 
		%t42 = comb.icmp eq %mispec_y,%t41 : i8
	  fsm.return %t42
	} 
	fsm.transition @l_x0_y00__Rollback guard  {
		 %t43 = hw.constant 0 : i8 
		%t44 = comb.icmp eq %mispec_y0,%t43 : i8
	  fsm.return %t44
	} 
}
fsm.state @y1__Rollback output  {

	 %t45 = hw.constant 1 : i1 
	
	 %t46 = hw.constant 0 : i8 
	
	 %t47 = hw.constant 1 : i8 
	
	 %t48 = hw.constant 1 : i8 
	
	 %t49 = hw.constant 1 : i1 
	
	 %t50 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_y = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t45,%t49,%default_commit_y,%t50,%t47,%default_rollback_l_x,%default_startStall_l_x,%t46,%default_rollback_y,%default_startStall_y,%t48,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @y1__Fill0 guard  {
		%t51 = hw.constant 1 : i1
	  fsm.return %t51
	} 
}
fsm.state @y1__Fill0 output  {

	 %t52 = hw.constant 1 : i1 
	
	 %t53 = hw.constant 0 : i8 
	
	 %t54 = hw.constant 1 : i8 
	
	 %t55 = hw.constant 1 : i8 
	
	 %t56 = hw.constant 1 : i1 
	
	 %t57 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_y = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t52,%t56,%default_commit_y,%t57,%t54,%default_rollback_l_x,%default_startStall_l_x,%t53,%default_rollback_y,%default_startStall_y,%t55,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @y1__Fill1 guard  {
		%t58 = hw.constant 1 : i1
	  fsm.return %t58
	} 
}
fsm.state @y1__Fill1 output  {

	 %t59 = hw.constant 1 : i1 
	
	 %t60 = hw.constant 0 : i8 
	
	 %t61 = hw.constant 1 : i8 
	
	 %t62 = hw.constant 1 : i8 
	
	 %t63 = hw.constant 1 : i1 
	
	 %t64 = hw.constant 1 : i8 
	
	 %t65 = hw.constant 0 : i8 
	
	 %t66 = hw.constant 1 : i8 
	
	 %t67 = hw.constant 1 : i1 
	
	 %t68 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t59,%t67,%t63,%t68,%t64,%default_rollback_l_x,%default_startStall_l_x,%t65,%default_rollback_y,%default_startStall_y,%t66,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t69 = hw.constant 1 : i1
	  fsm.return %t69
	} 
	fsm.transition @l_x0__Rollback guard  {
		 %t70 = hw.constant 0 : i8 
		%t71 = comb.icmp eq %mispec_l_x,%t70 : i8
	  fsm.return %t71
	} 
	fsm.transition @y1_y00__Rollback guard  {
		 %t72 = hw.constant 0 : i8 
		%t73 = comb.icmp eq %mispec_y0,%t72 : i8
	  fsm.return %t73
	} 
}
fsm.state @y00__Rollback output  {

	 %t74 = hw.constant 1 : i1 
	
	 %t75 = hw.constant 1 : i8 
	
	 %t76 = hw.constant 1 : i8 
	
	 %t77 = hw.constant 0 : i8 
	
	 %t78 = hw.constant 1 : i1 
	
	 %t79 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_y0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t74,%t78,%t79,%default_commit_y0,%t76,%default_rollback_l_x,%default_startStall_l_x,%t77,%default_rollback_y,%default_startStall_y,%t75,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @y00__Fill0 guard  {
		%t80 = hw.constant 1 : i1
	  fsm.return %t80
	} 
}
fsm.state @y00__Fill0 output  {

	 %t81 = hw.constant 1 : i1 
	
	 %t82 = hw.constant 1 : i8 
	
	 %t83 = hw.constant 1 : i8 
	
	 %t84 = hw.constant 0 : i8 
	
	 %t85 = hw.constant 1 : i1 
	
	 %t86 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_y0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t81,%t85,%t86,%default_commit_y0,%t83,%default_rollback_l_x,%default_startStall_l_x,%t84,%default_rollback_y,%default_startStall_y,%t82,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @y00__Fill1 guard  {
		%t87 = hw.constant 1 : i1
	  fsm.return %t87
	} 
}
fsm.state @y00__Fill1 output  {

	 %t88 = hw.constant 1 : i1 
	
	 %t89 = hw.constant 1 : i8 
	
	 %t90 = hw.constant 1 : i8 
	
	 %t91 = hw.constant 0 : i8 
	
	 %t92 = hw.constant 1 : i1 
	
	 %t93 = hw.constant 1 : i8 
	
	 %t94 = hw.constant 0 : i8 
	
	 %t95 = hw.constant 1 : i8 
	
	 %t96 = hw.constant 1 : i1 
	
	 %t97 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t88,%t96,%t97,%t92,%t93,%default_rollback_l_x,%default_startStall_l_x,%t94,%default_rollback_y,%default_startStall_y,%t95,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t98 = hw.constant 1 : i1
	  fsm.return %t98
	} 
	fsm.transition @l_x0__Rollback guard  {
		 %t99 = hw.constant 0 : i8 
		%t100 = comb.icmp eq %mispec_l_x,%t99 : i8
	  fsm.return %t100
	} 
	fsm.transition @y1__Rollback guard  {
		 %t101 = hw.constant 1 : i8 
		%t102 = comb.icmp eq %mispec_y,%t101 : i8
	  fsm.return %t102
	} 
}
fsm.state @l_x0_y1__Rollback output  {

	 %t103 = hw.constant 1 : i1 
	
	 %t104 = hw.constant 0 : i8 
	
	 %t105 = hw.constant 1 : i8 
	
	 %t106 = hw.constant 1 : i8 
	
	 %t107 = hw.constant 1 : i1 
	
	 %t108 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_y = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t103,%t107,%default_commit_y,%t108,%t105,%default_rollback_l_x,%default_startStall_l_x,%t104,%default_rollback_y,%default_startStall_y,%t106,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x0_y1__Fill0 guard  {
		%t109 = hw.constant 1 : i1
	  fsm.return %t109
	} 
}
fsm.state @l_x0_y1__Fill0 output  {

	 %t110 = hw.constant 1 : i1 
	
	 %t111 = hw.constant 0 : i8 
	
	 %t112 = hw.constant 1 : i8 
	
	 %t113 = hw.constant 1 : i8 
	
	 %t114 = hw.constant 1 : i1 
	
	 %t115 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_y = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t110,%t114,%default_commit_y,%t115,%t112,%default_rollback_l_x,%default_startStall_l_x,%t111,%default_rollback_y,%default_startStall_y,%t113,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x0_y1__Fill1 guard  {
		%t116 = hw.constant 1 : i1
	  fsm.return %t116
	} 
}
fsm.state @l_x0_y1__Fill1 output  {

	 %t117 = hw.constant 1 : i1 
	
	 %t118 = hw.constant 0 : i8 
	
	 %t119 = hw.constant 1 : i8 
	
	 %t120 = hw.constant 1 : i8 
	
	 %t121 = hw.constant 1 : i1 
	
	 %t122 = hw.constant 1 : i1 
	
	 %t123 = hw.constant 1 : i8 
	
	 %t124 = hw.constant 0 : i8 
	
	 %t125 = hw.constant 1 : i8 
	
	 %t126 = hw.constant 1 : i1 
	
	 %t127 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t117,%t126,%t122,%t127,%t123,%default_rollback_l_x,%default_startStall_l_x,%t124,%default_rollback_y,%default_startStall_y,%t125,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t128 = hw.constant 1 : i1
	  fsm.return %t128
	} 
	fsm.transition @l_x0_y1_y00__Rollback guard  {
		 %t129 = hw.constant 0 : i8 
		%t130 = comb.icmp eq %mispec_y0,%t129 : i8
	  fsm.return %t130
	} 
}
fsm.state @l_x0_y00__Rollback output  {

	 %t131 = hw.constant 1 : i1 
	
	 %t132 = hw.constant 1 : i8 
	
	 %t133 = hw.constant 1 : i8 
	
	 %t134 = hw.constant 0 : i8 
	
	 %t135 = hw.constant 1 : i1 
	
	 %t136 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_y0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t131,%t135,%t136,%default_commit_y0,%t133,%default_rollback_l_x,%default_startStall_l_x,%t134,%default_rollback_y,%default_startStall_y,%t132,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x0_y00__Fill0 guard  {
		%t137 = hw.constant 1 : i1
	  fsm.return %t137
	} 
}
fsm.state @l_x0_y00__Fill0 output  {

	 %t138 = hw.constant 1 : i1 
	
	 %t139 = hw.constant 1 : i8 
	
	 %t140 = hw.constant 1 : i8 
	
	 %t141 = hw.constant 0 : i8 
	
	 %t142 = hw.constant 1 : i1 
	
	 %t143 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_y0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t138,%t142,%t143,%default_commit_y0,%t140,%default_rollback_l_x,%default_startStall_l_x,%t141,%default_rollback_y,%default_startStall_y,%t139,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x0_y00__Fill1 guard  {
		%t144 = hw.constant 1 : i1
	  fsm.return %t144
	} 
}
fsm.state @l_x0_y00__Fill1 output  {

	 %t145 = hw.constant 1 : i1 
	
	 %t146 = hw.constant 1 : i8 
	
	 %t147 = hw.constant 1 : i8 
	
	 %t148 = hw.constant 0 : i8 
	
	 %t149 = hw.constant 1 : i1 
	
	 %t150 = hw.constant 1 : i1 
	
	 %t151 = hw.constant 1 : i8 
	
	 %t152 = hw.constant 0 : i8 
	
	 %t153 = hw.constant 1 : i8 
	
	 %t154 = hw.constant 1 : i1 
	
	 %t155 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t145,%t154,%t155,%t150,%t151,%default_rollback_l_x,%default_startStall_l_x,%t152,%default_rollback_y,%default_startStall_y,%t153,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t156 = hw.constant 1 : i1
	  fsm.return %t156
	} 
	fsm.transition @l_x0_y1__Rollback guard  {
		 %t157 = hw.constant 1 : i8 
		%t158 = comb.icmp eq %mispec_y,%t157 : i8
	  fsm.return %t158
	} 
}
fsm.state @y1_y00__Rollback output  {

	 %t159 = hw.constant 1 : i1 
	
	 %t160 = hw.constant 1 : i8 
	
	 %t161 = hw.constant 1 : i8 
	
	 %t162 = hw.constant 0 : i8 
	
	 %t163 = hw.constant 1 : i1 
	
	 %t164 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_y0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t159,%t163,%t164,%default_commit_y0,%t161,%default_rollback_l_x,%default_startStall_l_x,%t162,%default_rollback_y,%default_startStall_y,%t160,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @y1_y00__Fill0 guard  {
		%t165 = hw.constant 1 : i1
	  fsm.return %t165
	} 
}
fsm.state @y1_y00__Fill0 output  {

	 %t166 = hw.constant 1 : i1 
	
	 %t167 = hw.constant 1 : i8 
	
	 %t168 = hw.constant 1 : i8 
	
	 %t169 = hw.constant 0 : i8 
	
	 %t170 = hw.constant 1 : i1 
	
	 %t171 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_y0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t166,%t170,%t171,%default_commit_y0,%t168,%default_rollback_l_x,%default_startStall_l_x,%t169,%default_rollback_y,%default_startStall_y,%t167,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @y1_y00__Fill1 guard  {
		%t172 = hw.constant 1 : i1
	  fsm.return %t172
	} 
}
fsm.state @y1_y00__Fill1 output  {

	 %t173 = hw.constant 1 : i1 
	
	 %t174 = hw.constant 1 : i8 
	
	 %t175 = hw.constant 1 : i8 
	
	 %t176 = hw.constant 0 : i8 
	
	 %t177 = hw.constant 1 : i1 
	
	 %t178 = hw.constant 1 : i1 
	
	 %t179 = hw.constant 1 : i8 
	
	 %t180 = hw.constant 0 : i8 
	
	 %t181 = hw.constant 1 : i8 
	
	 %t182 = hw.constant 1 : i1 
	
	 %t183 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t173,%t182,%t183,%t178,%t179,%default_rollback_l_x,%default_startStall_l_x,%t180,%default_rollback_y,%default_startStall_y,%t181,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t184 = hw.constant 1 : i1
	  fsm.return %t184
	} 
	fsm.transition @l_x0__Rollback guard  {
		 %t185 = hw.constant 0 : i8 
		%t186 = comb.icmp eq %mispec_l_x,%t185 : i8
	  fsm.return %t186
	} 
}
fsm.state @l_x0_y1_y00__Rollback output  {

	 %t187 = hw.constant 1 : i1 
	
	 %t188 = hw.constant 1 : i8 
	
	 %t189 = hw.constant 1 : i8 
	
	 %t190 = hw.constant 0 : i8 
	
	 %t191 = hw.constant 1 : i1 
	
	 %t192 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_y0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t187,%t191,%t192,%default_commit_y0,%t189,%default_rollback_l_x,%default_startStall_l_x,%t190,%default_rollback_y,%default_startStall_y,%t188,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x0_y1_y00__Fill0 guard  {
		%t193 = hw.constant 1 : i1
	  fsm.return %t193
	} 
}
fsm.state @l_x0_y1_y00__Fill0 output  {

	 %t194 = hw.constant 1 : i1 
	
	 %t195 = hw.constant 1 : i8 
	
	 %t196 = hw.constant 1 : i8 
	
	 %t197 = hw.constant 0 : i8 
	
	 %t198 = hw.constant 1 : i1 
	
	 %t199 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_y0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t194,%t198,%t199,%default_commit_y0,%t196,%default_rollback_l_x,%default_startStall_l_x,%t197,%default_rollback_y,%default_startStall_y,%t195,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @l_x0_y1_y00__Fill1 guard  {
		%t200 = hw.constant 1 : i1
	  fsm.return %t200
	} 
}
fsm.state @l_x0_y1_y00__Fill1 output  {

	 %t201 = hw.constant 1 : i1 
	
	 %t202 = hw.constant 1 : i8 
	
	 %t203 = hw.constant 1 : i8 
	
	 %t204 = hw.constant 0 : i8 
	
	 %t205 = hw.constant 1 : i1 
	
	 %t206 = hw.constant 1 : i1 
	
	 %t207 = hw.constant 1 : i1 
	
	 %t208 = hw.constant 1 : i8 
	
	 %t209 = hw.constant 0 : i8 
	
	 %t210 = hw.constant 1 : i8 
	
	 %t211 = hw.constant 1 : i1 
	
	 %t212 = hw.constant 1 : i1 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t201,%t211,%t212,%t207,%t208,%default_rollback_l_x,%default_startStall_l_x,%t209,%default_rollback_y,%default_startStall_y,%t210,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t213 = hw.constant 1 : i1
	  fsm.return %t213
	} 
}
fsm.state @Init0 output  {

	 %t214 = hw.constant 1 : i1 
	
	 %t215 = hw.constant 1 : i8 
	
	 %t216 = hw.constant 0 : i8 
	
	 %t217 = hw.constant 1 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x = hw.constant 0 : i1
	
	%default_commit_y = hw.constant 0 : i1
	
	%default_commit_y0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t214,%default_commit_l_x,%default_commit_y,%default_commit_y0,%t215,%default_rollback_l_x,%default_startStall_l_x,%t216,%default_rollback_y,%default_startStall_y,%t217,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Init1 guard  {
		%t218 = hw.constant 1 : i1
	  fsm.return %t218
	} 
}
fsm.state @Init1 output  {

	 %t219 = hw.constant 1 : i1 
	
	 %t220 = hw.constant 1 : i8 
	
	 %t221 = hw.constant 0 : i8 
	
	 %t222 = hw.constant 1 : i8 
	
	%default_array_rollback = hw.constant 0 : i8
	
	%default_mu_rollback = hw.constant 0 : i8
	
	%default_rewind = hw.constant 0 : i8
	
	%default_commit_l_x = hw.constant 0 : i1
	
	%default_commit_y = hw.constant 0 : i1
	
	%default_commit_y0 = hw.constant 0 : i1
	
	%default_rollback_l_x = hw.constant 0 : i8
	
	%default_startStall_l_x = hw.constant 0 : i1
	
	%default_rollback_y = hw.constant 0 : i8
	
	%default_startStall_y = hw.constant 0 : i1
	
	%default_rollback_y0 = hw.constant 0 : i8
	
	%default_startStall_y0 = hw.constant 0 : i1
   	fsm.output %default_array_rollback,%default_mu_rollback,%default_rewind,%t219,%default_commit_l_x,%default_commit_y,%default_commit_y0,%t220,%default_rollback_l_x,%default_startStall_l_x,%t221,%default_rollback_y,%default_startStall_y,%t222,%default_rollback_y0,%default_startStall_y0:i8,i8,i8,i1,i1,i1,i1,i8,i8,i1,i8,i8,i1,i8,i8,i1
} transitions  {
	fsm.transition @Proceed guard  {
		%t223 = hw.constant 1 : i1
	  fsm.return %t223
	} 
}
}