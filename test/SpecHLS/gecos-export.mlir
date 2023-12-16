hw.module @SCC_0() {
	// node 0 at 0
	%t1 = SpecHLS.init "exit11" : i1 
	// node 0 at 1
	// node 0 at 2
	%t2 = SpecHLS.mu %t1,%t3 : i1 
	// node 0 at 3
	%t4 = SpecHLS.init "x" : memref<16xi32> 
	// node 0 at 4
	// node 0 at 5
	%t5 = SpecHLS.mu %t4,%t6 : memref<16xi32> 
	// node 0 at 6
	%t7 = SpecHLS.init "guard" : i1 
	// node 0 at 7
	// node 0 at 8
	%t8 = SpecHLS.mu %t7,%t9 : i1 
	// node 0 at 9
	%t10 = SpecHLS.init "guard" : i1 
	// node 0 at 10
	// node 0 at 11
	%t11 = SpecHLS.mu %t10,%t12 : i1 
	// node 0 at 12
	%t13 = SpecHLS.init "i" : i32 
	// node 0 at 13
	// node 0 at 14
	%t14 = SpecHLS.mu %t13,%t15 : i32 
	// node 0 at 15
	%t17 = comb.add %t16,%t18 : i32 
	// node 0 at 16
	%t18 = hw.constant 1 : i32 
	// node 0 at 17
	%t20 = comb.and %t19,%t11 : i1 
	// node 0 at 18
	%t19 = comb.and %t21,%t22 : i1 
	// node 0 at 19
	%t21 = comb.and %t23,%t8 : i1 
	// node 0 at 20
	%t23 = hw.constant 1 : i1 
	// node 0 at 21
	%t22 = hw.constant 1 : i1 
	// node 0 at 22
	%t24_idx = arith.index_cast %t14 : i32 to index
	%t24 = SpecHLS.alpha %t20 -> %t25 [%t24_idx], %t17: memref<16xi32>  
	// node 0 at 23
	%t16_idx = arith.index_cast %t14 : i32 to index
	%t16 = SpecHLS.read %t5:memref<16xi32>  [%t16_idx]   
	// node 0 at 24
	%t26 = comb.add %t14,%t27 : i32 
	// node 0 at 25
	%t27 = hw.constant 1 : i32 
	// node 0 at 26
	%t28 = comb.icmp slt %t26,%t29 : i32 
	// node 0 at 27
	%t29 = hw.constant 128 : i32 
	// node 0 at 28
	%t30 = comb.and %t8,%t11 : i1 
	// node 0 at 29
	// gamma exit11 :  1092822
	%t31 = SpecHLS.gamma %t30 ? %t2,%t28 :i1
	// node 0 at 30
	%t32 = comb.and %t8,%t11 : i1 
	// node 0 at 31
	// gamma x :  795156371
	%t33 = SpecHLS.gamma %t32 ? %t5,%t24 :memref<16xi32>
	// node 0 at 32
	%t34 = comb.and %t8,%t11 : i1 
	// node 0 at 33
	// gamma i :  77811359
	%t35 = SpecHLS.gamma %t34 ? %t14,%t26 :i32
	// node 0 at 34
	%t3 = SpecHLS.def "exit11_4" %t36: i1 
	// node 0 at 35
	%t38 = comb.or %t37,%t39 : i1 
	// node 0 at 36
	%t37 = comb.and %t8,%t40 : i1 
	// node 0 at 37
	%t40_0 = hw.constant 0 : i1
	%t40 = comb.icmp eq %t40_0,%t11: i1
	// node 0 at 38
	%t39 = comb.and %t8,%t11 : i1 
	// node 0 at 39
	// gamma exit11 :  537524656
	%t36 = SpecHLS.gamma %t38 ? %t2,%t31 :i1
	// node 0 at 40
	%t6 = SpecHLS.def "x_4" %t41: memref<16xi32> 
	// node 0 at 41
	%t43 = comb.or %t42,%t44 : i1 
	// node 0 at 42
	%t42 = comb.and %t8,%t45 : i1 
	// node 0 at 43
	%t45_0 = hw.constant 0 : i1
	%t45 = comb.icmp eq %t45_0,%t11: i1
	// node 0 at 44
	%t44 = comb.and %t8,%t11 : i1 
	// node 0 at 45
	// gamma x :  1790430792
	%t41 = SpecHLS.gamma %t43 ? %t5,%t33 :memref<16xi32>
	// node 0 at 46
	%t12 = SpecHLS.def "guard_4" %t46: i1 
	// node 0 at 47
	%t48 = comb.or %t47,%t49 : i1 
	// node 0 at 48
	%t47 = comb.and %t8,%t50 : i1 
	// node 0 at 49
	%t50_0 = hw.constant 0 : i1
	%t50 = comb.icmp eq %t50_0,%t11: i1
	// node 0 at 50
	%t49 = comb.and %t8,%t11 : i1 
	// node 0 at 51
	// gamma guard :  80903581
	%t46 = SpecHLS.gamma %t48 ? %t11,%t31 :i1
	// node 0 at 52
	%t15 = SpecHLS.def "i_5" %t51: i32 
	// node 0 at 53
	%t53 = comb.or %t52,%t54 : i1 
	// node 0 at 54
	%t52 = comb.and %t8,%t55 : i1 
	// node 0 at 55
	%t55_0 = hw.constant 0 : i1
	%t55 = comb.icmp eq %t55_0,%t11: i1
	// node 0 at 56
	%t54 = comb.and %t8,%t11 : i1 
	// node 0 at 57
	// gamma i :  1011276990
	%t51 = SpecHLS.gamma %t53 ? %t14,%t35 :i32
	// node 0 at 58
	%t9 = SpecHLS.def "guard_3" %t12: i1 
	// node 0 at 59
	// node 0 at 60
	// node 0 at 61
	// node 0 at 62
	// node 0 at 63
	// node 0 at 64
	// node 0 at 65
	// node 0 at 66
	%t56_0 = hw.constant 0 : i1
	%t56 = comb.icmp eq %t56_0,%t9: i1
	// node 0 at 67
	%t57 = hw.constant 1 : i1 
	// node 0 at 68
	%t58 = SpecHLS.delay %t57 -> %t56 by 4:i1 
	// node 0 at 69
	SpecHLS.exit %t58     
	// node 0 at 70
	%t25 = SpecHLS.sync %t5 : memref<16xi32>, %t16 : i32   
}
