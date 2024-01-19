ssp.instance @SCC_19 of "GammaMobilityProblem" {
				library {
					operator_type @op30 [latency<1>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op31 [latency<0>, incDelay<1.6849625: f32>, outDelay<1.6849625 : f32>]
					operator_type @op25 [latency<0>, incDelay<3.421928: f32>, outDelay<3.421928 : f32>]
					operator_type @op26 [latency<1>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op27 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op28 [latency<0>, incDelay<1.0: f32>, outDelay<1.0 : f32>]
					operator_type @op21 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op22 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op23 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op24 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op29 [latency<0>, incDelay<1.6849625: f32>, outDelay<1.6849625 : f32>]
					operator_type @op20 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op0 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op2 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op14 [latency<1>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op1 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op15 [latency<8>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op4 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op16 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op3 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op17 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op6 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op10 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op5 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op11 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op8 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op12 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op7 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op13 [latency<0>, incDelay<3.421928: f32>, outDelay<3.421928 : f32>]
					operator_type @op9 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op18 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op19 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
				}
				graph {
				%t10 = operation <@op0> @op_t10(@op_t1 [dist<2>])
				%t3 = operation <@op1> @op_t3(@op_t2 [dist<2>])
				%t5 = operation <@op2> @op_t5(%t3)
				%t14 = operation <@op12> @op_t14(%t5)
				%t6 = operation <@op13> @op_t6(@op_t4 [dist<8>],@op_t4 [dist<9>],@op_t4 [dist<4>],@op_t4 [dist<5>],@op_t4 [dist<6>],@op_t4 [dist<7>],@op_t4 [dist<1>],@op_t4 [dist<2>],@op_t4 [dist<3>],@op_t4 [dist<1>],@op_t4 [dist<2>],@op_t4 [dist<9>],@op_t4 [dist<3>],@op_t4 [dist<8>],@op_t4 [dist<3>],@op_t4 [dist<2>],@op_t4 [dist<1>],@op_t4 [dist<5>],@op_t4 [dist<6>],@op_t4 [dist<4>],@op_t4 [dist<7>],@op_t4 [dist<4>],@op_t4 [dist<7>],@op_t4 [dist<6>],@op_t4 [dist<5>],@op_t4 [dist<1>],@op_t4 [dist<8>],@op_t4 [dist<3>],@op_t4 [dist<2>],@op_t4 [dist<5>],@op_t4 [dist<4>],@op_t4 [dist<7>],@op_t4 [dist<6>],@op_t4 [dist<1>],@op_t4 [dist<2>],@op_t4 [dist<4>],@op_t4 [dist<2>],@op_t4 [dist<3>],@op_t4 [dist<1>],@op_t4 [dist<3>],@op_t4 [dist<2>],@op_t4 [dist<1>],@op_t4 [dist<1>],@op_t4 [dist<6>],@op_t4 [dist<5>],@op_t4 [dist<3>],@op_t4 [dist<4>],@op_t4 [dist<2>],@op_t4 [dist<5>],@op_t4 [dist<4>],@op_t4 [dist<3>],@op_t4 [dist<2>],@op_t4 [dist<1>],@op_t4 [dist<1>]) {"SpecHLS.gamma"}
				%t7 = operation <@op14> @op_t7(%t5,%t6)
				%t13 = operation <@op15> @op_t13(%t7)
				%t8 = operation <@op25> @op_t8(@op_t4 [dist<4>],@op_t4 [dist<5>],@op_t4 [dist<2>],@op_t4 [dist<3>],@op_t4 [dist<1>],@op_t4 [dist<4>],@op_t4 [dist<5>],@op_t4 [dist<6>],@op_t4 [dist<1>],@op_t4 [dist<2>],@op_t4 [dist<3>],@op_t4 [dist<1>],@op_t4 [dist<1>],@op_t4 [dist<3>],@op_t4 [dist<2>],@op_t4 [dist<2>],@op_t4 [dist<5>],@op_t4 [dist<4>],@op_t4 [dist<7>],@op_t4 [dist<6>],@op_t4 [dist<8>],@op_t4 [dist<9>],@op_t4 [dist<1>],@op_t4 [dist<2>],@op_t4 [dist<3>],@op_t4 [dist<4>],@op_t4 [dist<5>],@op_t4 [dist<6>],@op_t4 [dist<1>],@op_t4 [dist<7>],@op_t4 [dist<8>],@op_t4 [dist<9>],@op_t4 [dist<3>],@op_t4 [dist<1>],@op_t4 [dist<2>],@op_t4 [dist<2>],@op_t4 [dist<1>],@op_t4 [dist<3>],@op_t4 [dist<4>],@op_t4 [dist<5>],@op_t4 [dist<6>],@op_t4 [dist<7>],@op_t4 [dist<4>],@op_t4 [dist<8>],@op_t4 [dist<1>],@op_t4 [dist<2>],@op_t4 [dist<3>],@op_t4 [dist<4>],@op_t4 [dist<3>],@op_t4 [dist<5>],@op_t4 [dist<2>],@op_t4 [dist<6>],@op_t4 [dist<1>],@op_t4 [dist<7>]) {"SpecHLS.gamma"}
				%t9 = operation <@op26> @op_t9(%t8)
				%t11 = operation <@op27> @op_t11(%t9,%t3)
				%t12 = operation <@op28> @op_t12(%t9,%t7,%t10)
				%t2 = operation <@op29> @op_t2(%t3,%t11) {"SpecHLS.gamma"}
				%t4 = operation <@op30> @op_t4(%t12,%t13)
				%t1 = operation <@op31> @op_t1(%t4,%t10) {"SpecHLS.gamma"}
				}
			}