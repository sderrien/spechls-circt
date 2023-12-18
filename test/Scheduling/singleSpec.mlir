ssp.instance @SCC_10 of "GammaMobilityProblem" {
				library {
					operator_type @op2 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op1 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op4 [latency<0>, incDelay<2.74: f32>, outDelay<2.74 : f32>]
					operator_type @op3 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op6 [latency<0>, incDelay<2.43: f32>, outDelay<2.43 : f32>]
					operator_type @op10 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op5 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op11 [latency<0>, incDelay<2.1: f32>, outDelay<2.1 : f32>]
					operator_type @op8 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op7 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
					operator_type @op9 [latency<0>, incDelay<1.6849625: f32>, outDelay<1.6849625 : f32>]
					operator_type @op0 [latency<0>, incDelay<0.0: f32>, outDelay<0.0 : f32>]
				}
				graph {
				%t5 = operation <@op0> @op_t5(@op_t1 [dist<1>])
				%t3 = operation <@op1> @op_t3(@op_t2 [dist<1>])
				%t7 = operation <@op2> @op_t7(@op_t1 [dist<1>])
				%t4 = operation <@op4> @op_t4(%t3)
				%t6 = operation <@op6> @op_t6(%t4)
				%t2 = operation <@op9> @op_t2(%t3, %t4) {"SpecHLS.gamma"}
				%t8 = operation <@op10> @op_t8(%t5, %t6)
				%t1 = operation <@op11> @op_t1(%t7) {"SpecHLS.gamma"}
				}
			}