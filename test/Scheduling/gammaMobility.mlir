ssp.instance @mobility of "GammaMobilityProblem" {
  library {
    operator_type @_mu [latency<1>, incDelay<0.0>, outDelay<0.0>]
    operator_type @_compute [latency<1>, incDelay<0.0>, outDelay<0.0>]
    operator_type @_gamma [latency<1>, incDelay<0.0>, outDelay<0.0>]
    operator_type @_delay [latency<0>, incDelay<0.0>, outDelay<0.0>]
  }
  graph {
    %0 = operation<@_mu> @mu1(@op1 [dist<1>])
    %1 = operation<@_mu> @mu2(@gamma [dist<1>])
    %2 = operation<@_compute> @op1(%0)
    %3 = operation<@_compute> @op2(%2)
    %4 = operation<@_compute> @op3(%3)
    %5 = operation<@_compute> @op4(%1)
    %6 = operation<@_compute> @op5(%5)
    %8 = operation<@_delay> @op6(@op3 [dist<1>])
    %9 = operation<@_delay> @op7(@op5 [dist<1>])
    %7 = operation<@_gamma> @gamma(%8, %9) {"SpecHLS.gamma"}
  }
}