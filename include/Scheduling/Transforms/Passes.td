#ifndef SPECHLS_SCHEDULE_PASSES
#define SPECHLS_SCHEDULE_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"
include "SpecHLS/SpecHLSDialect.td"

def GecosSchedulePass : Pass<"gecos-schedule", "mlir::ModuleOp"> {
  let summary = "Schedule GammaMobilityProblem with ASAP.";
  let description = [{
    Schedule GammaMobilityProblem with ASAP.
  }];

let options = [
    Option<"schedulerOptions", "options", "std::string", "",
               "Scheduler-specific options.">
  ];

  let constructor = "SpecHLS::createGecosSchedulePass()";
  let dependentDialects = ["circt::ssp::SSPDialect"];
}
#endif // SPECHLS_TRANSFORMS_PASSES