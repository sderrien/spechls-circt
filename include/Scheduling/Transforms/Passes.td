#ifndef SPECHLS_SCHEDULING_PASSES
#define SPECHLS_SCHEDULING_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"
include "SpecHLS/SpecHLSDialect.td"

def MobilityPass : Pass<"mobility", "mlir::ModuleOp"> {
  let summary = "Compute mobility informations.";
  let description = [{
    Compute mobility informations.
  }];

let options = [
    Option<"period", "period", "float", "",
               "clock period time.">
  ];

  let constructor = "SpecHLS::createMobilityPass()";
  let dependentDialects = ["circt::ssp::SSPDialect"];
}
#endif // SPECHLS_SCHEDULING_PASSES