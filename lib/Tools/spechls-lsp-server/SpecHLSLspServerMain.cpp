#include "Dialect/ScheduleDialect/ScheduleDialectDialect.h"
#include "Dialect/ScheduleDialect/ScheduleDialectOps.cpp.inc"
#include "Dialect/ScheduleDialect/ScheduleDialectOps.h"
#include "Dialect/ScheduleDialect/ScheduleDialectOpsDialect.cpp.inc"
#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.cpp.inc"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSOpsDialect.cpp.inc"
#include "InitAllDialects.h"
#include "circt/InitAllDialects.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  circt::registerAllDialects(registry);
  SpecHLS::registerAllDialects(registry);

  return failed(mlir::MlirLspServerMain(argc, argv, registry));
}