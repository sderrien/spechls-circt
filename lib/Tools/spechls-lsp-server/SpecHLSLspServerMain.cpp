#include "InitAllDialects.h"
#include "SpecHLS/SpecHLSDialect.h"
#include "SpecHLS/SpecHLSOps.h"
#include "SpecHLS/SpecHLSOpsDialect.cpp.inc"
#include "SpecHLS/SpecHLSOps.cpp.inc"
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