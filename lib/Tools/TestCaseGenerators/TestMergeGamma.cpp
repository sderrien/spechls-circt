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

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardOps.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;

int main() {
  // Create MLIR context
  MLIRContext context;

  // Create an MLIR module
  auto module = ModuleOp::create("example_module", &context);

  // Create an MLIR function
  auto hwmodule = circt::hw::HWModuleOp::create(UnknownLoc::get(&context), "example_function");

//
//  // Add an MLIR operation to the block (e.g., a constant operation)
//  auto constantOp = builder.create<ConstantOp>(UnknownLoc::get(&context),
//                                               IntegerType::get(&context, 32),
//                                               builder.getI32IntegerAttr(42));

  // Add the constant operation as the return value
  builder.create<ReturnOp>(UnknownLoc::get(&context), constantOp.getResult());

  // Add the function to the module
  module->push_back(func);

  // Verify the module
  if (failed(verify(*module))) {
    module->dump();
    return 1;
  }

  // Dump the module
  module->dump();

  return 0;
}