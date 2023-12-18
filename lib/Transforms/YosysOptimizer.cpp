// #include "include/Transforms/YosysOptimizer.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

#include "LUTImporter.h"
#include "RTLILImporter.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"

#include "circt/Conversion/ExportVerilog.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h" // from @llvm-project
#include "mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/IR/DialectRegistry.h"       // from @llvm-project
#include "mlir/IR/Location.h"              // from @llvm-project
#include "mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/IR/ValueRange.h"            // from @llvm-project
#include "mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/Pass/PassManager.h"         // from @llvm-project
#include "mlir/Pass/PassRegistry.h"        // from @llvm-project
#include "mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/Transforms/Passes.h"        // from @llvm-project
#include "llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/Support/Debug.h"            // from @llvm-project
#include "llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "llvm/Support/raw_ostream.h"      // from @llvm-project

#include "Transforms/Passes.h"

// Block clang-format from reordering
// clang-format off
#include "kernel/yosys.h" // from @at_clifford_yosys
// clang-format on

#define DEBUG_TYPE "yosysoptimizer"

namespace mlir {
// namespace heir {
using std::string;

// $0: verilog filename
// $1: function name
// $2: yosys runfiles
// $3: abc path
// $4: abc fast option -fast
constexpr std::string_view kYosysTemplate = R"(
read_verilog {0};
hierarchy -check -top \{1};
proc; memory;
techmap -map {2}/techmap.v; opt;
abc -exe {3} -lut 3 {4};
opt_clean -purge;
rename -hide */c:*; rename -enumerate */c:*;
techmap -map {2}/map_lut_to_lut3.v; opt_clean -purge;
hierarchy -generate * o:Y i:*; opt; opt_clean -purge;
clean;
)";

struct YosysOptimizer
    : public SpecHLS::impl::YosysOptimizerBase<YosysOptimizer> {
  using YosysOptimizerBase::YosysOptimizerBase;

  YosysOptimizer()
      : yosysFilesPath(yosysFilesPath), abcPath(abcPath), abcFast(abcFast) {}

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<circt::comb::CombDialect, mlir::arith::ArithDialect,
                    mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override;

private:
  // Path to a directory containing yosys techlibs.
  std::string yosysFilesPath;
  // Path to ABC binary.
  std::string abcPath;

  bool abcFast;
};

LogicalResult runOnGenericOp(MLIRContext *context, circt::hw::HWModuleOp op) {

  // Translate function to Verilog. Translation will fail if the func contains
  // unsupported operations.
  // TODO(https://github.com/google/heir/issues/111): Directly convert MLIR to
  // Yosys' AST instead of using Verilog.
  //
  // After that is done, it might make sense to rewrite this as a
  // RewritePattern, which only runs if the body does not contain any comb ops,
  // and generalize this to support converting a secret.generic as well as a
  // func.func. It's necessary to wait for the migration because the Yosys API
  // used here maintains global state that apparently does not play nicely with
  // the instantiation of multiple rewrite patterns.

  // Invoke Yosys to translate to a combinational circuit and optimize.

std:
  string filename = string(op.getName().str()) + ".sv";
  string toplevel = string(op.getName().str());
  llvm::outs() << "input filename " << filename << "\n";
  Yosys::yosys_setup();
  Yosys::log_error_stderr = true;
  LLVM_DEBUG(Yosys::log_streams.push_back(&std::cout));

  llvm::outs() << "before yosys" << op.getName() << "\n";

  auto command = llvm::formatv(kYosysTemplate.data(), filename, toplevel, "./",
                               "./", "-fast");

  //  llvm::outs() << "yosys command " << command << "\n";
  //  Yosys::run_pass(command);

  Yosys::run_pass("read_verilog " + filename + "  ");
  llvm::outs() << "after read_verilog " << filename << "\n";
  Yosys::run_pass("proc;");
  //  Yosys::run_pass("proc; flatten; opt -full; synth ; abc -g AND,OR,XOR;");
  llvm::outs() << "after proc, flatten "
               << "\n";
  //  Yosys::run_pass("clean -purge");
  //  llvm::outs() << "after clean -purge " << "\n";
  //  Yosys::run_pass("write_blif"+toplevel+".blif");
  //  llvm::outs() << "write_blif"+toplevel+".blif" << "\n";

  // Translate Yosys result back to MLIR and insert into the func
  LLVM_DEBUG(Yosys::run_pass("dump;"));
  llvm::outs() << "after dump " << filename << "\n";
  std::stringstream cellOrder;
  Yosys::log_streams.push_back(&cellOrder);
  Yosys::run_pass("torder -stop * P*;");
  Yosys::log_streams.clear();
  auto topologicalOrder = getTopologicalOrder(cellOrder);
  LUTImporter lutImporter = LUTImporter(context);
  Yosys::RTLIL::Design *design = Yosys::yosys_get_design();
  func::FuncOp func =
      lutImporter.importModule(design->top_module(), topologicalOrder);
  Yosys::yosys_shutdown();

  // The pass changes the yielded value types, e.g., from an i8 to a
  // tensor<8xi1>. So the containing secret.generic needs to be updated and
  // conversions implemented on either side to convert the ints to tensors
  // and back again.
  //
  // convertOpOperands goes from i8 -> tensor.tensor<8xi1>
  // converOpResults from tensor.tensor<8xi1> -> i8
  SmallVector<Value> typeConvertedArgs;
  typeConvertedArgs.reserve(op->getNumOperands());
  //  if (failed(convertOpOperands(op, func, typeConvertedArgs))) {
  //    return failure();
  //  }

  int resultIndex = 0;
  //  for (Type ty : func.getFunctionType().getResults())
  //    op->getResult(resultIndex++).setType(secret::SecretType::get(ty));

  // Replace the func.return with a secret.yield
  op.getRegion().takeBody(func.getBody());
  op.getOperation()->setOperands(typeConvertedArgs);

  Block &block = op.getRegion().getBlocks().front();
  func::ReturnOp returnOp = cast<func::ReturnOp>(block.getTerminator());
  OpBuilder bodyBuilder(&block, block.end());
  //  bodyBuilder.create<secret::YieldOp>(returnOp.getLoc(),
  //                                      returnOp.getOperands());
  //  returnOp.erase();
  //  func.erase();

  DenseSet<Operation *> castOps;
  SmallVector<Value> typeConvertedResults;
  castOps.reserve(op->getNumResults());
  typeConvertedResults.reserve(op->getNumResults());
  //  if (failed(convertOpResults(op, castOps, typeConvertedResults))) {
  //    return failure();
  //  }

  LLVM_DEBUG(llvm::dbgs() << "Generic results: " << typeConvertedResults.size()
                          << "\n");
  //  LLVM_DEBUG(llvm::dbgs() << "Original results: " << op.getResults().size()
  //                          << "\n");

  //  op.getResults().replaceUsesWithIf(
  //      typeConvertedResults, [&](OpOperand &operand) {
  //        return !castOps.contains(operand.getOwner());
  //      });
  return success();
}

// Optimize the body of a secret.generic op.
// FIXME: consider utilizing
// https://mlir.llvm.org/docs/PassManagement/#dynamic-pass-pipelines
void YosysOptimizer::runOnOperation() {
  auto result = getOperation()->walk([&](circt::hw::HWModuleOp op) {
    llvm::outs() << "found moduleop" << op.getName();
    if (failed(runOnGenericOp(&getContext(), op))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
  }
}

} // namespace mlir
namespace SpecHLS {

std::unique_ptr<mlir::Pass> createYosysOptimizer() {
  return std::make_unique<mlir::YosysOptimizer>();
}

void registerYosysOptimizerPipeline() {

  mlir::PassPipelineRegistration<> pipeline(
      "yosys-optimizer", "The yosys optimizer pipeline.",
      [](mlir::OpPassManager &pm) {
        pm.addPass(circt::createExportSplitVerilogPass());
        pm.addPass(SpecHLS::createYosysOptimizer());
        pm.addPass(mlir::createCSEPass());
      });
}

//}  // namespace heir
} // namespace SpecHLS
