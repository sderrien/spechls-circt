// #include "include/Transforms/YosysOptimizer.h"

#include <cassert>
#include <chrono>
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

#include "SpecHLS/SpecHLSUtils.h"

#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h" // from @llvm-project
#include "mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/IR/DialectRegistry.h"       // from @llvm-project
#include "mlir/IR/Location.h"              // from @llvm-project
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"              // from @llvm-project
#include "mlir/IR/Value.h"              // from @llvm-project
#include "mlir/IR/ValueRange.h"         // from @llvm-project
#include "mlir/IR/Visitors.h"           // from @llvm-project
#include "mlir/Pass/PassManager.h"      // from @llvm-project
#include "mlir/Pass/PassRegistry.h"     // from @llvm-project
#include "mlir/Support/LLVM.h"          // from @llvm-project
#include "mlir/Support/LogicalResult.h" // from @llvm-project
#include "mlir/Transforms/Passes.h"     // from @llvm-project
#include "llvm/ADT/SmallVector.h"       // from @llvm-project
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"          // from @llvm-project
#include "llvm/Support/FormatVariadic.h" // from @llvm-project
#include "llvm/Support/raw_ostream.h"    // from @llvm-project

// Block clang-format from reordering
// clang-format off
#include "kernel/yosys.h" // from @at_clifford_yosys
// clang-format on

#define DEBUG_TYPE "yosysoptimizer"


namespace mlir {
// namespace heir {
using std::string;


struct YosysOptimizer
    : public SpecHLS::impl::YosysOptimizerBase<YosysOptimizer> {
  using YosysOptimizerBase::YosysOptimizerBase;

  YosysOptimizer(SpecHLS::YosysOptimizerOptions &options) {
    replace = options.replace;
  }
  YosysOptimizer()
      : yosysFilesPath(yosysFilesPath), abcPath(abcPath), abcFast(abcFast) {}

  void runOnOperation() override;

private:
  // Path to a directory containing yosys techlibs.
  std::string yosysFilesPath;
  // Path to ABC binary.
  std::string abcPath;

  bool abcFast;
};

circt::hw::HWModuleOp yosysBackend(MLIRContext *context, circt::hw::HWModuleOp op, bool replace) {

  string filename = string(op.getName().str()) + ".sv";
  if (!std::filesystem::exists(filename)) {
    llvm::outs() << "File  " << filename << " does not exists\n";
    return NULL;
  }

  string toplevel = string(op.getName().str());

  Yosys::yosys_setup();
  Yosys::log_error_stderr = true;
  LLVM_DEBUG(Yosys::log_streams.push_back(&std::cout));

  auto start = std::chrono::high_resolution_clock::now();
  Yosys::run_pass("read_verilog " + filename + "  ");
  Yosys::run_pass("proc; flatten;   ");
  Yosys::run_pass("opt -full;   ");
#ifdef USE_YOSYS_ABC
  Yosys::run_pass("synth -noabc ;  ");
#endif
  Yosys::run_pass("abc -exe \"/opt/yosys/yosys-abc\" -g AND,OR,XOR");
  Yosys::run_pass("write_verilog " + string(op.getName().str()) + "_yosys.sv");
  Yosys::run_pass("hierarchy -generate * o:Y i:*; opt; opt_clean -purge");
  Yosys::run_pass("clean -purge");

  auto stop = std::chrono::high_resolution_clock::now();
  std::stringstream cellOrder;
  Yosys::log_streams.push_back(&cellOrder);
  Yosys::run_pass("torder -stop * P*;");
  Yosys::log_streams.clear();
  auto topologicalOrder = getTopologicalOrder(cellOrder);
  LUTImporter lutImporter = LUTImporter(context);
  Yosys::RTLIL::Design *design = Yosys::yosys_get_design();

  circt::hw::HWModuleOp submodule =
      lutImporter.importModule(design->top_module(), topologicalOrder);

  Yosys::yosys_shutdown();
  auto duration = duration_cast<std::chrono::milliseconds>(stop - start);

  llvm::outs() << "Yosys synthesis successfull for " << op.getSymName() << " (" << duration.count() << " ms) \n";
#ifdef VERBOSE
  llvm::outs() << "Created module : " << submodule << "  \n";
#endif
    return submodule;
}

SmallVector<circt::hw::InstanceOp> listAllInstances(mlir::ModuleOp module, circt::hw::HWModuleOp op ) {
  SmallVector<circt::hw::InstanceOp> instances;
  auto result = module->walk([&instances](Operation *_op) {
    //llvm::outs() << "op   " << *_op << "\n";
    if (auto inst = dyn_cast<circt::hw::InstanceOp>(_op)) {
      llvm::outs() << "   found instance  " << inst.getModuleName() << "\n";
      instances.push_back(inst);
    }
    return WalkResult::advance();
  });
  return instances;
}

// Optimize the body of a secret.generic op.
// FIXME: consider utilizing
// https://mlir.llvm.org/docs/PassManagement/#dynamic-pass-pipelines

bool isSynthesizableModule(circt::hw::HWModuleOp op) {
  for (auto &_innerop : op.getBodyBlock()->getOperations()) {
    if (!SpecHLS::isControlLogicOperation (&_innerop)) return false;
  }
  return true;
}

bool hasControlNodePragma(mlir::MLIRContext *ctxt, Operation *op) {

  auto attr = op->getAttr(StringRef("#pragma"));
  if (attr != NULL) {
    llvm::outs() << "pragma " << attr << "\n";
    if (auto strAttr = attr.dyn_cast<mlir::StringAttr>()) {
      // Compare the attribute value with an existing string
      llvm::StringRef existingString = "CONTROL_NODE";
      if (strAttr.getValue().contains(existingString)) {
        return true;
      }
    }
  }
  return false;
}

void YosysOptimizer::runOnOperation() {

  //  mlir::MLIRContext ctxt;
  mlir::MLIRContext *ctxt = getOperation()->getContext();
  auto module = dyn_cast<ModuleOp>(getOperation());
  auto clone = dyn_cast<ModuleOp>(getOperation()->clone());

  llvm::DenseMap<circt::hw::HWModuleOp, circt::hw::HWModuleOp> hwMap;
  for (auto chwop : clone.getOps<circt::hw::HWModuleOp>()) {

  }

  if (clone==NULL || module==NULL) {
    llvm::outs() << "op " << clone->getName() << " not supported  \n";
    signalPassFailure();
  }

  clone->walk([&](circt::hw::HWModuleOp op) {
    llvm::outs() << "Analyzing HWModule " << op.getSymName() << "\n";

    if (hasControlNodePragma(ctxt, op)) {
      llvm::outs() << "   - module " << op.getSymName() << " has control node pragma\n";
      if (isSynthesizableModule(op)) {
        llvm::outs() << "   - module " << op.getSymName()
                     << " will be optimized through Yosys\n";
        return WalkResult::advance();
      }
    }
    llvm::outs() << "   - module " << op.getSymName() << " is ignored\n";
    op->remove();
    return WalkResult::advance();
  });

  circt::exportSplitVerilog(clone, "./");

  auto cloneHWmodules =clone.getOps<circt::hw::HWModuleOp>();

  SmallVector<std::string> optimizedModules;

  auto result = clone->walk([&](circt::hw::HWModuleOp op) {

    llvm::outs() << "Optimizing module " << op.getName() << "\n";
    circt::hw::HWModuleOp optimized = yosysBackend(&getContext(), op, replace);

    if (optimized==NULL) {
      llvm::outs() << "Yosys synthesis failed for module " << op.getName() << "\n";
    } else {
      optimizedModules.push_back(op.getSymName().str());
      module.push_back(optimized);
      if (replace) {
        auto oldSym = op.getSymNameAttr().getValue();
        auto newSym = optimized.getSymNameAttr();
        auto instances = listAllInstances(module,op);

        for (auto inst : instances) {
          // Check if the operation is a CallOp
          auto instName = inst.getModuleNameAttr().getValue();
          llvm::outs() << " - replace instance : " << inst ;
          if (instName == oldSym) {
            // Create a new CallOp with the replacement function
            inst.setModuleName(newSym);
            llvm::outs() << "\n\t\t by  " << inst ;
          }
          llvm::outs() << "\n";
        }
      }
    }
    // Iterate over HWModuleOp instances in the cloned module
    for (auto hwop : module.getOps<circt::hw::HWModuleOp>()) {
      llvm::outs() << "Analyzing module "<< hwop.getSymName() << " \n" ;
      for (auto name : optimizedModules) {
        if (name==hwop.getSymName().str())  {
          llvm::outs() << "Need to remove "<< hwop.getSymName() << " \n" ;
          hwop->remove();
        }
      }
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    llvm::outs() << "Yosys pass failed \n";
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
      });
}

//}  // namespace heir
} // namespace SpecHLS
