// #include "include/Transforms/YosysOptimizer.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

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
#include "mlir/IR/Verifier.h"              // from @llvm-project
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

#include "mlir/Transforms/InliningUtils.h"

// Block clang-format from reordering
// clang-format off
#include "kernel/yosys.h" // from @at_clifford_yosys
// clang-format on

#define DEBUG_TYPE "yosysoptimizer"
#define VERBOSE false

namespace mlir {
// namespace heir {
using mlir::InlinerInterface;
using std::string;

/// A simple implementation of the `InlinerInterface` that marks all inlining as
/// legal since we know that we only ever attempt to inline `HWModuleOp` bodies
/// at `InstanceOp` sites.
struct PrefixingInliner : public InlinerInterface {
  StringRef prefix;
  PrefixingInliner(MLIRContext *context, StringRef prefix)
      : InlinerInterface(context), prefix(prefix) {}

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }

  void handleTerminator(Operation *op, ValueRange valuesToRepl) const override {
    assert(isa<circt::hw::OutputOp>(op));
    for (auto [from, to] : llvm::zip(valuesToRepl, op->getOperands()))
      from.replaceAllUsesWith(to);
  }

  void processInlinedBlocks(
      iterator_range<Region::iterator> inlinedBlocks) override {
    for (Block &block : inlinedBlocks)
      block.walk([&](Operation *op) { updateNames(op); });
  }

  StringAttr updateName(StringAttr attr) const {
    if (attr.getValue().empty())
      return attr;
    return StringAttr::get(attr.getContext(), prefix + "/" + attr.getValue());
  }

  void updateNames(Operation *op) const {
    if (auto name = op->getAttrOfType<StringAttr>("name"))
      op->setAttr("name", updateName(name));
    if (auto name = op->getAttrOfType<StringAttr>("instanceName"))
      op->setAttr("instanceName", updateName(name));
    if (auto namesAttr = op->getAttrOfType<ArrayAttr>("names")) {
      SmallVector<Attribute> names(namesAttr.getValue().begin(),
                                   namesAttr.getValue().end());
      for (auto &name : names)
        if (auto nameStr = name.dyn_cast<StringAttr>())
          name = updateName(nameStr);
      op->setAttr("names", ArrayAttr::get(namesAttr.getContext(), names));
    }
  }
};

void listHWModuleOps(mlir::ModuleOp module) {
  module->walk([](circt::hw::HWModuleOp hwop) {
    if (VERBOSE) llvm::errs() << "HWModuleOp " << hwop.getSymName().str() << "\n";
  });
}

bool hasConstantOutputs(circt::hw::HWModuleOp op) {
  for (auto &_innerop : op.getBodyBlock()->getOperations()) {
    bool ok = TypeSwitch<Operation *, bool>(&_innerop)
                  .Case<circt::hw::ConstantOp>([&](auto op) { return true; })
                  .Case<circt::hw::OutputOp>([&](auto op) { return true; })
                  .Default([&](auto op) {
                    if (VERBOSE) llvm::errs()
                        << "Operation " << _innerop << "is not constant\n";
                    return false;
                  });

    if (!ok)
      return false;
  }
  return true;
}

struct YosysOptimizer : public SpecHLS::impl::YosysOptimizerBase<YosysOptimizer> {
  using YosysOptimizerBase::YosysOptimizerBase;
  
  YosysOptimizer(SpecHLS::YosysOptimizerOptions &options) {
    replace = options.replace;
  }
  YosysOptimizer(string yosysFilesPath, string abcPath, bool abcFast)
      : yosysFilesPath(yosysFilesPath), abcPath(abcPath), abcFast(abcFast) {}

  void runOnOperation() override;

private:
  // Path to a directory containing yosys techlibs.
  std::string yosysFilesPath;
  // Path to ABC binary.
  std::string abcPath;

  bool abcFast;
};

circt::hw::HWModuleOp yosysBackend(MLIRContext *context,
                                   circt::hw::HWModuleOp op, bool replace) {
  delete Yosys::yosys_design;
  Yosys::yosys_design = new Yosys::RTLIL::Design;

  string filename = string(op.getName().str()) + ".sv";
  if (!std::filesystem::exists(filename)) {
    if (VERBOSE) llvm::errs() << "File  " << filename << " does not exists\n";
    return NULL;
  }

  string toplevel = string(op.getName().str());

  Yosys::log_error_stderr = true;
  LLVM_DEBUG(Yosys::log_streams.push_back(&std::cout));

  auto start = std::chrono::high_resolution_clock::now();
  auto command =  "read_verilog " + filename ;
  Yosys::run_pass(command);
  Yosys::run_pass("dump;   ");
  Yosys::run_pass("proc; flatten;   ");
  Yosys::run_pass("opt -full;   ");
//#ifdef USE_YOSYS_ABC
  Yosys::run_pass("synth -noabc ;  ");
//#endif
  Yosys::run_pass("abc -exe \"/opt/yosys/yosys-abc\" -g AND,OR");
  Yosys::run_pass("write_verilog " + string(op.getName().str()) + "_yosys.sv");
  Yosys::run_pass("hierarchy -generate * o:Y i:*; opt; opt_clean -purge");
  Yosys::run_pass("clean -purge");

  auto stop = std::chrono::high_resolution_clock::now();
  std::stringstream cellOrder;
  Yosys::log_streams.push_back(&cellOrder);
  Yosys::run_pass("torder -stop * P*;");
  Yosys::log_streams.clear();
  auto topologicalOrder = getTopologicalOrder(cellOrder);
  RTLILImporter lutImporter = RTLILImporter(context);
  Yosys::RTLIL::Design *design = Yosys::yosys_get_design();

  circt::hw::HWModuleOp submodule =
      lutImporter.importModule(op, design->top_module(), topologicalOrder);

  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

  llvm::errs() << "Yosys synthesis successfull for " << op.getSymName() << " ("<< duration.count() << " ms) \n";
#ifdef VERBOSE
  if (VERBOSE) llvm::errs() << "Created module : " << submodule << "  \n";
#endif
  return submodule;
}

SmallVector<circt::hw::InstanceOp> listAllInstances(mlir::ModuleOp module,
                                                    circt::hw::HWModuleOp op) {
  SmallVector<circt::hw::InstanceOp> instances;

  module->walk([&instances, &op](circt::hw::HWModuleOp hwop) {
    if (VERBOSE) llvm::errs() << "in " << hwop.getName() << "\n";
    hwop->walk([&instances, &op](circt::hw::InstanceOp inst) {
      if (VERBOSE) llvm::errs() << "   - found instance  " << inst << "\n";
      if (inst.getModuleName() == op.getModuleName()) {
        if (VERBOSE) llvm::errs() << "     - save instance  " << inst << "\n";
        instances.push_back(inst);
      }
      return WalkResult::advance();
    });
    return WalkResult::advance();
  });

  return instances;
}

bool isSynthesizableModule(circt::hw::HWModuleOp op) {
  for (auto &_innerop : op.getBodyBlock()->getOperations()) {
    if (!SpecHLS::isControlLogicOperation(&_innerop))
      return false;
  }
  return true;
}


LogicalResult replaceInstance(circt::hw::HWModuleOp old,
                               circt::hw::HWModuleOp _new) {

  auto oldSym = old.getSymNameAttr().getValue();
  auto newSym = _new.getSymNameAttr();
  auto module = dyn_cast<ModuleOp>(old->getParentOp());

  if (module == NULL) {
    llvm::errs() << "Cannot extract module op for   " << old << "\n";
    return failure();
  }
  auto instances = listAllInstances(module, old);
  if (VERBOSE) llvm::errs() << "Found " << instances.size() << " instances of "
               << old.getName() << "\n";
  auto isConstant = hasConstantOutputs(_new);
  // Set modul as private to enable inliing using arc inline pass
  if (isConstant)
    _new.setPrivate();

  for (auto inst : instances) {
    // Check if the operation is a CallOp
    if (VERBOSE) llvm::errs() << "instance " << inst << "\n";
    auto instName = inst.getModuleNameAttr().getValue();
    if (VERBOSE) llvm::errs() << " - replace instance : " << inst;
    if (instName == oldSym) {
      // Create a new CallOp with the replacement function
      inst.setModuleName(newSym);
      if (VERBOSE) llvm::errs() << "\n\t\t by  " << inst;
    }
    if (VERBOSE) llvm::errs() << "\n";

    if (isConstant) {
    }
  }
  old->remove();

  return success();
}

void YosysOptimizer::runOnOperation() {

  auto module = dyn_cast<ModuleOp>(getOperation());

  // Une des difficultés et que l'API d'export de verilog fonctionne au niveau
  // ModuleOP et pas au niveau HWModuleOp. On commence donc par cloner le module
  // Pour ne conserver que les HWModules identifiés comme  synthetizables et
  // pour lesquelles on trouve un Pragma Control_Node

  // supprime les HWModule qui ne seront pas optimisés via Yosys

  module->walk([&](circt::hw::HWModuleOp op) {
    if (VERBOSE) llvm::errs() << "Analyzing HWModule " << op.getSymName() << "\n";
    if (SpecHLS::hasControlNodePragma( op)) {
      if (VERBOSE) llvm::errs() << "   - module " << op.getSymName()
                   << " has control node pragma\n";

      OpPassManager dynamicPM("hw.module");

      // Add a pass on the top-level module operation.
      dynamicPM.addPass(SpecHLS::createConvertSpecHLSToCombPass());
      if (failed(runPipeline(dynamicPM, op))) {
        llvm::errs() << "   - error for " << op.getSymName() << " \n";
      } else {
        if (VERBOSE) llvm::errs() << "   - module lowered to \n" << op << " \n";
      }
    }

    return WalkResult::advance();
  });


  auto clone = dyn_cast<ModuleOp>(getOperation()->clone());
  if (clone == NULL || module == NULL) {
    if (VERBOSE) llvm::errs() << "op " << clone->getName() << " not supported  \n";
    signalPassFailure();
  }

  DenseMap<circt::hw::HWModuleOp, circt::hw::HWModuleOp> cloneMap;

  // supprime les HWModule qui ne seront pas optimisés via Yosys
  clone->walk([&](circt::hw::HWModuleOp op) {
    if (VERBOSE) llvm::errs() << "Analyzing HWModule " << op.getSymName() << "\n";
    if (SpecHLS::hasControlNodePragma( op)) {
      if (VERBOSE) llvm::errs() << "   - module " << op.getSymName()
                   << " has control node pragma\n";


      if (isSynthesizableModule(op)) {
        if (VERBOSE) llvm::errs() << "   - module " << op.getSymName()
                     << " will be optimized through Yosys\n";
        if (VERBOSE) llvm::errs() << op << "\n";

        return WalkResult::advance();
      }
    }
    if (VERBOSE) llvm::errs() << "   - module " << op.getSymName() << " is ignored\n";
    op->remove();
    return WalkResult::advance();
  });

  clone->dump();
  circt::exportSplitVerilog(clone, "./");


  SmallVector<std::string> optimizedModules;

  clone->walk([&](circt::hw::HWModuleOp cloneop) {
    module->walk([&](circt::hw::HWModuleOp op) {
      if (cloneop.getSymName() == op.getSymName()) {
        if (VERBOSE) llvm::errs() << " match " << op.getName() << "\n";
        cloneMap[cloneop] = op;
      }
    });
  });
  Yosys::yosys_setup();
  // applique Yosys sur tout les HWModules restants
  auto result = clone->walk([&](circt::hw::HWModuleOp op) {
    if (VERBOSE) llvm::errs() << "Optimizing module " << op.getName() << "\n";
    circt::hw::HWModuleOp optimized = yosysBackend(&getContext(), op, replace);

    if (optimized == NULL) {
      op.emitError("Yosys synthesis failed for module " + op.getName());
      return WalkResult::skip();

    } else {
      optimizedModules.push_back(op.getSymName().str());
      module.push_back(optimized);

        if (VERBOSE) llvm::errs() << "replacing " << op.getName() << " by " << optimized.getName() << "\n";
        auto originalOp = cloneMap[op];
        if (originalOp == NULL) {
          op.emitError("error  " + op.getName() + " by " + optimized.getName() +
                       "\n");
          return WalkResult::interrupt();
        }
        replaceInstance(originalOp, optimized);
    }
    return WalkResult::advance();
  });
  Yosys::yosys_shutdown();

  if (result.wasInterrupted()) {
    if (VERBOSE) llvm::errs() << "Yosys pass failed \n";
    signalPassFailure();
  }
  mlir::verify(mlir::OperationPass<ModuleOp>::getOperation(),true);
}

} // namespace mlir
namespace SpecHLS {

std::unique_ptr<mlir::Pass> createYosysOptimizerPass() {
  return std::make_unique<mlir::YosysOptimizer>();
}

//}  // namespace heir
} // namespace SpecHLS