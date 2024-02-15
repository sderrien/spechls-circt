
#include "Transforms/Passes.h"
#include "circt/Dialect/SSP/SSPOps.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/StringExtras.h"
#include <circt/Dialect/SSP/SSPPasses.h>
#include <iostream>
#include <mlir/Pass/PassManager.h>

namespace SpecHLS {

struct SchedulePass : public impl::SchedulePassBase<SchedulePass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    float period;
    llvm::SmallVector<circt::ssp::InstanceOp> instanceOps;
    for (auto instOp : moduleOp.getOps<circt::ssp::InstanceOp>()) {
      if (mlir::FloatAttr periodAttr =
              instOp->getAttrOfType<mlir::FloatAttr>("SpecHLS.period")) {
        period = periodAttr.getValue().convertToFloat();
      } else {
        llvm::errs() << "Error: ssp instance must have a 'period' attribute.\n";
        exit(1);
      }
    }
    mlir::OpPassManager dynamicPM("builtin.module");
    std::unique_ptr<mlir::Pass> pass = circt::ssp::createSchedulePass();
    auto optionsResults =
        pass->initializeOptions("options=cycle-time=" + std::to_string(period));
    if (failed(optionsResults)) {
      llvm::errs() << "Error initializing ssp options.\n";
      exit(1);
    }
    dynamicPM.addPass(std::move(pass));
    if (failed(runPipeline(dynamicPM, moduleOp))) {
      llvm::errs() << "Error running circt scheduler.\n";
      exit(1);
    }

    // Scotch to avoid having to implement an ArrayAttr wrapper for eclipse
    for (auto instOp : moduleOp.getOps<circt::ssp::InstanceOp>()) {
      if (instOp->hasAttrOfType<mlir::ArrayAttr>("sspProperties")) {
        for (auto &&property :
             instOp->getAttrOfType<mlir::ArrayAttr>("sspProperties")) {
          if (auto iiAttr = llvm::dyn_cast<circt::ssp::InitiationIntervalAttr>(
                  property)) {
            instOp->setAttr("SpecHLS.II", mlir::IntegerAttr::get(
                                              mlir::IntegerType::get(
                                                  moduleOp.getContext(), 32),
                                              iiAttr.getValue()));
          }
        }
      }
      for (auto &&sspOperation : instOp.getDependenceGraph()) {
        if (sspOperation.hasAttrOfType<mlir::ArrayAttr>("sspProperties")) {
          for (auto &&property :
               sspOperation.getAttrOfType<mlir::ArrayAttr>("sspProperties")) {
            if (auto z = llvm::dyn_cast<circt::ssp::StartTimeInCycleAttr>(
                    property)) {
              sspOperation.setAttr("SpecHLS.z", z.getValue());
            }
            if (auto t = llvm::dyn_cast<circt::ssp::StartTimeAttr>(property)) {
              sspOperation.setAttr(
                  "SpecHLS.t",
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(moduleOp.getContext(), 32),
                      t.getValue()));
            }
          }
        }
      }
    }
  }
};

std::unique_ptr<mlir::Pass> createSchedulePass() {
  return std::make_unique<SchedulePass>();
}

} // namespace SpecHLS