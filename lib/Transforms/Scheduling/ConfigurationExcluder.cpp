#include "Transforms/Passes.h"
#include "circt/Dialect/SSP/SSPOps.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/StringExtras.h"
#include <circt/Dialect/SSP/SSPAttributes.h>
#include <iostream>
#include <memory>

namespace SpecHLS {

struct ConfigurationExcluderPass
    : public impl::ConfigurationExcluderPassBase<ConfigurationExcluderPass> {
  void runOnOperation() override {
    auto circuitOp = getOperation();
    double targetClock =
        circuitOp->getAttrOfType<mlir::FloatAttr>("targetClock")
            .getValueAsDouble();
    auto &region = circuitOp->getRegion(0);
    int sumDistances = 0;

    for (auto &op : region.getOps()) {
      for (auto &attr : op.getAttrOfType<mlir::ArrayAttr>("distances")) {
        auto distanceAttr = llvm::cast<mlir::IntegerAttr>(attr);
        sumDistances += distanceAttr.getInt();
      }
    }
    int iterationCount = 2 * sumDistances + 2;
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<int>> startTimes;
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<double>>
        startTimeInCycles;


    for (auto &op : region.getOps()) {
      startTimes.try_emplace(&op, llvm::SmallVector<int>());
      startTimes[&op].reserve(iterationCount);
      startTimeInCycles.try_emplace(&op, llvm::SmallVector<double>());
      startTimeInCycles[&op].reserve(iterationCount);
    }

    for (int iteration = 0; iteration < iterationCount; ++iteration) {
      for (auto &op : region.getOps()) {
        auto distanceArray =
            op.getAttrOfType<mlir::ArrayAttr>("distances").getValue();

        bool isGamma = op.hasAttr("SpecHLS.gamma");
        bool isMu = op.hasAttr("SpecHLS.mu");
        int nextCycle = isGamma ? std::numeric_limits<int>::max() : 0;
        double nextTimeInCycle = 0.0;

        // Compute unrolled schedule
        for (unsigned predIndex = 0; predIndex < op.getNumOperands();
             ++predIndex) {
          int distance =
              llvm::cast<mlir::IntegerAttr>(distanceArray[predIndex]).getInt();
          if (iteration - distance < 0) {
            nextCycle = 0;
            nextTimeInCycle = 0.0;
          } else {
            int predEndCycle = 0;
            double predEndTimeInCycle = 0.0;

            mlir::Operation *pred = op.getOperand(predIndex).getDefiningOp();
            int predLatency =
                pred->getAttrOfType<mlir::IntegerAttr>("latency").getInt();
            double predInDelay = pred->getAttrOfType<mlir::FloatAttr>("inDelay")
                                     .getValueAsDouble();
            double predOutDelay =
                pred->getAttrOfType<mlir::FloatAttr>("outDelay")
                    .getValueAsDouble();

            int diff = iteration - distance;
            if (diff >= 0 &&
                diff < static_cast<ptrdiff_t>(startTimes[pred].size())) {
              if (predLatency > 0) {
                predEndCycle =
                    startTimes[pred][iteration - distance] + predLatency;
                predEndTimeInCycle = predOutDelay;
              } else if (startTimeInCycles[pred][iteration - distance] +
                             predInDelay + predOutDelay >
                         targetClock) {
                predEndCycle = startTimes[pred][iteration - distance] + 1;
                predEndTimeInCycle = predOutDelay;
              } else {
                predEndCycle = startTimes[pred][iteration - distance];
                predEndTimeInCycle =
                    startTimeInCycles[pred][iteration - distance] +
                    predOutDelay;
              }

              if (isGamma) {
                if (nextCycle > predEndCycle) {
                  nextCycle = predEndCycle;
                  nextTimeInCycle = predEndTimeInCycle;
                } else if (nextCycle == predEndCycle) {
                  nextTimeInCycle =
                      std::min(nextTimeInCycle, predEndTimeInCycle);
                }
              } else if (predEndCycle > nextCycle) {
                nextCycle = predEndCycle;
                nextTimeInCycle = predEndTimeInCycle;
              } else if (predEndCycle == nextCycle) {
                nextTimeInCycle = std::max(nextTimeInCycle, predEndTimeInCycle);
              }
            } else {
              nextCycle = 0;
              nextTimeInCycle = 0.0;
            }
          }
        }

        if (isMu && (iteration > sumDistances + 1) &&
            (nextCycle - startTimes[&op][iteration - 1] > 1)) {
          circuitOp->setAttr(
              "SpecHLS.allowUnitII",
              mlir::BoolAttr::get(circuitOp.getContext(), false));
          return;
        }

        startTimes[&op].push_back(nextCycle);
        startTimeInCycles[&op].push_back(nextTimeInCycle);
      }
    }

    circuitOp->setAttr("SpecHLS.allowUnitII",
                       mlir::BoolAttr::get(circuitOp.getContext(), true));
  }
};

std::unique_ptr<mlir::Pass> createConfigurationExcluderPass() {
  return std::make_unique<ConfigurationExcluderPass>();
}

} // namespace SpecHLS
