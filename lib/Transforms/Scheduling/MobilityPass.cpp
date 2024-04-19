#include "Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/StringExtras.h"

#include "Dialect/ScheduleDialect/ScheduleDialectDialect.h"
#include "Dialect/ScheduleDialect/ScheduleDialectOps.h"
#include "Dialect/ScheduleDialect/ScheduleDialectOpsDialect.cpp.inc"
#include "Dialect/ScheduleDialect/ScheduleDialectOpsTypes.h.inc"

namespace SpecHLS {

struct MobilityPass : public impl::MobilityPassBase<MobilityPass> {
  void runOnOperation() override {
    auto circuitOp = getOperation();
    double targetClock =
        circuitOp->getAttrOfType<mlir::FloatAttr>("targetClock")
            .getValueAsDouble();
    auto &region = circuitOp->getRegion(0);

    int sumDistances = 0;
    llvm::SmallVector<mlir::Operation *> gammas;

    for (auto &op : region.getOps()) {
      if (op.hasAttr("SpecHLS.gamma"))
        gammas.push_back(&op);
      for (auto &attr : op.getAttrOfType<mlir::ArrayAttr>("distances")) {
        auto distanceAttr = llvm::cast<mlir::IntegerAttr>(attr);
        sumDistances += distanceAttr.getInt();
      }
    }
    int iterationCount = 2 * sumDistances + 2;

    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<int>> startTimesAsap;
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<double>>
        startTimeInCyclesAsap;
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<int>> startTimesAlap;
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<double>>
        startTimeInCyclesAlap;

    for (auto &op : region.getOps()) {
      startTimesAsap.try_emplace(&op, llvm::SmallVector<int>());
      startTimesAsap[&op].reserve(iterationCount);

      startTimeInCyclesAsap.try_emplace(&op, llvm::SmallVector<double>());
      startTimeInCyclesAsap[&op].reserve(iterationCount);

      startTimesAlap.try_emplace(&op, llvm::SmallVector<int>());
      startTimesAlap[&op].reserve(iterationCount);

      startTimeInCyclesAlap.try_emplace(&op, llvm::SmallVector<double>());
      startTimeInCyclesAlap[&op].reserve(iterationCount);
    }

    for (int iteration = 0; iteration < iterationCount; ++iteration) {
      for (auto &op : region.getOps()) {
        auto distanceArray =
            op.getAttrOfType<mlir::ArrayAttr>("distances").getValue();

        bool isGamma = op.hasAttr("SpecHLS.gamma");
        int nextCycleAsap = isGamma ? std::numeric_limits<int>::max() : 0;
        double nextTimeInCycleAsap = 0.0;
        int nextCycleAlap = 0;
        double nextTimeInCycleAlap = 0.0;

        // Compute unrolled schedule
        for (unsigned predIndex = 0; predIndex < op.getNumOperands();
             ++predIndex) {
          int distance =
              llvm::cast<mlir::IntegerAttr>(distanceArray[predIndex]).getInt();
          if (iteration - distance < 0) {
            nextCycleAsap = 0;
            nextTimeInCycleAsap = 0.0;
          } else {
            int predEndCycleAsap = 0;
            double predEndTimeInCycleAsap = 0.0;
            int predEndCycleAlap = 0;
            double predEndTimeInCycleAlap = 0.0;

            mlir::Operation *pred = op.getOperand(predIndex).getDefiningOp();
            int predLatency =
                pred->getAttrOfType<mlir::IntegerAttr>("latency").getInt();
            double predInDelay = pred->getAttrOfType<mlir::FloatAttr>("inDelay")
                                     .getValueAsDouble();
            double predOutDelay =
                pred->getAttrOfType<mlir::FloatAttr>("outDelay")
                    .getValueAsDouble();

            auto computePredEnd =
                [=](int predStartCycle, double predStartTimeInCycle,
                    int &predEndCycle, double &predEndTimeInCycle) {
                  if (predLatency > 0) {
                    predEndCycle = predStartCycle + predLatency;
                    predEndTimeInCycle = predOutDelay;
                  } else if (predStartTimeInCycle + predInDelay + predOutDelay >
                             targetClock) {
                    predEndCycle = predStartCycle + 1;
                    predEndTimeInCycle = predOutDelay;
                  } else {
                    predEndCycle = predStartCycle;
                    predEndTimeInCycle = predStartTimeInCycle + predOutDelay;
                  }
                };

            int diff = iteration - distance;
            if (diff >= 0) {
              if (diff < static_cast<ptrdiff_t>(startTimesAsap[pred].size())) {
                computePredEnd(
                    startTimesAsap[pred][iteration - distance],
                    startTimeInCyclesAsap[pred][iteration - distance],
                    predEndCycleAsap, predEndTimeInCycleAsap);
              }
              if (diff < static_cast<ptrdiff_t>(startTimesAlap[pred].size())) {
                computePredEnd(
                    startTimesAlap[pred][iteration - distance],
                    startTimeInCyclesAlap[pred][iteration - distance],
                    predEndCycleAlap, predEndTimeInCycleAlap);
              }
            }

            // ASAP
            if (diff >= 0 &&
                diff < static_cast<ptrdiff_t>(startTimesAsap[pred].size())) {
              if (isGamma) {
                if (nextCycleAsap > predEndCycleAsap) {
                  nextCycleAsap = predEndCycleAsap;
                  nextTimeInCycleAsap = predEndTimeInCycleAsap;
                } else if (nextCycleAsap == predEndCycleAsap) {
                  nextTimeInCycleAsap =
                      std::min(nextTimeInCycleAsap, predEndTimeInCycleAsap);
                }
              } else if (predEndCycleAsap > nextCycleAsap) {
                nextCycleAsap = predEndCycleAsap;
                nextTimeInCycleAsap = predEndTimeInCycleAsap;
              } else if (predEndCycleAsap == nextCycleAsap) {
                nextTimeInCycleAsap =
                    std::max(nextTimeInCycleAsap, predEndTimeInCycleAsap);
              }
            } else {
              nextCycleAsap = 0;
              nextTimeInCycleAsap = 0.0;
            }
            // ALAP
            if (diff >= 0 &&
                diff < static_cast<ptrdiff_t>(startTimesAlap[pred].size())) {
              if (predEndCycleAlap > nextCycleAlap) {
                nextCycleAlap = predEndCycleAlap;
                nextTimeInCycleAlap = predEndTimeInCycleAlap;
              } else if (predEndCycleAlap == nextCycleAlap) {
                nextTimeInCycleAlap =
                    std::max(nextTimeInCycleAlap, predEndTimeInCycleAlap);
              }
            }
          }
        }
        startTimesAsap[&op].push_back(nextCycleAsap);
        startTimeInCyclesAsap[&op].push_back(nextTimeInCycleAsap);
        startTimesAlap[&op].push_back(nextCycleAlap);
        startTimeInCyclesAlap[&op].push_back(nextTimeInCycleAlap);
      }
    }

    // Compute Mobilities
    for (auto *g : gammas) {
      int mobility = 0;
      for (int iteration = sumDistances + 1; iteration < iterationCount;
           ++iteration) {
        int candidateMobility =
            (startTimesAlap[g][iteration] - startTimesAlap[g][iteration - 1]) -
            (startTimesAsap[g][iteration] - startTimesAsap[g][iteration - 1]);
        mobility = std::max(mobility, candidateMobility);
      }
      g->setAttr("SpecHLS.mobility",
                 mlir::IntegerAttr::get(
                     mlir::IntegerType::get(g->getContext(), 32), mobility));
    }
  }
};

std::unique_ptr<mlir::Pass> createMobilityPass() {
  return std::make_unique<MobilityPass>();
}

} // namespace SpecHLS
