#include "Scheduling/Transforms/SchedulePass.h"
#include "Scheduling/Algorithms.h"
#include "Scheduling/Problems.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/SSP/SSPOps.h"
#include "circt/Dialect/SSP/Utilities.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace circt;
using namespace ssp;

namespace circt {
namespace ssp {
template <>
struct Default<GammaMobilityProblem> {
  static constexpr auto operationProperties =
      Default<scheduling::ChainingCyclicProblem>::operationProperties;
  static constexpr auto operatorTypeProperties =
      Default<scheduling::ChainingCyclicProblem>::operatorTypeProperties;
  static constexpr auto dependenceProperties =
      Default<scheduling::ChainingCyclicProblem>::dependenceProperties;
  static constexpr auto instanceProperties =
      Default<scheduling::ChainingCyclicProblem>::instanceProperties;
};
} // namespace ssp
} // namespace circt

namespace SpecHLS {

static std::optional<float> getCycleTime(StringRef options) {
  for (StringRef option : llvm::split(options, ',')) {
    if (option.consume_front("cycle-time="))
      return std::stof(option.str());
  }
  return std::nullopt;
}

struct GecosSchedulePass
    : public impl::GecosSchedulePassBase<GecosSchedulePass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();

    SmallVector<InstanceOp> instanceOps;
    OpBuilder builder(&getContext());
    for (auto instOp : moduleOp.getOps<InstanceOp>()) {
      builder.setInsertionPoint(instOp);

      auto problemName = instOp.getProblemName();
      if (!problemName.equals("GammaMobilityProblem")) {
        llvm::errs() << "gecos-schedule: Unsupported problem '" << problemName
                     << "'\n";
        return;
      }
      auto cycleTime = getCycleTime(schedulerOptions.getValue());
      if (!cycleTime) {
        llvm::errs() << "gecos-schedule: Missing option 'cycle-time'\n";
        return;
      }
      auto prob = loadProblem<GammaMobilityProblem>(instOp);
      InstanceOp scheduledOp;
      if (failed(prob.check()) || failed(scheduleASAP(prob, *cycleTime)) ||
          failed(prob.verify()))
        scheduledOp = {};
      else
        scheduledOp = saveProblem(prob, builder);
      if (!scheduledOp)
        return signalPassFailure();
      instanceOps.push_back(instOp);
    }

    llvm::for_each(instanceOps, [](InstanceOp op) { op.erase(); });
  }
};

std::unique_ptr<mlir::Pass> createGecosSchedulePass() {
  return std::make_unique<GecosSchedulePass>();
}

}; // namespace SpecHLS