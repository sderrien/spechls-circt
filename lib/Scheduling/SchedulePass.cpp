#include "Scheduling/Transforms/SchedulePass.h"
#include "Scheduling/Algorithms.h"
#include "Scheduling/Problems.h"
#include "Scheduling/SchedulingProperty.h"
#include "Scheduling/Transforms/Passes.h"
#include "circt/Dialect/SSP/SSPOps.h"
#include "circt/Dialect/SSP/Utilities.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace circt;
using namespace ssp;
template <>
struct circt::ssp::Default<SpecHLS::GammaMobilityProblem> {
  static constexpr auto operationProperties = std::tuple_cat(
      circt::ssp::Default<
          scheduling::ChainingCyclicProblem>::operationProperties,
      std::make_tuple(SpecHLS::MinPositionAttr()));
  static constexpr auto operatorTypeProperties = circt::ssp::Default<
      scheduling::ChainingCyclicProblem>::operatorTypeProperties;
  static constexpr auto dependenceProperties = circt::ssp::Default<
      scheduling::ChainingCyclicProblem>::dependenceProperties;
  static constexpr auto instanceProperties = circt::ssp::Default<
      scheduling::ChainingCyclicProblem>::instanceProperties;
};

namespace SpecHLS {

namespace {

std::optional<float> getCycleTime(StringRef options) {
  for (StringRef option : llvm::split(options, ',')) {
    if (option.consume_front("cycle-time="))
      return std::stof(option.str());
  }
  return std::nullopt;
}

} // namespace

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