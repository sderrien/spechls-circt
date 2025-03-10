#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

using namespace circt;
using namespace mlir;

struct OperationData {
  Operation *op;
  float asapCycle;
  float alapCycle;
  float mobility;
  float scheduledCycle;
  SmallVector<Operation*, 4> predecessors;
  SmallVector<Operation*, 4> successors;
};

void createFSM(ModuleOp module, SmallVector<OperationData, 16> &schedule) {
  // Create an OpBuilder in the context of the provided module
  OpBuilder builder(module.getBodyRegion());

  // Create the FSM operation

  //  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef name, StringRef initialState, FunctionType function_type, ArrayRef<NamedAttribute> attrs = {}, ArrayRef<DictionaryAttr> argAttrs = {});
  auto fsmOp = builder.create<fsm::MachineOp>(module.getLoc(), "ScheduledFSM", "idle",builder.getFunctionType({},{}));

  // The entry state of the FSM
  auto entryState = builder.create<fsm::StateOp>(fsmOp.getLoc(), "EntryState");

  // Map to hold state operations for each cycle
  std::map<float, fsm::StateOp> stateMap;

  // Define the states based on scheduled cycles
  for (const auto &opData : schedule) {
    if (stateMap.find(opData.scheduledCycle) == stateMap.end()) {
      std::string stateName = "State" + std::to_string(static_cast<int>(opData.scheduledCycle));
      auto state = builder.create<fsm::StateOp>(fsmOp.getLoc(), stateName);
      stateMap[opData.scheduledCycle] = state;
    }
  }

  // Set the initial state transition from EntryState to the first scheduled state
  auto firstState = stateMap.begin()->second;
  builder.create<fsm::TransitionOp>(entryState.getLoc(), entryState, firstState);

  // Create transitions between the states based on the schedule
  fsm::StateOp prevState = firstState;
  for (auto it = stateMap.begin(); it != stateMap.end(); ++it) {
    if (it == stateMap.begin()) continue; // Skip the first state as it's already connected

    auto currentState = it->second;
    builder.create<fsm::TransitionOp>(prevState.getLoc(), prevState, currentState);
    prevState = currentState;
  }

  // Final transition to a halt state or loop to the first state if required
  auto haltState = builder.create<fsm::StateOp>(fsmOp.getLoc(), "HaltState");
  builder.create<fsm::TransitionOp>(prevState.getLoc(), prevState, haltState);

  // Create output logic for each state
  for (const auto &opData : schedule) {
    auto state = stateMap[opData.scheduledCycle];
    // Use the builder to insert operations or actions within the state
    builder.setInsertionPointToEnd(state->getBlock());

    // Example: Placeholder for actual operation handling
    builder.create<SomeOperationType>(state.getLoc(), /*operation-specific arguments*/);
  }
}

int main() {
  // Set up the MLIR context and module
  MLIRContext context;
  context.loadDialect<fsm::FSMDialect, hw::HWDialect>();
  ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));

  // Example schedule to pass to the FSM creation function
  SmallVector<OperationData, 16> schedule = {
      // Fill with example OperationData
  };

  // Call the function to create the FSM based on the schedule
  createFSM(module, schedule);

  // Output the IR (for demonstration purposes)
  module.print(llvm::outs());
  return 0;
}
