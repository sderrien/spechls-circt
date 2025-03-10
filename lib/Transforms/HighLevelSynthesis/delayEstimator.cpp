#include "mlir/IR/Operation.h"
#include "circt/Dialect/Comb/CombOps.h"

using namespace circt;
using namespace mlir;

float getCombDelay(Operation *op) {
  // Constants for delays
  const float logic_delay = 1.0;           // Constant delay for logic operations
  const float arithmetic_base_delay = 1.0; // Base delay for arithmetic operations
  const float multiplier_delay = 2.0;      // Delay for multiplication
  const float shift_delay = 1.5;           // Delay for shift operations
  const float constant_delay = 0.5;        // Delay for constant operations
  const float small_delay = 0.2;           // Delay for simple operations like extract, concat

  // Determine the bitwidth from the operation's result type, defaulting to 1 if unknown
  int bitwidth = 1;
  if (auto resultType = op->getResult(0).getType().dyn_cast<IntegerType>()) {
    bitwidth = resultType.getWidth();
  }

  // Use TypeSwitch to handle different operation types and assign delays
  float delay = TypeSwitch<Operation*, float>(op)
                    // Handle logic operations with a constant delay
                    .Case<comb::AndOp, comb::OrOp, comb::XorOp, comb::NotOp>([&](Operation *op) {
                      return logic_delay;
                    })
                    // Handle arithmetic operations (add, sub) with delays depending on bitwidth
                    .Case<comb::AddOp, comb::SubOp>([&](Operation *op) {
                      return arithmetic_base_delay + 0.1 * bitwidth;
                    })
                    // Handle multiplication operations with a higher base delay and additional bitwidth scaling
                    .Case<comb::MulOp>([&](Operation *op) {
                      return multiplier_delay + 0.2 * bitwidth;
                    })
                    // Handle shift operations (shl, shr) with specific delays
                    .Case<comb::ShlOp, comb::ShrOp>([&](Operation *op) {
                      return shift_delay + 0.15 * bitwidth;
                    })
                    // Handle simple operations like extract and concat with a minimal delay
                    .Case<comb::ExtractOp, comb::ConcatOp>([&](Operation *op) {
                      return small_delay;
                    })
                    // Handle constant operations with a fixed minimal delay
                    .Case<comb::ConstantOp>([&](Operation *op) {
                      return constant_delay;
                    })
                    // Default case for unknown or unsupported operations
                    .Default([&](Operation *op) {
                      // Default delay for any other operations not explicitly handled above
                      return logic_delay;
                    });

  return delay;
}
