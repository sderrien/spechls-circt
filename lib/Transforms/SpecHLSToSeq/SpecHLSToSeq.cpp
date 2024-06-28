//===- SpecHLSToSeq.cpp
//----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Transforms/Passes.h"
#include "Transforms/SpecHLSConversion.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

//===----------------------------------------------------------------------===//
// Convert Comb to Arith pass
//===----------------------------------------------------------------------===//

namespace {

// CRTP pattern
struct ConvertSpecHLSToSeqPass
    : public SpecHLS::impl::SpecHLSToSeqBase<ConvertSpecHLSToSeqPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertSpecHLSToSeqPass::runOnOperation() {

  auto hwmodule = getOperation();

  Value *clock = NULL;
  Value *reset = NULL;

  for (auto arg : hwmodule.getBody().getArguments()) {
    if (arg.getType().isa<circt::seq::ClockType>()) {
      clock = &arg; // assume single clock signal
    }
  }

  OpBuilder builder(hwmodule->getContext());
  if (!clock) {
    auto clkType = builder.getType<circt::seq::ClockType>();

    hwmodule.insertInput(0,"en",builder.getI1Type());
    hwmodule.insertInput(0,"clk",clkType);

    clock = &hwmodule.getBody().getArguments()[0]; // assume single clock signal
    reset = &hwmodule.getBody().getArguments()[1]; // assume single clock signal
  }

  DelayOpToShiftRegOpConversion pat(&getContext(),clock,reset);

  llvm::DenseMap<MuOp, circt::seq::HLMemOp> memMap;

  mlir::RewritePatternSet patterns(&getContext());
  MuOpToRegConversion m2r(&getContext(), clock, reset, memMap);
  patterns.add(std::make_unique<MuOpToRegConversion>(m2r));
  if (failed(applyPatternsAndFoldGreedily(hwmodule, std::move(patterns)))) {
    signalPassFailure();
  }
  llvm::errs() << "There\n";


  mlir::RewritePatternSet patterns2(&getContext());


  DelayOpToShiftRegOpConversion d2s(&getContext(), clock, reset);
  AlphaOpToHLWriteConversion a2w(&getContext(), clock, reset, memMap);
  ArrayReadOpToHLReadConversion a2r(&getContext(), clock, reset, memMap);
  llvm::errs() << "Here\n";
//
//
  patterns2.add(std::make_unique<DelayOpToShiftRegOpConversion>(d2s));
  patterns2.add(std::make_unique<AlphaOpToHLWriteConversion>(a2w));
  patterns2.add(std::make_unique<ArrayReadOpToHLReadConversion>(a2r));
  llvm::errs() << "Again\n";
  if (failed(applyPatternsAndFoldGreedily(hwmodule, std::move(patterns2)))) {

    signalPassFailure();
  }
  llvm::errs() << "done\n";
}

namespace SpecHLS {

std::unique_ptr<OperationPass<circt::hw::HWModuleOp>>
createConvertSpecHLSToSeqPass() {
  return std::make_unique<ConvertSpecHLSToSeqPass>();
}

void registerConvertSpecHLSToSeqPass() {
  PassRegistration<ConvertSpecHLSToSeqPass>();
}
} // namespace SpecHLS