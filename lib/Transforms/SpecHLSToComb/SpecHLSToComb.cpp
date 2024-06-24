//===- SpecHLSToComb.cpp
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
struct ConvertSpecHLSToCombPass
    : public SpecHLS::impl::SpecHLSToCombBase<ConvertSpecHLSToCombPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertSpecHLSToCombPass::runOnOperation() {

  auto op = getOperation();

  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<LookUpTableToTruthTableOpConversion>(&getContext());
  patterns.insert<GammaToMuxOpConversion>(&getContext());
  patterns.insert<RollbackToCombConversion>(&getContext());

  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
    signalPassFailure();
  }
}

namespace SpecHLS {

std::unique_ptr<OperationPass<circt::hw::HWModuleOp>>
createConvertSpecHLSToCombPass() {
  return std::make_unique<ConvertSpecHLSToCombPass>();
}

void registerConvertSpecHLSToCombPass() {
  PassRegistration<ConvertSpecHLSToCombPass>();
}
} // namespace SpecHLS
