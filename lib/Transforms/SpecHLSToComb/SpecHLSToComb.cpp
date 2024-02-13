//===- SpecHLSToComb.cpp
//----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SpecHLS/SpecHLSDialect.h"
#include "SpecHLS/SpecHLSOps.h"
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
struct ConvertSpecHLSToCombPass : public SpecHLS::impl::SpecHLSToCombBase<ConvertSpecHLSToCombPass> {
  void runOnOperation() override;
  //  virtual StringRef getName() ;
  //  virtual std::unique_ptr<Pass> clonePass() ;
};
} // namespace

//void populateSpecHLSToCombConversionPatterns(
//    TypeConverter &converter, mlir::RewritePatternSet &patterns) {
// // patterns.add<GammaToMuxOpConversion>( converter, patterns.getContext());
//  patterns.add<LookUpTableToTruthTableOpConversion>( converter, patterns.getContext());
//}

void ConvertSpecHLSToCombPass::runOnOperation() {

  auto op = getOperation();

  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<LookUpTableToTruthTableOpConversion>(&getContext());
  patterns.insert<GammaToMuxOpConversion>(&getContext());

  if (failed( applyPatternsAndFoldGreedily(op,std::move(patterns)))) {
    signalPassFailure();
  }

}

namespace SpecHLS {

std::unique_ptr<OperationPass<circt::hw::HWModuleOp>> createConvertSpecHLSToCombPass() {
  return std::make_unique<ConvertSpecHLSToCombPass>();
}

void registerConvertSpecHLSToCombPass() {
  PassRegistration<ConvertSpecHLSToCombPass>();
}
} // namespace SpecHLS
