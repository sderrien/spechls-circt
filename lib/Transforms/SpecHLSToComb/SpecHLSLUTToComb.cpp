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

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Convert Comb to Arith pass
//===----------------------------------------------------------------------===//

namespace {

// CRTP pattern
struct ConvertSpecHLSLUTToCombPass
    : public SpecHLS::impl::SpecHLSLUTToCombBase<ConvertSpecHLSLUTToCombPass> {
  void runOnOperation() override;
  //  virtual StringRef getName() ;
  //  virtual std::unique_ptr<Pass> clonePass() ;
};
} // namespace

void populateSpecHLSLUTToCombConversionPatterns(
    TypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<LookUpTableToTruthTableOpConversion>(patterns.getContext());
}

void ConvertSpecHLSLUTToCombPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<SpecHLSDialect>();
  target.addLegalDialect<CombDialect>();
  target.addIllegalOp<SpecHLS::GammaOp>();
  target.addIllegalOp<SpecHLS::LookUpTableOp>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;

  converter.addConversion([](Type type) { return type; });

  populateSpecHLSLUTToCombConversionPatterns(converter, patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    signalPassFailure();
}

namespace SpecHLS {

std::unique_ptr<OperationPass<SpecHLS::LookUpTableOp>>
createConvertSpecHLSLUTToCombPass() {
  return std::make_unique<ConvertSpecHLSLUTToCombPass>();
}

void registerConvertSpecHLSLUTToCombPass() {
  PassRegistration<ConvertSpecHLSLUTToCombPass>();
}
} // namespace SpecHLS