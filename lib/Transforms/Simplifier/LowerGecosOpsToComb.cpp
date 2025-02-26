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
#include "mlir/IR/Verifier.h"

//===----------------------------------------------------------------------===//
// Convert Comb to Arith pass
//===----------------------------------------------------------------------===//

namespace SpecHLS {


struct GecosCustomOpConversion : OpRewritePattern<GecosOp> {
  using OpRewritePattern<GecosOp>::OpRewritePattern;

  LogicalResult insertBoolCast(GecosOp op, PatternRewriter &rewriter) const {
    for (int k =0;k<op->getNumOperands();k++) {
      auto input = op.getOperand(k);
      auto inputType= input.getType();
      if (inputType.getIntOrFloatBitWidth()!=1) {
        auto cast = rewriter.create<SpecHLS::CastOp>(op.getLoc(),rewriter.getIntegerType(1),input);
        op.setOperand(k,cast);
      }
    }
    auto resultType= op.getResult().getType();
    if (resultType.isUnsignedInteger()) {
      auto cast = rewriter.create<SpecHLS::CastOp>(op.getLoc(),rewriter.getIntegerType(resultType.getIntOrFloatBitWidth()),op.getResult());
      op.getResult().replaceAllUsesExcept(cast,cast);
    }
    return success();
  }

  LogicalResult insertIntegerCast(GecosOp op, PatternRewriter &rewriter) const {
    for (int k =0;k<op->getNumOperands();k++) {
      auto input = op.getOperand(k);
      auto inputType= input.getType();
      if (inputType.isUnsignedInteger()) {
        auto cast = rewriter.create<SpecHLS::CastOp>(op.getLoc(),rewriter.getIntegerType(inputType.getIntOrFloatBitWidth()),input);
        op.setOperand(k,cast);
      }
    }
    auto resultType= op.getResult().getType();
    if (resultType.isUnsignedInteger()) {
      auto cast = rewriter.create<SpecHLS::CastOp>(op.getLoc(),rewriter.getIntegerType(resultType.getIntOrFloatBitWidth()),op.getResult());
      op.getResult().replaceAllUsesExcept(cast,cast);
    }
    return success();
  }

  LogicalResult matchAndRewrite(GecosOp op, PatternRewriter &rewriter) const {

    if (op.getNameAttr()=="add") {
      assert(op->getNumOperands()>1);

      insertIntegerCast(op, rewriter);
      auto resType =rewriter.getIntegerType(op.getResult().getType().getIntOrFloatBitWidth());
      auto orop = rewriter.create<comb::AddOp>(
          op.getLoc(),resType,
          op.getOperands());
      rewriter.replaceOp(op, orop);
      return success();

    } else if (op.getNameAttr()=="lor") {
      assert(op->getNumOperands()>1);
      insertIntegerCast(op,rewriter);
      auto resType =rewriter.getIntegerType(op.getResult().getType().getIntOrFloatBitWidth());
      auto orop = rewriter.create<comb::OrOp>(op.getLoc(),resType,op.getOperands());
      rewriter.replaceOp(op,orop);
      return success();

    } else if (op.getNameAttr()=="lnot") {
      assert(op->getNumOperands()==1);
      insertIntegerCast(op,rewriter);
      auto resType =rewriter.getIntegerType(op.getResult().getType().getIntOrFloatBitWidth());
      auto zero = rewriter.create<hw::ConstantOp>(op.getLoc(),op.getOperand(0).getType(),0);
      auto neop = rewriter.create<comb::ICmpOp>(op.getLoc(),ICmpPredicate::eq,op.getOperand(0),zero);
      rewriter.replaceOp(op,neop);
      return success();

      auto xorop = rewriter.create<comb::XorOp>(op.getLoc(),op.getOperand(0),op.getOperand(0));
      mlir::verify(xorop);
      rewriter.replaceOp(op,xorop);
      return success();

    } else if (op.getNameAttr()=="ne") {
      assert(op->getNumOperands()==2);
      insertIntegerCast(op,rewriter);
      auto resType =rewriter.getIntegerType(op.getResult().getType().getIntOrFloatBitWidth());
      auto icmpOp = rewriter.create<comb::ICmpOp>(op.getLoc(),ICmpPredicate::ne,op.getOperand(0),op.getOperand(1));
      rewriter.replaceOp(op,icmpOp);
      return success();

    } else if (op.getNameAttr()=="eq") {
      assert(op->getNumOperands()==2);
      insertIntegerCast(op,rewriter);
      auto resType =rewriter.getIntegerType(op.getResult().getType().getIntOrFloatBitWidth());
      auto neop = rewriter.create<comb::ICmpOp>(op.getLoc(),ICmpPredicate::eq,op.getOperand(0),op.getOperand(1));
      rewriter.replaceOp(op,neop);
      return success();

    } else if (op.getNameAttr()=="lt") {
      assert(op->getNumOperands()==2);
      auto unsignedLHS = op.getOperand(0).getType().isUnsignedInteger();
      auto unsignedRHS = op.getOperand(1).getType().isUnsignedInteger();
      auto predicate = (unsignedLHS && unsignedRHS) ? ICmpPredicate::ult :ICmpPredicate::slt;
      insertIntegerCast(op,rewriter);
      auto resType =rewriter.getIntegerType(op.getResult().getType().getIntOrFloatBitWidth());
      auto icmpOp = rewriter.create<comb::ICmpOp>(op.getLoc(),predicate,op.getOperand(0),op.getOperand(1));
      rewriter.replaceOp(op,icmpOp);
      return success();

    } else if (op.getNameAttr()=="le") {
      assert(op->getNumOperands()==2);
      auto unsignedLHS = op.getOperand(0).getType().isUnsignedInteger();
      auto unsignedRHS = op.getOperand(1).getType().isUnsignedInteger();
      auto predicate = (unsignedLHS && unsignedRHS) ? ICmpPredicate::ule :ICmpPredicate::sle;
      insertIntegerCast(op,rewriter);
      auto resType =rewriter.getIntegerType(op.getResult().getType().getIntOrFloatBitWidth());
      auto icmpOp = rewriter.create<comb::ICmpOp>(op.getLoc(),predicate,op.getOperand(0),op.getOperand(1));
      rewriter.replaceOp(op,icmpOp);
      return success();
    } else if (op.getNameAttr()=="ge") {
      assert(op->getNumOperands()==2);
      auto unsignedLHS = op.getOperand(0).getType().isUnsignedInteger();
      auto unsignedRHS = op.getOperand(1).getType().isUnsignedInteger();
      auto predicate = (unsignedLHS && unsignedRHS) ? ICmpPredicate::uge :ICmpPredicate::sge;
      insertIntegerCast(op,rewriter);
      auto resType =rewriter.getIntegerType(op.getResult().getType().getIntOrFloatBitWidth());
      auto icmpOp = rewriter.create<comb::ICmpOp>(op.getLoc(),predicate,op.getOperand(0),op.getOperand(1));
      rewriter.replaceOp(op,icmpOp);
      return success();
  } else if (op.getNameAttr()=="gt") {
    assert(op->getNumOperands()==2);
      auto unsignedLHS = op.getOperand(0).getType().isUnsignedInteger();
      auto unsignedRHS = op.getOperand(1).getType().isUnsignedInteger();
      auto predicate = (unsignedLHS && unsignedRHS) ? ICmpPredicate::ugt :ICmpPredicate::sgt;
      insertIntegerCast(op,rewriter);
      auto resType =rewriter.getIntegerType(op.getResult().getType().getIntOrFloatBitWidth());
      auto icmpOp = rewriter.create<comb::ICmpOp>(op.getLoc(),predicate,op.getOperand(0),op.getOperand(1));
      rewriter.replaceOp(op,icmpOp);
      return success();
    } else if (op.getNameAttr()=="and") {
      assert(op->getNumOperands()>1);
      insertIntegerCast(op,rewriter);
      auto andop = rewriter.create<comb::AndOp>(op.getLoc(),rewriter.getIntegerType(op.getResult().getType().getIntOrFloatBitWidth()),op.getOperands());
      mlir::verify(andop);
      rewriter.replaceOp(op,andop);
      return success();

    } else if (op.getNameAttr()=="or") {
      assert(op->getNumOperands()>1);
      insertIntegerCast(op,rewriter);
      auto orop = rewriter.create<comb::OrOp>(op.getLoc(),rewriter.getIntegerType(op.getResult().getType().getIntOrFloatBitWidth()),op.getOperands());
      rewriter.replaceOp(op,orop);
      return success();

    } else if (op.getNameAttr()=="land") {
      assert(op->getNumOperands()>1);
      insertIntegerCast(op,rewriter);
      auto andop = rewriter.create<comb::AndOp>(op.getLoc(),rewriter.getIntegerType(op.getResult().getType().getIntOrFloatBitWidth()),op.getOperands());
      mlir::verify(andop);
      rewriter.replaceOp(op,andop);
      return success();
    }
    return failure();
  };
};

// CRTP pattern
struct LowerGecosOpsToCombPass : public SpecHLS::impl::LowerGecosOpsToCombPassBase<LowerGecosOpsToCombPass> {

  void runOnOperation() {

    auto op = getOperation();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.insert<GecosCustomOpConversion>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createLowerGecosOpsToCombPass() {
  return std::make_unique<LowerGecosOpsToCombPass>();
}

} // namespace SpecHLS
