//
// Created by Steven on 30/12/2023.
//

#include "mlir/Pass/Pass.h"

#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace circt;
using namespace hw;
using namespace comb;
using namespace mlir;
using namespace SpecHLS;

#define VERBOSE false
//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace SpecHLS {}
namespace {



struct CastWLBackPropagation : OpRewritePattern<CastOp> {

  using OpRewritePattern<CastOp>::OpRewritePattern;

  bool isValidOperation(Operation *op) const {
    if (op->getNumOperands()==2 && op->getNumResults()==1) {
      return TypeSwitch<Operation *, bool>(op)
          .Case<circt::comb::AddOp>([&](auto op) { return true; })
          .Case<circt::comb::SubOp>([&](auto op) { return true; })
          .Case<circt::comb::AndOp>([&](auto op) { return true; })
          .Case<circt::comb::OrOp>([&](auto op) { return true; })
          .Case<circt::comb::XorOp>([&](auto op) { return true; })
          .Case<circt::comb::ExtractOp>([&](auto op) { return true; });
//          .Case<circt::comb::ConcatOp>([&](auto op) { return true; })
//          .Case<circt::comb::ShlOp>([&](auto op) { return true; })
//          .Case<circt::comb::ShrSOp>([&](auto op) { return true; })
//          .Case<circt::comb::ShrUOp>([&](auto op) { return true; });
    }
    return false;
  }

  // This function propaates CastOp backward when the input type has more bits than the output type.
  // If the operand's defining operation (defOp) is valid, it moves the cast to each operand
  // of defOp, replacing the original CastOp and reducing redundancy in the IR.
  //
  // Example : when casting a 32-bit integer down to a 16-bit integer after some arithmetic operations,
  // instead of casting the final result, it might be more efficient to cast the inputs to
  // 16-bit integers before performing the arithmetic, thus simplifying the subsequent operations.
  LogicalResult matchAndRewrite(CastOp op, PatternRewriter &rewriter) const  {
    auto outType = dyn_cast<IntegerType>(op.getResult().getType());
    auto inType = dyn_cast<IntegerType>(op.getOperand().getType());
    if (inType && outType) {

      uint32_t inWL  = inType.getWidth();
      uint32_t outWL  = outType.getWidth();

      if (inWL > outWL && inType.isSignless()) {
        if (auto defOp = op.getOperand().getDefiningOp()) {
          if (isValidOperation(defOp)) {
            if (VERBOSE) llvm::errs() << "Simplify  " << defOp << " and " << op << "\n";
            // Modify the operands of defOp one by one
            auto newOutType =rewriter.getIntegerType(outWL);
            for (unsigned i = 0, e = defOp->getNumOperands(); i < e; ++i) {
              auto operand = defOp->getOperand(i);
              // Create a CastOp for each operand that needs to be cast
              auto newCast = rewriter.create<CastOp>(op.getLoc(), newOutType, operand);
              // Update defOp's operand with the new casted value
              defOp->setOperand(i, newCast.getResult());
            }
            defOp->getResult(0).setType(newOutType);
            op->getOperand(0).setType(newOutType);

            // Replace the original CastOp with the result of defOp
            mlir::verify(op);
            return success();
          }
        }
      }
    }

    return failure();
  }

};

struct ConstantICmpOpWLReduction : OpRewritePattern<ICmpOp> {

  using OpRewritePattern<ICmpOp>::OpRewritePattern;

  bool isValidOperation(Operation *op) const {
    if (op->getNumOperands()==2 && op->getNumResults()==1) {
      return TypeSwitch<Operation *, bool>(op)
          .Case<circt::comb::AddOp>([&](auto op) { return true; })
          .Case<circt::comb::SubOp>([&](auto op) { return true; })
          .Case<circt::comb::AndOp>([&](auto op) { return true; })
          .Case<circt::comb::OrOp>([&](auto op) { return true; })
          .Case<circt::comb::XorOp>([&](auto op) { return true; })
          .Case<circt::comb::ICmpOp>([&](auto op) { return true; });
    }
    return false;
  }

  bool transform(Operation* op, ConstantOp* imm, SpecHLS::CastOp* cast, PatternRewriter &rewriter) const {
    if (VERBOSE) llvm::errs() << "narrowing " << *op << "with "<< *imm << " and " << *cast << "\n";
    auto narrowedType = dyn_cast<circt::hw::TypeVariant<mlir::IntegerType, circt::hw::IntType>>(cast->getOperand().getType());

    auto wideType = op->getResult(0).getType();

    cast->getResult().setType(narrowedType);

    imm->getResult().setType(narrowedType);

    op->getResult(0).setType(narrowedType);
    op->getOperand(0).setType(narrowedType);
    op->getOperand(1).setType(narrowedType);
    auto loc = op->getLoc();

    auto outputCast = rewriter.create<SpecHLS::CastOp>(loc,wideType,op->getResult(0));
    op->getResult(0).replaceAllUsesWith(outputCast.getResult());
    mlir::verify(outputCast);
    return true;
  }
  
  LogicalResult matchAndRewrite(ICmpOp op, PatternRewriter &rewriter) const override {

    if (VERBOSE) llvm::errs() << "matchAndRewrite t-" << op <<  "\n" ;
    if (op.getPredicate()==ICmpPredicate::eq) {
      auto lhsOp = op->getOperand(0).getDefiningOp();
      auto rhsOp = op->getOperand(1).getDefiningOp();
      if (lhsOp && rhsOp) {
        auto cast = dyn_cast<SpecHLS::CastOp>(lhsOp);
        auto cst = dyn_cast<ConstantOp>(rhsOp);
        if (cast && cst) {
        auto inputIntType = dyn_cast<IntegerType>(cast.getOperand().getType());
        auto outputIntType = dyn_cast<IntegerType>(cast.getResult().getType());
        if (inputIntType && outputIntType ) {
          auto inWL = inputIntType.getWidth();
          auto outWL = outputIntType.getWidth();
          if (outWL > inWL) {
            if (inputIntType.isSignless()) {
              if (VERBOSE) llvm::errs() << "Found pattern with :\n\t-" << op << "\n\t-"
                           << op->getOperand(0) << "\n\t-" << op->getOperand(1)
                           << "\n";
              auto newValue = cst.getValue().getSExtValue();
              auto newType = rewriter.getIntegerType(inWL);
              auto newConstantOp =
                  rewriter.create<ConstantOp>(cst->getLoc(), newType, newValue);
              op.setOperand(0, newConstantOp);
              op.setOperand(1, cast.getOperand());
              if (VERBOSE) llvm::errs() << "Replaced pattern with :\n\t-" << op << "\n\t-"
                           << op->getOperand(0) << "\n\t-" << op->getOperand(1)
                           << "\n";
              mlir::verify(op);

              return success();
            } else if (inputIntType.isUnsigned()) {
              if (VERBOSE) llvm::errs() << "Found pattern with :\n\t-" << op << "\n\t-"
                           << op->getOperand(0) << "\n\t-" << op->getOperand(1)
                           << "\n";
              auto newValue = cst.getValue().getZExtValue();
              auto newType = rewriter.getIntegerType(inWL);

              op.setOperand(0, rewriter.create<ConstantOp>(
                                   cst->getLoc(), inputIntType,
                                   cst.getValue().getSExtValue()));
              op.setOperand(1, rewriter.create<CastOp>(cst->getLoc(), newType,
                                                       cast->getOperand(0)));

              if (VERBOSE) llvm::errs() << "Replaced pattern with :\n\t-" << op << "\n\t-"
                           << op->getOperand(0) << "\n\t-" << op->getOperand(1)
                           << "\n";
              mlir::verify(op);

              return success();
            }
          }
        }
        }
      }
    }
    if (VERBOSE) llvm::errs() << "no match for " << op <<  "\n" ;
    return failure();
  }

};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Comb to Arith pass
//===----------------------------------------------------------------------===//

namespace {

// CRTP pattern
struct WordLengthPropagationPass : public SpecHLS::impl::WordLengthPropagationPassBase<WordLengthPropagationPass> {
  void runOnOperation() override;
  //  virtual StringRef getName() ;
  //  virtual std::unique_ptr<Pass> clonePass() ;
  };
} // namespace


void WordLengthPropagationPass::runOnOperation() {

  auto *ctx = &getContext();

  RewritePatternSet patterns(ctx);
    patterns.insert<ConstantICmpOpWLReduction>(ctx);
//  patterns.insert<CastWLBackPropagation>(ctx);

  if (failed(applyPatternsAndFoldGreedily(getOperation(),std::move(patterns)))) {
    if (VERBOSE) llvm::errs() << "rewrite failed  \n";
    signalPassFailure();
  }
  mlir::verify(getOperation(), true);

}

namespace SpecHLS {

  std::unique_ptr<OperationPass<ModuleOp>> createWordLengthPropagationPass() {
    return std::make_unique<WordLengthPropagationPass>();
  }
} // namespace SpecHLS
