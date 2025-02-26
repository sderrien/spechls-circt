//===- MergeGammas.cpp - Arith-to-comb mapping pass ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the MergeGammas pass.
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace circt;
using namespace SpecHLS;

namespace SpecHLS {

mlir::LogicalResult mergeGammaPair(Builder& builder, GammaOp &outerGamma, GammaOp &innerGamma) {

}
// requires at least C++11
const std::string vformat(const char * const zcFormat, ...) {

  // initialize use of the variable argument array
  va_list vaArgs;
  va_start(vaArgs, zcFormat);

  // reliably acquire the size
  // from a copy of the variable argument array
  // and a functionally reliable call to mock the formatting
  va_list vaArgsCopy;
  va_copy(vaArgsCopy, vaArgs);
  const int iLen = std::vsnprintf(NULL, 0, zcFormat, vaArgsCopy);
  va_end(vaArgsCopy);

  // return a formatted string without risking memory mismanagement
  // and without assuming any compiler or platform specific behavior
  std::vector<char> zc(iLen + 1);
  std::vsnprintf(zc.data(), zc.size(), zcFormat, vaArgs);
  va_end(vaArgs);
  return std::string(zc.data(), iLen); }

struct GammaMergingPattern : OpRewritePattern<GammaOp> {
  bool verbose = false;
  using OpRewritePattern<GammaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GammaOp outerGamma,
                                PatternRewriter &rewriter) const override {

    auto nbOuterInputs = outerGamma.getInputs().size();
    for (int innerIndex = 0; innerIndex < outerGamma.getInputs().size();
         innerIndex++) {
      auto input = outerGamma.getInputs()[innerIndex].getDefiningOp();
      if (input != NULL) {
        auto innerGamma = dyn_cast<SpecHLS::GammaOp>(input);
        if (innerGamma) {

          if (verbose) llvm::errs() << "Attempting to merge :\n" ;
          if (verbose) llvm::errs() << "  - Inner -> "<< innerGamma << "\n";
          if (verbose) llvm::errs() << "  - Outer -> "<< outerGamma << "\n";

          auto users = innerGamma->getUsers();
          auto nbUsers = std::distance(users.begin(), users.end());

          auto nbInnerInputs = innerGamma.getInputs().size();
          int newDepth = int(ceil(log(nbOuterInputs + nbInnerInputs) / log(2)));
          //  Value* muxOperands = new Value[cwidth * (cwidth+1)];
          Operation::operand_range in = innerGamma.getInputs();
          SmallVector<Value, 8> muxOperands;

          auto outerGammaSelType = outerGamma.getSelect().getType();
          auto innerGammaSelType = innerGamma.getSelect().getType();

          int innerSelectBW = innerGammaSelType.getIntOrFloatBitWidth();
          int outerSelectBW = outerGammaSelType.getIntOrFloatBitWidth();

          auto outerNBinput = outerGamma.getInputs().size() ;
          auto innerNBinput = innerGamma.getInputs().size() ;
          auto mergeNBinput = outerNBinput + innerNBinput-1;

          int innerPow2Inputs = 1 << innerSelectBW;
          int outerPow2Inputs = 1 << outerSelectBW;

          /* The pass first collects both inner and outer gamma operands and store
           * them into the muxOperands vector*/
          int innerPortMap[innerPow2Inputs];
          int outerPortMap[outerPow2Inputs];
          int dontCareOffset  = innerNBinput + outerNBinput -1;
          int offset =0;

          for (int pos = 0; pos < innerIndex; pos++) {
              muxOperands.push_back(outerGamma.getInputs()[pos]);
              outerPortMap[pos]= pos;
          }
          outerPortMap[innerIndex]= dontCareOffset;


          for (int pos = 0; pos < nbInnerInputs; pos++) {
            muxOperands.push_back(innerGamma.getInputs()[pos]);
            innerPortMap[pos]= pos+innerIndex;
          }

          for (int pos = innerIndex + 1; pos < nbOuterInputs; pos++) {
            muxOperands.push_back(outerGamma.getInputs()[pos]);
            outerPortMap[pos]= pos + nbInnerInputs-1;
          }

          /* Padding port mappin tables when the number of inputs is not a power of two */
          for (int pos = nbOuterInputs; pos < outerPow2Inputs ; pos++) {
            outerPortMap[pos]= dontCareOffset;
          }
          for (int pos = nbInnerInputs; pos < innerPow2Inputs ; pos++) {
            innerPortMap[pos]= dontCareOffset;
          }


        if (verbose) {
          llvm::errs() << vformat("┌─%6s─┬─%32s─┐\n", "id", "operand");
          for (int pos = 0; pos < muxOperands.size() ; pos++) {
            llvm::errs() << vformat("| %6d | ", pos, outerPortMap[pos]);
            llvm::errs() << " " << muxOperands[pos] << "|\n";
          }
          llvm::errs() << vformat("└─%6s─┴─%32s─┘\n", "──────", "──────");

          llvm::errs() << vformat("┌─%6s─┬─%6s─┐\n", "outer", "merged");
          for (int pos = 0; pos < outerPow2Inputs ; pos++) {
            llvm::errs() << vformat("| %6d | %6d |\n", pos, outerPortMap[pos]);
          }
          llvm::errs() << vformat("└─%6s─┴─%6s─┘\n", "──────", "──────");

          llvm::errs() << vformat("┌─%6s─┬─%6s─┐\n", "inner", "merged");
          for (int pos = 0; pos < innerPow2Inputs ; pos++) {
            llvm::errs() << vformat("| %6d | %6d |\n", pos, innerPortMap[pos]);
          }
          llvm::errs() << vformat("└─%6s─┴─%6s─┘\n", "──────", "──────");
        }



          auto controlType = rewriter.getIntegerType(
              outerGammaSelType.getIntOrFloatBitWidth() +
              innerGammaSelType.getIntOrFloatBitWidth());

          /* Cast select inputs into standard integers (in case they are unsigned) */
          auto castOuterSelect = rewriter.create<SpecHLS::CastOp>(outerGamma.getLoc(),rewriter.getIntegerType(outerGammaSelType.getIntOrFloatBitWidth()), outerGamma.getSelect());
          auto castInnerSelect = rewriter.create<SpecHLS::CastOp>(innerGamma.getLoc(),rewriter.getIntegerType(innerGammaSelType.getIntOrFloatBitWidth()),innerGamma.getSelect());
          mlir::verify(castOuterSelect);
          mlir::verify(castInnerSelect);


          /* Create merged select command from inner and  outer gamma select inputs */
          /* IMPORTANT : concat op operate in big endian mode (MSB is first operand) */
          auto concatOp = rewriter.create<comb::ConcatOp>(
              outerGamma.getLoc(), controlType,
              ValueRange({castOuterSelect,castInnerSelect }));
          mlir::verify(concatOp);

          ArrayAttr tab;
          SmallVector<int, 1024> content;

          if (verbose) llvm::errs() << concatOp << "\n";

          /* Fills the LookupTable with the reindexing information */

          if (verbose) llvm::errs() << vformat("┌─%6s─┬─%6s─┬─%6s─┐\n", "inner", "outer","input");
            for (int outer = 0; outer < outerPow2Inputs; outer++) {
              for (int inner = 0; inner < innerPow2Inputs; inner++) {
                int input = 0;

                if (outer == innerIndex) {
                  input = (innerPortMap[inner]);
                } else {
                  input = (outerPortMap[outer]);
                }
                content.push_back(input);
                if (verbose) llvm::errs()
                    << vformat("| %6d | %6d | %6d |", inner, outer, input);
                if (input < muxOperands.size())
                  if (verbose) llvm::errs() << " " << muxOperands[input] << "\n";
                else
                      if (verbose)  llvm::errs() << " undefined \n";
              }

            if (verbose) llvm::errs() << vformat("└─%6s─┴─%6s─┴─%6s─┘\n", "──────", "──────",
                                    "──────");
            //          for (int o = 0; o < innerIndex; o++) {
            //            for (int inner = 0; inner < innerPow2Inputs; inner++) {
            //              if (verbose)
            //                  llvm::errs() << "rewiring outer " << o << " to " << offset << " at " << content.size() << "\n";
            //              content.push_back(offset);
            //            }
            //            offset++;
            //          }
            //
            //          for (int inner = 0; inner < innerPow2Inputs; inner++) {
            //            if (verbose)
            //                llvm::errs() << "rewiring inner " << inner << " to " << offset << " at " << content.size() << "\n";
            //            content.push_back(offset);
            //            if (inner<innerNBinput) offset++;
            //          }
            //
            //          for (int o = innerIndex + 1; o < outerPow2Inputs; o++) {
            //            for (int inner = 0; inner < innerPow2Inputs; inner++) {
            //              if (verbose)
            //                  llvm::errs() << "rewiring outer " << o << " to " << offset << " at " << content.size() << "\n";
            //              content.push_back(offset);
            //            }
            //            if (o<(outerNBinput-1)) offset++;
            //          }
            //
          }
          // FIXME :
          int lutAddressWidth = APInt(32, content.size() - 1).getActiveBits();
          int lutResultWidth = APInt(32, mergeNBinput).getActiveBits();
          if (verbose) {
            llvm::errs() << "LUT content size " << content.size() << "\n";
            llvm::errs() << " -> outerNBinputs  " << outerNBinput << "\n";
            llvm::errs() << " -> innerNBinputs  " << innerNBinput << "\n";
            llvm::errs() << " -> address width " << lutAddressWidth << " bist, concat ->  "<< concatOp << "\n";
            llvm::errs() << " -> nbinputs  " << mergeNBinput << " on  "<< lutResultWidth << " bits\n";
          }


          if (lutAddressWidth>lutResultWidth) {
            if (verbose) llvm::errs() << " -> reducing Gamma select with to  " << lutResultWidth << " bits\n";
            lutAddressWidth = lutResultWidth;
          }

          for (int o = content.size(); o < (1<<lutAddressWidth); o++) {
            if (verbose)
                llvm::errs() << "padding lut content at " << content.size() << "with"  << offset << "\n";
            content.push_back(offset);
          }

          mlir::Type lutType = rewriter.getIntegerType(lutAddressWidth);

          auto lutSelect = rewriter.create<LookUpTableOp>(
              outerGamma.getLoc(), lutType, concatOp.getResult(),
              rewriter.getI32ArrayAttr(content));
          if (verbose)
            llvm::errs() << "creating LUT  " << lutSelect << "\n";

          mlir::verify(lutSelect);

          auto newGammaOp = rewriter.create<GammaOp>(
              outerGamma.getLoc(), outerGamma.getResult().getType(), outerGamma.getNameAttr(),
              lutSelect.getResult(), ValueRange(muxOperands));
          mlir::verify(newGammaOp);

          if (verbose) llvm::errs() << "Merged into " << newGammaOp << "\n";

          auto parent = outerGamma->getParentOp();
          rewriter.replaceOp(outerGamma, newGammaOp);
          if (nbUsers == 1)
            rewriter.eraseOp(innerGamma);
          mlir::verify(newGammaOp, true);
          if (verbose) llvm::errs() << "Merge successful\n";
          return success();
        }
      }
    }

    if (verbose) llvm::errs() << "Failure \n";
    return failure();
  }
};

struct MergeGammasPass : public impl::MergeGammasPassBase<MergeGammasPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);

    patterns.insert<GammaMergingPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      llvm::errs() << "partial conversion failed pattern  \n";
      signalPassFailure();
    }

    mlir::verify(getOperation(), true);
  }
};

std::unique_ptr<OperationPass<>> createMergeGammasPass() {
  return std::make_unique<MergeGammasPass>();
}
} // namespace SpecHLS
