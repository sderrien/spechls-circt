//===- GenerateC.cpp - Arith-to-comb mapping pass ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the GenerateC pass.
//
//===----------------------------------------------------------------------===//
// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/Pass/Pass.h"

#include "SpecHLS/SpecHLSDialect.h"
#include "SpecHLS/SpecHLSOps.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace circt;
using namespace SpecHLS;

std::string replace_all(std::string str, const std::string &remove, const std::string &insert)
{
  std::string::size_type pos = 0;
  while ((pos = str.find(remove, pos)) != std::string::npos)
  {
    str.replace(pos, remove.size(), insert);
    pos++;
  }

  return str;
}

namespace SpecHLS {

struct GenerateCPass : public impl::GenerateCPassBase<GenerateCPass> {

  /// Manages the indentation as we traverse the IR nesting.
  int indent;
  struct IdentRAII {
    int &indent;
    IdentRAII(int &indent) : indent(indent) {}
    ~IdentRAII() { --indent; }
  };
  void resetIndent() { indent = 0; }
  IdentRAII pushIndent() { return IdentRAII(++indent); }

  llvm::raw_ostream &printIndent() {
    for (int i = 0; i < indent; ++i)
      llvm::outs() << "  ";
    return llvm::outs();
  }

  void printOperation(Operation *op) {
    // Print the operation itself and some of its properties
    printIndent() << "visiting op: '" << op->getName() << "' with "
                  << op->getNumOperands() << " operands and "
                  << op->getNumResults() << " results\n";
    // Print the operation attributes
    if (!op->getAttrs().empty()) {
      printIndent() << op->getAttrs().size() << " attributes:\n";
      //      for (NamedAttribute attr : op->getAttrs())
      //        printIndent() << " - '" << attr.first << "' : '" << attr.second
      //                      << "'\n";
    }

    // Recurse into each of the regions attached to the operation.
    printIndent() << " " << op->getNumRegions() << " nested regions:\n";
    auto indent = pushIndent();

    if (op->getRegions().empty()) {
      TypeSwitch<Operation *>(op)
          .Case<SpecHLS::MuOp, SpecHLS::GammaOp>([&](auto op) { return true; })
          .Default([&](auto op) {});
    }

    for (Region &region : op->getRegions())
      printRegion(region);
  }

  void printRegion(Region &region) {
    // A region does not hold anything by itself other than a list of blocks.
    printIndent() << "Region with " << region.getBlocks().size()
                  << " blocks:\n";
    auto indent = pushIndent();
    for (Block &block : region.getBlocks())
      printBlock(block);
  }

  void printBlock(Block &block) {
    // Print the block intrinsics properties (basically: argument list)
    printIndent()
        << "Block with " << block.getNumArguments() << " arguments, "
        << block.getNumSuccessors()
        << " successors, and "
        // Note, this `.size()` is traversing a linked-list and is O(n).
        << block.getOperations().size() << " operations\n";

    // A block main role is to hold a list of Operations: let's recurse into
    // printing each operation.
    auto indent = pushIndent();
    for (Operation &op : block.getOperations())
      printOperation(&op);
  }

public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    auto op = getOperation();

    bool res =
        TypeSwitch<Operation *, bool>(op)
            .Case<circt::comb::AndOp>([&](auto op) {
              llvm::outs() << " found and " << *op << "\n";
              return true;
            })
            .Case<circt::comb::OrOp>([&](auto op) {
              return true;
            })
            .Case<circt::comb::XorOp>([&](auto op) {
              return true;
            })
            .Case<circt::comb::ExtractOp>([&](circt::comb::ExtractOp extract) {
              auto pattern = R"(
                if ({{expr}}) {
                    {}
                } else {
                    {}
                )";

              std::string str = replace_all(replace_all(replace_all(pattern,
                       "{{first_name}}", "Homer"),
                       "{{last_name}}", "Simpson"),"{{location}}", "Springfield");
                return true;
                  ;
            })
            .Case<circt::comb::ConcatOp>(
                [&](auto op) { return true; })
            .Case<circt::comb::MuxOp>([&](auto op) { return true; })
            .Case<circt::comb::TruthTableOp>(
                [&](auto op) { return true; })
            .Case<SpecHLS::LookUpTableOp>(
                [&](auto op) { return true; })
            .Default([&](auto op) {
              llvm::outs() << " default filter  " << *op << "\n";
              return false;
            });
    llvm::outs() << " res " << res << "\n";


    //
    //    target.addIllegalDialect<arith::ArithDialect>();
    //    MapArithTypeConverter typeConverter;
    // RewritePatternSet patterns(ctx);
    //
    // patterns.insert<LookUpMergingPattern>(ctx);
    // llvm::errs() << "inserted pattern  \n";

    //    if
    //    (failed(applyPatternsAndFoldGreedily(getOperation(),std::move(patterns))))
    //    {
    //      llvm::errs() << "partial conversion failed pattern  \n";
    signalPassFailure();

  }
};

std::unique_ptr<OperationPass<>> createGenerateCPass() {
  return std::make_unique<GenerateCPass>();
}
} // namespace SpecHLS
