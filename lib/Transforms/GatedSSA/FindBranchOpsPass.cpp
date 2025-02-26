#include "FindBranchOpsPass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/raw_ostream.h"


using namespace mlir;
using namespace func;

namespace {
class FindBranchOpsPass : public PassWrapper<FindBranchOpsPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FindBranchOpsPass)

  explicit FindBranchOpsPass() {}

  StringRef getArgument() const final { return "find-branch-ops"; }

  /// Detailed description of the pass
  StringRef getDescription() const final {
    return "This pass performs the following operations:\n"
           "1. Identifies all back-edges and loop header blocks in a function.\n"
           "2. Checks if the function has only one loop header block, triggers an error otherwise.\n"
           "3. Uses the loop header block as the target block for further analysis.\n"
           "4. Finds all branch operations targeting the given block and constructs maps of source blocks and conditions.\n"
           "5. Inserts Gamma and Encoder operations at the start of the block.\n"
           "6. Replaces memref.store operations with SpecHLS.alphaOp operations.\n"
           "7. Identifies SESE regions in the function using dominance information.\n"
           "8. Flattens the function into a single superblock.\n"
           "9. Prints all useful intermediate info/result data structures for debugging.";
  }

  void runOnOperation() override {
    FuncOp function = getOperation();

    // Identify all back-edges and loop header blocks
    auto backEdges = identifyBackEdges(function);
    llvm::errs() << "Back-Edges:\n";
    for (auto &edge : backEdges) {
      llvm::errs() << "  From: " << edge.first << " To: " << edge.second << "\n";
    }

    // Identify loop headers
    llvm::DenseSet<Block *> loopHeaders;
    for (auto &edge : backEdges) {
      loopHeaders.insert(edge.second);
    }

    // Check if there is exactly one loop header block
    if (loopHeaders.size() != 1) {
      function.emitError() << "The function must have exactly one loop header block, found: " << loopHeaders.size();
      return;
    }

    Block *targetBlock = *loopHeaders.begin();
    llvm::errs() << "Target Block: " << targetBlock << "\n";

    // Initialize maps for branch information
    std::map<Block *, std::vector<std::pair<Block *, Value>>> branchMap;
    std::map<Block *, std::vector<std::pair<Block *, SmallVector<Value, 4>>>> argMap;

    // Populate branchMap and argMap with branch information
    OpBuilder builder(targetBlock->getTerminator());
    for (auto &block : function) {
      for (auto &op : block) {
        if (auto brOp = dyn_cast<cf::BranchOp>(op)) {
          if (brOp.getDest() == targetBlock) {
            branchMap[targetBlock].emplace_back(&block, nullptr);
            argMap[targetBlock].emplace_back(&block, brOp.getDestOperands());
          }
        } else if (auto condBrOp = dyn_cast<mlir::cf::CondBranchOp>(op)) {
          if (condBrOp.getTrueDest() == targetBlock) {
            branchMap[targetBlock].emplace_back(&block, condBrOp.getCondition());
            argMap[targetBlock].emplace_back(&block, condBrOp.getTrueOperands());
          } else if (condBrOp.getFalseDest() == targetBlock) {
            auto negCondition = builder.create<arith::XOrIOp>(op.getLoc(), condBrOp.getCondition(),condBrOp.getCondition());
            branchMap[targetBlock].emplace_back(&block, negCondition);
            argMap[targetBlock].emplace_back(&block, condBrOp.getFalseOperands());
          }
        }
      }
    }

    // Print branchMap and argMap for debugging
    llvm::errs() << "Branch Map:\n";
    for (const auto &entry : branchMap) {
      llvm::errs() << "  Target Block: " << entry.first << "\n";
      for (const auto &pair : entry.second) {
        llvm::errs() << "    Source Block: " << pair.first << " Condition: " << (pair.second ? "Non-null" : "Null") << "\n";
      }
    }

    llvm::errs() << "Arg Map:\n";
    for (const auto &entry : argMap) {
      llvm::errs() << "  Target Block: " << entry.first << "\n";
      for (const auto &pair : entry.second) {
        llvm::errs() << "    Source Block: " << pair.first << " Args: ";
        for (const auto &arg : pair.second) {
          llvm::errs() << arg << " ";
        }
        llvm::errs() << "\n";
      }
    }

    // For every target block, add Gamma and Encoder ops
    for (auto &entry : branchMap) {
      Block *block = entry.first;
      auto &conditions = entry.second;
      auto &args = argMap[block];

      if (conditions.empty()) continue;

      // Create encoder op with conditions
      builder.setInsertionPointToStart(block);
      SmallVector<Value, 4> predicateValues;
      for (auto &cond : conditions) {
        if (cond.second) {
          predicateValues.push_back(cond.second);
        }
      }
      Value encoderResult = builder.create<SpecHLS::EncoderOp>(block->getParent()->getLoc(), builder.getIndexType(), predicateValues);

      // Create Gamma ops for each block argument
      for (auto &arg : block->getArguments()) {
        SmallVector<Value, 4> gammaInputs;
        for (auto &argPair : args) {
          gammaInputs.push_back(argPair.second[arg.getArgNumber()]);
        }
        SmallVector<Value, 1> controlSignal = {encoderResult};
//        static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::FlatSymbolRefAttr name, ::mlir::Value select, ::mlir::ValueRange inputs);
//        static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::FlatSymbolRefAttr name, ::mlir::Value select, ::mlir::ValueRange inputs);
//        static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::llvm::StringRef name, ::mlir::Value select, ::mlir::ValueRange inputs);
//        static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, ::mlir::Value select, ::mlir::ValueRange inputs);
//        static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});

        builder.create<SpecHLS::GammaOp>(block->getParent()->getLoc(), arg.getType(), StringRef("dummy"), controlSignal, gammaInputs);
      }

      // Replace memref.store with SpecHLS.alphaOp in the block.
      for (auto &op : llvm::make_early_inc_range(*block)) {
        if (auto storeOp = dyn_cast<memref::StoreOp>(&op)) {
          builder.setInsertionPoint(&op);
///*
// * static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::FlatSymbolRefAttr name, ::mlir::Value memref, ::mlir::Value value, ::mlir::ValueRange indices, /*optional*/::mlir::Value we);
//          static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::FlatSymbolRefAttr name, ::mlir::Value memref, ::mlir::Value value, ::mlir::ValueRange indices, /*optional*/::mlir::Value we);
//          static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::FlatSymbolRefAttr name, ::mlir::Value memref, ::mlir::Value value, ::mlir::ValueRange indices, /*optional*/::mlir::Value we);
//          static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::StringRef name, ::mlir::Value memref, ::mlir::Value value, ::mlir::ValueRange indices, /*optional*/::mlir::Value we);
//          static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef name, ::mlir::Value memref, ::mlir::Value value, ::mlir::ValueRange indices, /*optional*/::mlir::Value we);
//          static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, ::mlir::Value memref, ::mlir::Value value, ::mlir::ValueRange indices, /*optional*/::mlir::Value we);
//          static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
//          static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
//
// */
          builder.create<SpecHLS::AlphaOp>(storeOp.getLoc(), storeOp.getValueToStore().getType(),"tst",storeOp.getMemRef(),storeOp.getValueToStore(), storeOp.getIndices());
          storeOp.erase();
        }
      }
    }

    // Identify all SESE regions in the function
    auto seseRegions = identifySESERegions(function);

    // Print SESE regions for debugging
    llvm::errs() << "SESE Regions:\n";
    for (const auto &entry : seseRegions) {
      llvm::errs() << "  Entry Block: " << entry.first << "\n";
      for (const auto &exitEntry : entry.second) {
        llvm::errs() << "    Exit Block: " << exitEntry.first << "\n";
        llvm::errs() << "    Blocks: ";
        for (const auto &block : exitEntry.second) {
          llvm::errs() << block << " ";
        }
        llvm::errs() << "\n";
      }
    }

    // Flatten all blocks into a single superblock
    flattenFunctionIntoSingleBlock(function);
  }

private:
  /// Identifies back-edges in the given function using dominance information.
  /// A back-edge exists if the target block of a branch dominates the block containing the branch.
  std::vector<std::pair<Block *, Block *>> identifyBackEdges(FuncOp function) {
    DominanceInfo domInfo(function);
    std::vector<std::pair<Block *, Block *>> backEdges;

    // Iterate through all blocks and their operations to find branches
    for (Block &block : function) {
      for (Operation &op : block) {
        if (auto brOp = dyn_cast<cf::BranchOp>(&op)) {
          Block *dest = brOp.getDest();
          // Check if the destination block dominates the current block
          if (domInfo.dominates(dest, &block)) {
            backEdges.emplace_back(&block, dest);
          }
        } else if (auto condBrOp = dyn_cast<cf::CondBranchOp>(&op)) {
          Block *trueDest = condBrOp.getTrueDest();
          Block *falseDest = condBrOp.getFalseDest();
          // Check if the true or false destination blocks dominate the current block
          if (domInfo.dominates(trueDest, &block)) {
            backEdges.emplace_back(&block, trueDest);
          }
          if (domInfo.dominates(falseDest, &block)) {
            backEdges.emplace_back(&block, falseDest);
          }
        }
      }
    }

    return backEdges;
  }

  /// Identifies SESE (Single Entry Single Exit) regions in the given function using dominance information.
  /// An SESE region is defined by an entry block, an exit block, and the blocks within the region.
  std::map<Block *, std::map<Block *, std::vector<Block *>>> identifySESERegions(FuncOp function) {
    std::map<Block *, std::map<Block *, std::vector<Block *>>> seseRegions;
    PostDominanceInfo domInfo(function);

    // Iterate through all pairs of blocks to determine SESE regions
    for (Block &entryBlock : function) {
      for (Block &exitBlock : function) {
        if (&entryBlock == &exitBlock)
          continue;

        // Check if the entry block dominates the exit block
        if (domInfo.postDominates(&entryBlock, &exitBlock)) {
          SmallVector<Block *, 4> regionBlocks;
          for (Block &block : function) {
            // A block is within the SESE region if it is dominated by the entry block and post-dominated by the exit block
            if (domInfo.postDominates(&entryBlock, &block) &&
                domInfo.postDominates(&exitBlock, &block)) {
              regionBlocks.push_back(&block);
            }
          }
          if (!regionBlocks.empty()) {
            seseRegions[&entryBlock][&exitBlock] = std::move(regionBlocks);
          }
        }
      }
    }

    return seseRegions;
  }

  /// Flattens the given function into a single superblock by moving all operations from other blocks into the superblock.
  /// Also replaces terminators with branches to maintain correct control flow.
  void flattenFunctionIntoSingleBlock(FuncOp function) {
    // Create a new block to serve as the superblock
    Block *superBlock = new Block();
    function.getBlocks().push_front(superBlock);

    OpBuilder builder(superBlock, superBlock->begin());

    // Move all operations from each block to the superblock
    for (auto &block : llvm::make_early_inc_range(function)) {
      if (&block == superBlock) continue; // Skip the newly created superblock
      superBlock->getOperations().splice(superBlock->end(), block.getOperations());
    }

    // Handle the original terminator of the function
    if (function.getBlocks().back().empty()) {
      function.getBlocks().pop_back();
    }

    // Replace terminators with branches to ensure correct control flow
    for (auto &op : llvm::make_early_inc_range(*superBlock)) {
      if (auto brOp = dyn_cast<cf::BranchOp>(&op)) {
        builder.setInsertionPoint(&op);
        builder.create<cf::BranchOp>(op.getLoc(), brOp.getDest());
        op.erase();
      } else if (auto condBrOp = dyn_cast<cf::CondBranchOp>(&op)) {
        builder.setInsertionPoint(&op);
        builder.create<cf::CondBranchOp>(op.getLoc(), condBrOp.getCondition(), condBrOp.getTrueDest(), condBrOp.getTrueDestOperands(), condBrOp.getFalseDest(), condBrOp.getFalseDestOperands());
        op.erase();
      }
    }

    // Ensure the superblock is the entry block
    function.getBlocks().splice(function.getBlocks().begin(), function.getBlocks(), superBlock);
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> mlir::createFindBranchOpsPass() {
  return std::make_unique<FindBranchOpsPass>();
}

void mlir::cf::registerFindBranchOpsPass() {
  PassPipelineRegistration<>(
      "find-branch-ops",
      "Find all branch operations to a given basic block.",
      [](OpPassManager &pm) {
        pm.addPass(createFindBranchOpsPass());
      }
  );
}
