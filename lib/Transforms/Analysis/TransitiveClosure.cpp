#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Analysis/SliceAnalysis.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/Support/raw_ostream.h"
#include "Common/TransitiveClosure.h"
#include <functional>
#include <unordered_map>
#include <vector>


using namespace mlir;



#ifndef  __SpecHLSTransitiveDependencyAnalyzer_IMPL
#define __SpecHLSTransitiveDependencyAnalyzer_IMPL

  void PathAnalyzer::mapOperations() {
    size_t index = 0;
    for (mlir::Block &block : region) {
      for (mlir::Operation &op : block) {
        opToIndexMap[&op] = index;
        indexToOp.push_back(&op);
        ++index;
      }
    }
  }

  void PathAnalyzer::buildAdjacencyMatrix(std::function<bool(mlir::Operation *)> filter) {

    std::vector<llvm::BitVector> &forwardMatrix = *forwardAdjMatrix;
    std::vector<llvm::BitVector> &backwardMatrix = *backwardAdjMatrix;

    for (mlir::Block &block : region) {
      for (mlir::Operation &op : block) {
        size_t opIndex = opToIndexMap[&op];
        for (mlir::Value operand : op.getOperands()) {
          if (auto *defOp = operand.getDefiningOp()) {
            if (!filter(defOp)) {
              continue;
            }
            size_t defOpIndex = opToIndexMap[defOp];
            forwardMatrix[defOpIndex][opIndex]=true ;
          }
        }
      }
    }
    /*
     * Transpose adjacency matrix to obtain reverse dep graph
     */
    size_t n = forwardMatrix.size();
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        backwardMatrix[i][j] = forwardMatrix[j][i] ;
      }
    }

  }

  void PathAnalyzer::computeTransitiveClosure() {
    std::vector<llvm::BitVector> &backwardMatrix = *backwardAdjMatrix;
    std::vector<llvm::BitVector> &forwardMatrix = *forwardAdjMatrix;

    size_t n = forwardMatrix.size();
    for (size_t k = 0; k < n; ++k) {
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          forwardMatrix[i][j] = forwardMatrix[i][j] || (forwardMatrix[i][k] && forwardMatrix[k][j]);
          backwardMatrix[i][j] = backwardMatrix[i][j] || (backwardMatrix[i][k] && backwardMatrix[k][j]);
        }
      }
    }


  }


  PathAnalyzer::PathAnalyzer(Region &region, bool intraIteration)
      : region(region) {
    llvm::DenseMap<Operation *, size_t> opToIndexMap;
    size_t numOps = 0;
    for (mlir::Block &block : region) {
      numOps += block.getOperations().size();
    }

    indexToOp.reserve(numOps);
    mapOperations();

    forwardAdjMatrix = new std::vector<llvm::BitVector>(numOps, llvm::BitVector(numOps, false));
    std::vector<llvm::BitVector> &mat = *adjMatrix;
    // Define a lambda function for filtering defining operations
    auto filter = [](mlir::Operation *op) -> bool {
      // Example filter: include all operations
      // Modify this lambda to filter specific defining operations as needed
      return true;
    };

    buildAdjacencyMatrix(filter);
    computeTransitiveClosure();

    // Print the transitive closure matrix
    llvm::outs() << "Transitive closure matrix:\n";
    for (size_t i = 0; i < numOps; ++i) {
      for (size_t j = 0; j < numOps; ++j) {

        llvm::outs() <<( mat[i][j] ? "1 " : "0 ");
      }
      llvm::outs() << "\n";
    }
  }

  bool PathAnalyzer::isTransitiveSuccessor(mlir::Operation *op,mlir::Operation *succ) {
    int j= opToIndexMap[op];
    int i= opToIndexMap[succ];
    std::vector<llvm::BitVector> &fowardMatrix = *forwardAdjMatrix;
    return fowardMatrix[i][j];
  }

  bool PathAnalyzer::isTransitivePredecessor(mlir::Operation *op,mlir::Operation *pred) {
    int j= opToIndexMap[op];
    int i= opToIndexMap[pred];
    std::vector<llvm::BitVector> &backwardMatrix = *backwardAdjMatrix;
    return backwardMatrix[i][j];
  }

  llvm::SmallVector<mlir::Operation *> PathAnalyzer::getTransitiveSuccessors(mlir::Operation *op) {
    llvm::SmallVector<mlir::Operation *> res;
    std::vector<llvm::BitVector> &backwardMatrix = *backwardAdjMatrix;
    std::vector<llvm::BitVector> &forwardMatrix = *forwardAdjMatrix;

    int j= opToIndexMap[op];
    for (size_t i = 0; i < indexToOp.size(); ++i) {
      if (forwardMatrix[i][j]) {
        mlir::Operation *b=indexToOp[i];
        res.push_back(b);
      }
    }
    return res;
  }

  llvm::SmallVector<mlir::Operation *> PathAnalyzer::getTransitivePredecessors(mlir::Operation *op) {
    llvm::SmallVector<mlir::Operation *> res;
    std::vector<llvm::BitVector> &backwardMatrix = *backwardAdjMatrix;
    std::vector<llvm::BitVector> &forwardMatrix = *forwardAdjMatrix;

    int j= opToIndexMap[op];
    for (size_t i = 0; i < indexToOp.size(); ++i) {
      if (forwardMatrix[i][j]) {
        mlir::Operation *b=indexToOp[i];
        res.push_back(b);
      }
    }
    return res;
  }
#endif
