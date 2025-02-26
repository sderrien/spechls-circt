#ifndef SPECHLS_DIALECT_DEFUSESCC_H
#define SPECHLS_DIALECT_DEFUSESCC_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>
#include <iostream>

// DefUseSCC class to hold SCCs.
class DefUseSCC {
public:
  llvm::SmallVector<mlir::Operation *, 4> operations; // SCC components

  DefUseSCC() = default;


  // Stream output operator for DefUseSCC
  friend std::ostream &operator<<(std::ostream &os, const DefUseSCC &scc) {
    os << "SCC with " << scc.operations.size() << " operations:\n";
    for (mlir::Operation *op : scc.operations) {

      os << "  Operation: " << op << "\n"; // Customize for useful printing
    }
    return os;
  }
};

// Class to compute SCCs using Tarjan's algorithm.
class SCCComputer {
public:
  using SCCCollection = std::vector<DefUseSCC>;

  SCCComputer() = default;

  SCCComputer::SCCCollection computeSCCs(mlir::Block &block) ;

private:
  void dfs(mlir::Operation *op);
  int index = 0;
  llvm::DenseMap<mlir::Operation *, int> opIndex, opLowLink;
  llvm::SmallVector<mlir::Operation *> stack;
  llvm::DenseSet<mlir::Operation *> onStack;
  SCCCollection sccs;
};


#endif // SPECHLS_DIALECT_DEFUSESCC_H
