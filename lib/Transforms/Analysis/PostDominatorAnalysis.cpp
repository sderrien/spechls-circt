#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <iostream>

using namespace mlir;
using namespace llvm;

class PostDominatorAnalysis {
  DenseMap<Operation*, SmallPtrSet<Operation*, 16>> postDom;

public:
  PostDominatorAnalysis(Operation* operation) {
    // Iterate through all operations in the region of the provided operation
    Region& region = operation->getRegion(0); // Assumes there's only one region

    SmallVector<Operation*, 16> nodes;
    for (Block& block : region) {
      for (Operation& op : block) {
        nodes.push_back(&op);
      }
    }

    // Initialize post-dominator sets for each operation
    for (Operation* op : nodes) {
      postDom[op] = SmallPtrSet<Operation*, 16>(nodes.begin(), nodes.end());
    }

    bool changed = true;

    // Iterate until no more changes occur
    while (changed) {
      changed = false;
      for (Operation* op : nodes) {
        SmallPtrSet<Operation*, 16> tmp;
        SmallVector<Operation*, 16> successors;

        // Collect successors by iterating through the successors of the block containing the operation
        for (auto* succ : op->getBlock()->getSuccessors()) {
          for (Operation& succOp : *succ) {
            successors.push_back(&succOp);
          }
        }

        if (!successors.empty()) {
          tmp = postDom[successors.front()];
          for (size_t i = 1; i < successors.size(); ++i) {
            // Intersect the current set with the post-dominator sets of successors
            SmallPtrSet<Operation*, 16> intersection;
            for (Operation* postDomOp : postDom[successors[i]]) {

