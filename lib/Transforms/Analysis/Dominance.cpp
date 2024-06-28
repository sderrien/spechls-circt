//===- Dominance.cpp - Dominator analysis for CFGs ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of dominance related classes and instantiations of extern
// templates.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

using namespace mlir;
using namespace mlir::detail;

template class llvm::DominatorTreeBase<Operation, /*IsPostDom=*/false>;
template class llvm::DominatorTreeBase<Operation, /*IsPostDom=*/true>;
template class llvm::DomTreeNodeBase<Operation>;

//===----------------------------------------------------------------------===//
// DominanceInfoBase
//===----------------------------------------------------------------------===//

template <bool IsPostDom>
DominanceInfoBase<IsPostDom>::~DominanceInfoBase() {
  for (auto entry : dominanceInfos)
    delete entry.second.getPointer();
}

template <bool IsPostDom> void DominanceInfoBase<IsPostDom>::invalidate() {
  for (auto entry : dominanceInfos)
    delete entry.second.getPointer();
  dominanceInfos.clear();
}

template <bool IsPostDom>
void DominanceInfoBase<IsPostDom>::invalidate(Block *block) {
  auto it = dominanceInfos.find(block);
  if (it != dominanceInfos.end()) {
    delete it->second.getPointer();
    dominanceInfos.erase(it);
  }
}

/// Return the dom tree and "hasSSADominance" bit for the given block.  The
/// DomTree will be null for single-block blocks.  This lazily constructs the
/// DomTree on demand when needsDomTree=true.
template <bool IsPostDom>
auto DominanceInfoBase<IsPostDom>::getDominanceInfo(Block *block,
                                                    bool needsDomTree) const
    -> llvm::PointerIntPair<DomTree *, 1, bool> {
  // Check to see if we already have this information.
  auto itAndInserted = dominanceInfos.insert({block, {nullptr, true}});
  auto &entry = itAndInserted.first->second;

  // This method builds on knowledge that multi-block blocks always have
  // SSADominance.  Graph blocks are only allowed to be single-block blocks,
  // but of course single-block blocks may also have SSA dominance.
  if (!itAndInserted.second) {
    // We do have it, so we know the 'hasSSADominance' bit is correct, but we
    // may not have constructed a DominatorTree yet.  If we need it, build it.
    if (needsDomTree && !entry.getPointer() && !block->hasOneOperation()) {
      auto *domTree = new DomTree();
      domTree->recalculate(*block);
      entry.setPointer(domTree);
    }
    return entry;
  }

  // Nope, lazily construct it.  Create a DomTree if this is a multi-block
  // block.
  if (!block->hasOneOperation()) {
    auto *domTree = new DomTree();
    domTree->recalculate(*block);
    entry.setPointer(domTree);
    // Multiblock blocks always have SSA dominance, leave `second` set to true.
    return entry;
  }

  // Single block blocks have a more complicated predicate.
  if (Operation *parentOp = block->getParentOp()) {
    if (!parentOp->isRegistered()) { // We don't know about unregistered ops.
      entry.setInt(false);
    } else if (auto blockKindItf = dyn_cast<BlockKindInterface>(parentOp)) {
      // Registered ops can opt-out of SSA dominance with
      // BlockKindInterface.
      entry.setInt(blockKindItf.hasSSADominance(block->getBlockNumber()));
    }
  }

  return entry;
}

/// Return the ancestor block enclosing the specified block.  This returns null
/// if we reach the top of the hierarchy.
static Operation *getAncestorOperation(Operation *block) {
  if (Operation *ancestorOp = block->getParentOp())
    return ancestorOp->getOperation();
  return nullptr;
}

/// Walks up the list of containers of the given block and calls the
/// user-defined traversal function for every pair of a block and block that
/// could be found during traversal. If the user-defined function returns true
/// for a given pair, traverseAncestors will return the current block. Nullptr
/// otherwise.
template <typename FuncT>
static Operation *traverseAncestors(Operation *block, const FuncT &func) {
  do {
    // Invoke the user-defined traversal function for each block.
    if (func(block))
      return block;
  } while ((block = getAncestorOperation(block)));
  return nullptr;
}

/// Tries to update the given block references to live in the same block by
/// exploring the relationship of both blocks with respect to their blocks.
static bool tryGetOperationsInSameBlock(Operation *&a, Operation *&b) {
  // If both block do not live in the same block, we will have to check their
  // parent operations.
  Block *aBlock = a->getParent();
  Block *bBlock = b->getParent();
  if (aBlock == bBlock)
    return true;

  // Iterate over all ancestors of `a`, counting the depth of `a`. If one of
  // `a`s ancestors are in the same block as `b`, then we stop early because we
  // found our NCA.
  size_t aBlockDepth = 0;
  if (Operation *aResult = traverseAncestors(a, [&](Operation *block) {
        ++aBlockDepth;
        return block->getParent() == bBlock;
      })) {
    a = aResult;
    return true;
  }

  // Iterate over all ancestors of `b`, counting the depth of `b`. If one of
  // `b`s ancestors are in the same block as `a`, then we stop early because
  // we found our NCA.
  size_t bBlockDepth = 0;
  if (Operation *bResult = traverseAncestors(b, [&](Operation *block) {
        ++bBlockDepth;
        return block->getParent() == aBlock;
      })) {
    b = bResult;
    return true;
  }

  // Otherwise we found two blocks that are siblings at some level.  Walk the
  // deepest one up until we reach the top or find an NCA.
  while (true) {
    if (aBlockDepth > bBlockDepth) {
      a = getAncestorOperation(a);
      --aBlockDepth;
    } else if (aBlockDepth < bBlockDepth) {
      b = getAncestorOperation(b);
      --bBlockDepth;
    } else {
      break;
    }
  }

  // If we found something with the same level, then we can march both up at the
  // same time from here on out.
  while (a) {
    // If they are at the same level, and have the same parent block then we
    // succeeded.
    if (a->getParent() == b->getParent())
      return true;

    a = getAncestorOperation(a);
    b = getAncestorOperation(b);
  }

  // They don't share an NCA, perhaps they are in different modules or
  // something.
  return false;
}

template <bool IsPostDom>
Operation *
DominanceInfoBase<IsPostDom>::findNearestCommonDominator(Operation *a,
                                                         Operation *b) const {
  // If either a or b are null, then conservatively return nullptr.
  if (!a || !b)
    return nullptr;

  // If they are the same block, then we are done.
  if (a == b)
    return a;

  // Try to find blocks that are in the same block.
  if (!tryGetOperationsInSameBlock(a, b))
    return nullptr;

  // If the common ancestor in a common block is the same block, then return
  // it.
  if (a == b)
    return a;

  // Otherwise, there must be multiple blocks in the block, check the
  // DomTree.
  return getDomTree(a->getParent()).findNearestCommonDominator(a, b);
}

/// Return true if the specified block A properly dominates block B.
template <bool IsPostDom>
bool DominanceInfoBase<IsPostDom>::properlyDominates(Operation *a, Operation *b) const {
  assert(a && b && "null blocks not allowed");

  // A block dominates itself but does not properly dominate itself.
  if (a == b)
    return false;

  // If both blocks are not in the same block, `a` properly dominates `b` if
  // `b` is defined in an operation block that (recursively) ends up being
  // dominated by `a`. Walk up the list of containers enclosing B.
  Block *blockA = a->getParent();
  if (blockA != b->getParent()) {
    b = blockA ? blockA->findAncestorOperationInBlock(*b) : nullptr;
    // If we could not find a valid block b then it is a not a dominator.
    if (b == nullptr)
      return false;

    // Check to see if the ancestor of `b` is the same block as `a`.  A properly
    // dominates B if it contains an op that contains the B block.
    if (a == b)
      return true;
  }

  // Otherwise, they are two different blocks in the same block, use DomTree.
  return getDomTree(blockA).properlyDominates(a, b);
}

/// Return true if the specified block is reachable from the entry block of
/// its block.
template <bool IsPostDom>
bool DominanceInfoBase<IsPostDom>::isReachableFromEntry(Operation *a) const {
  // If this is the first block in its block, then it is obviously reachable.
  Block *block = a->getParent();
  if (&block->front() == a)
    return true;

  // Otherwise this is some block in a multi-block block.  Check DomTree.
  return getDomTree(block).isReachableFromEntry(a);
}

template class detail::DominanceInfoBase</*IsPostDom=*/true>;
template class detail::DominanceInfoBase</*IsPostDom=*/false>;

//===----------------------------------------------------------------------===//
// DominanceInfo
//===----------------------------------------------------------------------===//

/// Return true if operation `a` properly dominates operation `b`.  The
/// 'enclosingOpOk' flag says whether we should return true if the `b` op is
/// enclosed by a block on 'a'.
bool DominanceInfo::properlyDominatesImpl(Operation *a, Operation *b,
                                          bool enclosingOpOk) const {
  Operation *aOperation = a->getOperation(), *bOperation = b->getOperation();
  assert(aOperation && bOperation && "operations must be in a block");

  // An instruction dominates, but does not properlyDominate, itself unless this
  // is a graph block.
  if (a == b)
    return !hasSSADominance(aOperation);

  // If these ops are in different blocks, then normalize one into the other.
  Block *aBlock = aOperation->getParent();
  if (aBlock != bOperation->getParent()) {
    // Scoot up b's block tree until we find an operation in A's block that
    // encloses it.  If this fails, then we know there is no post-dom relation.
    b = aBlock ? aBlock->findAncestorOpInBlock(*b) : nullptr;
    if (!b)
      return false;
    bOperation = b->getOperation();
    assert(bOperation->getParent() == aBlock);

    // If 'a' encloses 'b', then we consider it to dominate.
    if (a == b && enclosingOpOk)
      return true;
  }

  // Ok, they are in the same block now.
  if (aOperation == bOperation) {
    // Dominance changes based on the block type. In a block with SSA
    // dominance, uses inside the same block must follow defs. In other
    // blocks kinds, uses and defs can come in any order inside a block.
    if (hasSSADominance(aOperation)) {
      // If the blocks are the same, then check if b is before a in the block.
      return a->isBeforeInOperation(b);
    }
    return true;
  }

  // If the blocks are different, use DomTree to resolve the query.
  return getDomTree(aBlock).properlyDominates(aOperation, bOperation);
}

/// Return true if the `a` value properly dominates operation `b`, i.e if the
/// operation that defines `a` properlyDominates `b` and the operation that
/// defines `a` does not contain `b`.
bool DominanceInfo::properlyDominates(Value a, Operation *b) const {
  // block arguments properly dominate all operations in their own block, so
  // we use a dominates check here, not a properlyDominates check.
  if (auto blockArg = dyn_cast<OperationArgument>(a))
    return dominates(blockArg.getOwner(), b->getOperation());

  // `a` properlyDominates `b` if the operation defining `a` properlyDominates
  // `b`, but `a` does not itself enclose `b` in one of its blocks.
  return properlyDominatesImpl(a.getDefiningOp(), b, /*enclosingOpOk=*/false);
}

//===----------------------------------------------------------------------===//
// PostDominanceInfo
//===----------------------------------------------------------------------===//

/// Returns true if statement 'a' properly postdominates statement b.
bool PostDominanceInfo::properlyPostDominates(Operation *a, Operation *b) {
  auto *aOperation = a->getOperation(), *bOperation = b->getOperation();
  assert(aOperation && bOperation && "operations must be in a block");

  // An instruction postDominates, but does not properlyPostDominate, itself
  // unless this is a graph block.
  if (a == b)
    return !hasSSADominance(aOperation);

  // If these ops are in different blocks, then normalize one into the other.
  Block *aBlock = aOperation->getParent();
  if (aBlock != bOperation->getParent()) {
    // Scoot up b's block tree until we find an operation in A's block that
    // encloses it.  If this fails, then we know there is no post-dom relation.
    b = aBlock ? aBlock->findAncestorOpInBlock(*b) : nullptr;
    if (!b)
      return false;
    bOperation = b->getOperation();
    assert(bOperation->getParent() == aBlock);

    // If 'a' encloses 'b', then we consider it to postdominate.
    if (a == b)
      return true;
  }

  // Ok, they are in the same block.  If they are in the same block, check if b
  // is before a in the block.
  if (aOperation == bOperation) {
    // Dominance changes based on the block type.
    if (hasSSADominance(aOperation)) {
      // If the blocks are the same, then check if b is before a in the block.
      return b->isBeforeInOperation(a);
    }
    return true;
  }

  // If the blocks are different, check if a's block post dominates b's.
  return getDomTree(aBlock).properlyDominates(aOperation, bOperation);
}
