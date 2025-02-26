//===- RegionGraphTraits.h - llvm::GraphTraits for CFGs ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements specializations of llvm::GraphTraits for various MLIR
// CFG data types.  This allows the generic LLVM graph algorithms to be applied
// to CFGs.
//
//===----------------------------------------------------------------------===//
  
#ifndef MLIR_IR_REGIONGRAPHTRAITS_H
#define MLIR_IR_REGIONGRAPHTRAITS_H
include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SCCIterator.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/OperationSupport.h"


namespace llvm {


template <typename Operation>
class OperationUseIterator
    : public llvm::iterator_facade_base<OperationUseIterator<Operation>,
                                        std::forward_iterator_tag,
                                        Operation> {
  using NodeRef = Operation *;
public:
  OperationUseIterator(mlir::Operation *op = nullptr) {
    current = op->use_begin();
  }

  static NodeRef getEntryNode(Operation *op) {
    return op;
  }

  static inline ChildIteratorType child_begin(NodeRef node) {
    return node->user_begin();
  }

  static inline ChildIteratorType child_end(NodeRef node) {
    return node->user_end();
  }

  using nodes_iterator = Block::iterator;

  static nodes_iterator nodes_begin(Operation *op) {
    return op->getBlock()->begin();
  }

  static nodes_iterator nodes_end(Operation *op) {
    return op->getBlock()->end();
  }

  static size_t size(Operation *op) {
    return std::distance(nodes_begin(op), nodes_end(op));
  }

//  /// Returns the operation that owns this use.
//  Operation *getUser() const { return current->getOwner(); }
//
//  /// Returns the current operands.
//  Operation *getOperation() const { return (Operation *)current->getOwner(); }
//  Operation &operator*() const { return *getOperand(); }

  using llvm::iterator_facade_base<OperationUseIterator<Operation>,
                                   std::forward_iterator_tag,
                                   Operation>::operator++;
  OperationUseIterator &operator++() {
    current;
  }

  bool operator==(const OperationUseIterator &rhs) const {
    return current == rhs.current ;
  }

protected:
  mlir::Operation::use_iterator current;
};

namespace {
// Define the GraphTraits for operations within a block
template <> struct GraphTraits<mlir::Operation *> {
  using NodeRef = mlir::Operation *;
  using ChildIteratorType = mlir::Operation::user_iterator;

  static NodeRef getEntryNode(mlir::Operation *op) {
    return op;
  }

  static inline ChildIteratorType child_begin(NodeRef node) {
    return node->user_begin();
  }

  static inline ChildIteratorType child_end(NodeRef node) {
    return node->user_end();
  }

  using nodes_iterator = Block::iterator;

  static nodes_iterator nodes_begin(Operation *op) {
    return op->getBlock()->begin();
  }

  static nodes_iterator nodes_end(Operation *op) {
    return op->getBlock()->end();
  }

  static size_t size(Operation *op) {
    return std::distance(nodes_begin(op), nodes_end(op));
  }
};
/*
template <>
struct GraphTraits<mlir::Operation *> {
  using ChildIteratorType = use_iterator;
  using Node = mlir::Value;
  using NodeRef = Node *;
  
  static NodeRef getEntryNode(NodeRef bb) { return bb; }
  
  static ChildIteratorType child_begin(NodeRef node) {
    return node->succ_begin();
  }
  static ChildIteratorType child_end(NodeRef node) { return node->succ_end(); }
};
  
template <>
struct GraphTraits<Inverse<mlir::Value *>> {
  using ChildIteratorType = mlir::Value::pred_iterator;
  using Node = mlir::Value;
  using NodeRef = Node *;
  static NodeRef getEntryNode(Inverse<NodeRef> inverseGraph) {
    return inverseGraph.Graph;
  }
  static inline ChildIteratorType child_begin(NodeRef node) {
    return node->pred_begin();
  }
  static inline ChildIteratorType child_end(NodeRef node) {
    return node->pred_end();
  }
};
  
template <>
struct GraphTraits<const mlir::Value *> {
  using ChildIteratorType = mlir::Value::succ_iterator;
  using Node = const mlir::Value;
  using NodeRef = Node *;
  
  static NodeRef getEntryNode(NodeRef node) { return node; }
  
  static ChildIteratorType child_begin(NodeRef node) {
    return const_cast<mlir::Value *>(node)->succ_begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return const_cast<mlir::Value *>(node)->succ_end();
  }
};
 */
  
template <>
struct GraphTraits<Inverse<const mlir::Value *>> {
  using ChildIteratorType = mlir::Value::pred_iterator;
  using Node = const mlir::Value;
  using NodeRef = Node *;
  
  static NodeRef getEntryNode(Inverse<NodeRef> inverseGraph) {
    return inverseGraph.Graph;
  }
  
  static ChildIteratorType child_begin(NodeRef node) {
    return const_cast<mlir::Value *>(node)->pred_begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return const_cast<mlir::Value *>(node)->pred_end();
  }
};
  
template <>
struct GraphTraits<mlir::Region *> : public GraphTraits<mlir::Value *> {
  using GraphType = mlir::Region *;
  using NodeRef = mlir::Value *;
  
  static NodeRef getEntryNode(GraphType fn) { return &fn->front(); }
  
  using nodes_iterator = pointer_iterator<mlir::Region::iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn->end());
  }
};
  
template <>
struct GraphTraits<Inverse<mlir::Region *>>
    : public GraphTraits<Inverse<mlir::Value *>> {
  using GraphType = Inverse<mlir::Region *>;
  using NodeRef = NodeRef;
  
  static NodeRef getEntryNode(GraphType fn) { return &fn.Graph->front(); }
  
  using nodes_iterator = pointer_iterator<mlir::Region::iterator>;
  static nodes_iterator nodes_begin(GraphType fn) {
    return nodes_iterator(fn.Graph->begin());
  }
  static nodes_iterator nodes_end(GraphType fn) {
    return nodes_iterator(fn.Graph->end());
  }
};
  
} // namespace llvm
  
#endif