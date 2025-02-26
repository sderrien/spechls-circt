#include "mlir/IR/Operation.h"
#include "llvm/ADT/GraphTraits.h"

namespace llvm {

// Custom iterator to traverse the operations that use the results of a given operation.
class OperationUsesIterator {
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = mlir::Operation*;
  using difference_type = std::ptrdiff_t;
  using pointer = mlir::Operation**;
  using reference = mlir::Operation*&;

private:
  mlir::Operation::result_iterator resultIt;
  mlir::Operation::result_iterator resultEnd;
  mlir::Value::use_iterator useIt;
  mlir::Value::use_iterator useEnd;

  void advanceToNextUse() {
    while (true) {
      if (useIt != useEnd) {
        // Found a valid use
        return;
      }
      // Advance to next result
      if (resultIt == resultEnd) {
        // No more results
        useIt = useEnd = mlir::Value::use_iterator();
        return;
      }
      mlir::Value result = *resultIt;
      ++resultIt;
      useIt = result.use_begin();
      useEnd = result.use_end();
    }
  }

public:
  // Delete the default constructor
  OperationUsesIterator() = delete;

  // Constructor for begin iterator
  explicit OperationUsesIterator(mlir::Operation* op)
      : resultIt(op->result_begin()),
        resultEnd(op->result_end()),
        useIt(),
        useEnd() {
    if (resultIt != resultEnd) {
      mlir::Value result = *resultIt;
      ++resultIt;
      useIt = result.use_begin();
      useEnd = result.use_end();
      advanceToNextUse();
    } else {
      // No results; set useIt and useEnd to default
      useIt = useEnd = mlir::Value::use_iterator();
    }
  }

  // Constructor for end iterator
  OperationUsesIterator(mlir::Operation* op, bool)
      : resultIt(op->result_end()),
        resultEnd(op->result_end()),
        useIt(),
        useEnd() {
    // Set useIt and useEnd to default for the end iterator
    useIt = useEnd = mlir::Value::use_iterator();
  }

  mlir::Operation* operator*() const {
    return useIt->getOwner();
  }

  OperationUsesIterator& operator++() {
    ++useIt;
    advanceToNextUse();
    return *this;
  }

  bool operator==(const OperationUsesIterator& other) const {
    return resultIt == other.resultIt && useIt == other.useIt;
  }

  bool operator!=(const OperationUsesIterator& other) const {
    return !(*this == other);
  }
};

// Specialization of GraphTraits for mlir::Operation* to represent the def-use chain.
template <>
struct GraphTraits<mlir::Operation*> {
  using NodeRef = mlir::Operation*;
  using ChildIteratorType = OperationUsesIterator;

  static NodeRef getEntryNode(mlir::Operation* Op) { return Op; }

  static ChildIteratorType child_begin(NodeRef N) {
    return OperationUsesIterator(N);
  }

  static ChildIteratorType child_end(NodeRef N) {
    return OperationUsesIterator(N, /*isEnd=*/true);
  }
};

} // namespace llvm
