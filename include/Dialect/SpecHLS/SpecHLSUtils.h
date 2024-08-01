//===- SpecHLSOps.h - SpecHLS dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Operation.h"
#include "circt/Dialect/HW/HWOps.h"

namespace SpecHLS {

  bool isControlLogicOperation(mlir::Operation *op);
  bool hasControlNodePragma( mlir::Operation *op);
  bool hasPragmaContaining(mlir::Operation *op, llvm::StringRef keyword);
  std::string getPragma(mlir::Operation *op) ;
  bool hasConstantOutputs(circt::hw::HWModuleOp op);
  void removePragmaAttr(mlir::Operation * op, llvm::StringRef name);
  void setPragmaAttr(mlir::Operation * op, mlir::StringAttr value);
}
