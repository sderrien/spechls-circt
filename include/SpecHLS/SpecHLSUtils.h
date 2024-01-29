//===- SpecHLSOps.h - SpecHLS dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Operation.h"

namespace SpecHLS {

  bool isControlLogicOperation(mlir::Operation *op);

  std::string getPragma(mlir::Operation *op) ;

}