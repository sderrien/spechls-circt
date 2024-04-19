//===- SpecHLSDialect.cpp - SpecHLS dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace SpecHLS;

//===----------------------------------------------------------------------===//
// SpecHLS dialect.
//===----------------------------------------------------------------------===//

void SpecHLSDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/SpecHLS/SpecHLSOps.cpp.inc"
      >();
}

Operation *SpecHLSDialect::materializeConstant(OpBuilder &builder,
                                               Attribute value, Type type,
                                               Location loc) {
  auto coeffs = dyn_cast<mlir::IntegerAttr>(value);
  if (!coeffs)
    return nullptr;
  return builder.create<circt::hw::ConstantOp>(loc, type, coeffs);
}