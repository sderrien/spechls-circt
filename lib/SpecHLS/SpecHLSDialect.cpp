//===- SpecHLSDialect.cpp - SpecHLS dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SpecHLS/SpecHLSDialect.h"
#include "Scheduling/SchedulingProperty.h"
#include "SpecHLS/SpecHLSOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace SpecHLS;

//===----------------------------------------------------------------------===//
// SpecHLS dialect.
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Scheduling/SchedulingProperty.cpp.inc"
#undef GET_ATTRDEF_CLASSES

void SpecHLSDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SpecHLS/SpecHLSOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Scheduling/SchedulingProperty.cpp.inc"
      >();
}