//===- ScheduleDialectDialect.cpp - SpecHLS dialect ---------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/ScheduleDialect/ScheduleDialectDialect.h"
#include "Dialect/ScheduleDialect/ScheduleDialectOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace SpecHLS;

void ScheduleDialectDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/ScheduleDialect/ScheduleDialectOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/ScheduleDialect/ScheduleDialectOpsTypes.cpp.inc"
      >();
}