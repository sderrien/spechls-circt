//===- ScheduleDialectOps.cpp - SpecHLS dialect ---------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/ScheduleDialect/ScheduleDialectOps.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/ScheduleDialect/ScheduleDialectOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/ScheduleDialect/ScheduleDialectOps.cpp.inc"