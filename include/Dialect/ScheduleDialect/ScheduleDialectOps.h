//===- SpecHLSOps.h - SpecHLS dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SpecHLS_ScheduleDialectOPS_H
#define SpecHLS_ScheduleDialectOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/ScheduleDialect/ScheduleDialectDialect.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/ScheduleDialect/ScheduleDialectOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "Dialect/ScheduleDialect/ScheduleDialectOps.h.inc"

#endif // SpecHLS_ScheduleDialectOPS_H