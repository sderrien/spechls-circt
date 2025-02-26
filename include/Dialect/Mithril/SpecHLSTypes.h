//===- SpecHLSTYPES.h - SpecHLS dialect TYPES -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SpecHLS_SpecHLSTYPES_H
#define SpecHLS_SpecHLSTYPES_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/TypeID.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/SpecHLS/SpecHLSOpsTypes.h.inc"

namespace SpecHLS {}
#endif // SpecHLS_SpecHLSTYPES_H
