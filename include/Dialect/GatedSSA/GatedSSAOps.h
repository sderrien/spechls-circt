//===- GatedSSAOps.h - GatedSSA dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GatedSSA_GatedSSAOPS_H
#define GatedSSA_GatedSSAOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/TypeID.h"
#include "Dialect/GatedSSA/GatedSSATypes.h"

#define GET_OP_CLASSES
#include "Dialect/GatedSSA/GatedSSAOps.h.inc"

#endif // GatedSSA_GatedSSAOPS_H