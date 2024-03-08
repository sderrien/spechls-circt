#ifndef MLIR_CAPI_IR_H
#define MLIR_CAPI_IR_H

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"


#define DEFINE_C_API_PTR_METHODS(mlirPortInfo, circt::hw::PortInfo)
DEFINE_C_API_METHODS(MlirAttribute, mlir::Attribute)