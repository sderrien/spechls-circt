//
// Created by Steven on 28/10/2023.
//

#ifndef SPECHLS_DIALECT_SPECHLS_H
#define SPECHLS_DIALECT_SPECHLS_H

//===-- spechls-c/SpecHLS.h - C API for Comb dialect -----------*- C -*-===//
//
// This header declares the C interface for registering and accessing the
// Comb dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

//#include "mlir/IR/Dialect.h"

#include "mlir-c/IR.h"

//#include "SpecHLS/SpecHLSOpsDialect.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(SpecHLS, spechls);

#ifdef __cplusplus
}
#endif


#endif // SPECHLS_DIALECT_SPECHLS_H
