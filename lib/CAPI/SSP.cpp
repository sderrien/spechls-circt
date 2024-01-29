//===- Seq.cpp - C interface for the Seq dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/SSP.h"
#include "circt/Dialect/SSP/SSPDialect.h"
#include "circt/Dialect/SSP/SSPPasses.h"

#include "mlir/CAPI/Registration.h"

extern "C" {

// bool mlirAttributeIsAArray(MlirAttribute attr) {
//   return llvm::isa<mlir::ArrayAttr>(unwrap(attr));
// }

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Scheduling, ssp, circt::ssp::SSPDialect)
void registerSSPPasses() { circt::ssp::registerPasses(); }

}



