//===- InitAllDialects.h - CIRCT Dialects Registration ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialects and
// passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef SPECHLS_INITALLDIALECTS_H_
#define SPECHLS_INITALLDIALECTS_H_


#include "/IR/Dialect.h"

namespace SpecHLS {

// Add all the MLIR dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<
      SpecHLS::SpecHLSDialect
  >();
  // clang-format off
  // clang-format on
}

} // namespace spechls

#endif // SPECHLS_INITALLDIALECTS_H_
