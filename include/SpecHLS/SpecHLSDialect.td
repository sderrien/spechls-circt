//===- SpecHLSDialect.td - SpecHLS dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SpecHLS_DIALECT
#define SpecHLS_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// SpecHLS dialect definition.
//===----------------------------------------------------------------------===//

def SpecHLS_Dialect : Dialect {
    let name = "SpecHLS";
    let summary = "A SpecHLS out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::SpecHLS";
}

//===----------------------------------------------------------------------===//
// Base SpecHLS operation definition.
//===----------------------------------------------------------------------===//

class SpecHLS_Gamma<string mnemonic, list<Trait> traits = []> :
        Op<SpecHLS_Dialect, mnemonic, traits>;

#endif // SpecHLS_DIALECT
