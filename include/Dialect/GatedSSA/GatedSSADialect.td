//===- GatedSSADialect.td - GatedSSA dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GatedSSA_DIALECT
#define GatedSSA_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// GatedSSA dialect definition.
//===----------------------------------------------------------------------===//

def GatedSSA_Dialect : Dialect {
    let name = "GatedSSA";
    let summary = "A Dialect for Gated SSA.";
    let description = [{
        This dialect is tu support Gated SSA in MLIR .
    }];
    let cppNamespace = "GatedSSA";
    let hasConstantMaterializer = 1;
    let useDefaultTypePrinterParser = 1;

  }

//===----------------------------------------------------------------------===//
// Base GatedSSA operation definition.
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect.
class GatedSSAOp<string mnemonic, list<Trait> traits = []> :
    Op<GatedSSA_Dialect, mnemonic, traits>;

#endif // GatedSSA_DIALECT