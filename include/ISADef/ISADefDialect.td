//===- ISADeclDialect.td - ISADecl dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ISADecl_DIALECT
#define ISADecl_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ISADecl dialect definition.
//===----------------------------------------------------------------------===//

def ISADecl_Dialect : Dialect {
    let name = "ISADecl";
    let summary = "A Dialect for defining ISA .";
    let description = [{
        This dialect is to allow the definition of ISA in MLIR/CIRCT .
    }];
    let cppNamespace = "ISADecl";
  }

//===----------------------------------------------------------------------===//
// Base ISADecl operation definition.
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect.
class ISADeclOp<string mnemonic, list<Trait> traits = []> :
    Op<ISADecl_Dialect, mnemonic, traits>;

#endif // ISADecl_DIALECT
