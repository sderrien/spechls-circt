//===- ScheduleDialectDialect.td - SpecHLS dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ScheduleDialect_DIALECT
#define ScheduleDialect_DIALECT

include "mlir/IR/OpBase.td"

def ScheduleDialect_Dialect : Dialect {
    let name = "ScheduleDialect";
    let summary = "A dialect for schedule passes.";
    let description = [{
        A dialect for schedule passes.
    }];
    let cppNamespace = "SpecHLS";
    let useDefaultTypePrinterParser = 1;
  }

// Base class for the operation in this dialect.
class ScheduleDialectOp<string mnemonic, list<Trait> traits = []> :
    Op<ScheduleDialect_Dialect, mnemonic, traits>;

#endif // ScheduleDialect_DIALECT