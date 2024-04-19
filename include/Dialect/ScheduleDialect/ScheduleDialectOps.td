//===- ScheduleDialectDialect.td - SpecHLS dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ScheduleDialect_OPS
#define ScheduleDialect_OPS

include "Dialect/ScheduleDialect/ScheduleDialectDialect.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinLocationAttributes.td"

def ScheduleOperation_Type
    : TypeDef<ScheduleDialect_Dialect, "ScheduleOperation", []> {
  let mnemonic = "ScheduleOperation";
}

def ScheduleDialect_circuit : ScheduleDialectOp<"circuit", [RegionKindInterface, NoTerminator, IsolatedFromAbove]> {
    let summary = "Represent the circuit to schedule.";
    let description = [{
        Represent the circuit to schedule.
    }];

    let regions = (region SizedRegion<1>:$opsRegion);
    let arguments = (ins Builtin_FloatAttr:$targetClock);
    let results = (outs );


  code extraModuleClassDeclaration = [{

    // Implement RegionKindInterface.
    static mlir::RegionKind getRegionKind(unsigned index) { return mlir::RegionKind::Graph;}

    }];

    let extraClassDeclaration = extraModuleClassDeclaration # [{}];

    let assemblyFormat = [{ attr-dict $targetClock $opsRegion}];
}


def ScheduleDialect_operation : ScheduleDialectOp<"operation", [HasParent<"circuit">]> {
    let summary = "Represent an operation to schedule.";
    let description = [{
        Represent an operation to schedule.
    }];

    let arguments = (ins SymbolNameAttr:$sym_name,
                         Variadic<ScheduleOperation_Type>:$dependences,
                         ArrayAttr:$distances,
    Builtin_FloatAttr:$inDelay, Builtin_IntegerAttr:$latency, Builtin_FloatAttr:$outDelay);
    let results = (outs ScheduleOperation_Type:$res);

    let assemblyFormat = [{
        attr-dict $sym_name `(` $dependences `:` $distances `,` $inDelay `,` $latency `,` $outDelay `)`
    }];
}

#endif // ScheduleDialect_OPS