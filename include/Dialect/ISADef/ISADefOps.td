//===- ISADeclOps.td - ISADecl dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ISADecl_OPS
#define ISADecl_OPS

include "ISADeclDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/EnumAttr.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// Include the definition of the necessary tablegen constructs for defining
// our attributes.
include "mlir/IR/AttrTypeBase.td"

def ISADecl_ISAOp : ISADeclOp<"instruction_set"> {
    let summary = "Defines an instruction set";
    let description = [{
        The `ISADecl.instruction_set` defines an instruction set.

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        ISADecl.instruction_set "RiscV" {
        }
        ```
    }];
    let regions = (region AnyRegion<1>:$instructions);

    let arguments = (ins StringAttribute:$name);
    let results = (outs );

    let hasCustomAssemblyFormat = 1;
}

def ISADecl_InsnOp : ISADeclOp<"instruction"> {
    let summary = "Illustrates how to define an instruction.";
    let description = [{
        The `ISADecl.instruction` operation defines a machine instruction with its encoding and its operational semantics.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = ISADecl.instruction ldw
            format {%op} {
                fom
            }
            opsem {

            }
        ```
    }];
    let regions = (region SizedRegion<1>:$format, SizedRegion<1>:$opsem);

    let arguments = (ins StringAttribute:$name, Variadic<AnyType>:$fields);
    let results = (outs );

    let hasCustomAssemblyFormat = 1;
}

#endif // ISADecl_OPS
