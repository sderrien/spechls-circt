//===- SpecHLSOps.td - SpecHLS dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SpecHLS_OPS
#define SpecHLS_OPS

include "SpecHLSDialect.td"
include "/mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SpecHLS_GammaOp : SpecHLS_Gamma<"gamma", [Pure]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `SpecHLS.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = SpecHLS.foo %0 : i32
        ```
    }];

    let arguments = (ins SignlessIntegerLike:$select, Variadic<AnyType>:$inputs);
    let results = (outs AnyType:$res);

    let builders = [
        OpBuilder<(ins), [{
        /* nothing to do */ }]>,
    ];
    let assemblyFormat = "attr-dict  `(` $select `:` type($select) `,` ($inputs^ `:` type($inputs))? `)` `:`  type($res)";
}

def SpecHLS_AlphaOp : SpecHLS_Alpha<"alpha", [Pure]> {
    let summary = "Array update operation.";
    let description = [{
        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = SpecHLS.foo %0 : i32
        ```
    }];

    let arguments = (ins AnyMemRef:$array, AnyType:$value, I1:$we,Variadic<AnyInteger>:$indices);
    let results = (outs AnyMemRef:$res);

    let builders = [
        OpBuilder<(ins), [{
        /* nothing to do */ }]>,
    ];
    let assemblyFormat = "attr-dict  `(` $array `:` type($array) $value type($we) $we `:` type($value) `,` ($indices^ `:` type($indices))? `)` `:`  type($res)";
}

def SpecHLS_DelayOp : SpecHLS_Delay<"delay", [Pure]> {
    let summary = "Delay node operation.";
    let description = [{
        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = SpecHLS.foo %0 : i32
        ```
    }];

    let arguments = (ins AnyType:$d, I1:$we,  AnyType:$init);
    let results = (outs AnyType:$q);

    let builders = [
        OpBuilder<(ins), [{
        /* nothing to do */ }]>,
    ];
    let assemblyFormat = "attr-dict  `(` $d `:` type($d) `,` $we `,` $init `:` type($init) `)` `:` type($q)";
}



#endif // SpecHLS_OPS
