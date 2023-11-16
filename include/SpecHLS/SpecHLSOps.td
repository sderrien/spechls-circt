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
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/EnumAttr.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// Include the definition of the necessary tablegen constructs for defining
// our attributes.
include "mlir/IR/AttrTypeBase.td"



def SpecHLS_GammaOp : SpecHLSOp<"gamma", [Pure]> {
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

 
    let hasCustomAssemblyFormat = 1;
    //let assemblyFormat = "attr-dict  `(` $select `:` type($select) `,` ($inputs^ `:` type($inputs))? `)` `:`  type($res)";
}

def SpecHLS_AlphaOp : SpecHLSOp<"alpha", [Pure,AttrSizedOperandSegments,
    TypesMatchWith<"value type must match element type of 'memref'",
                     "memref", "value",
                     "::llvm::cast<mlir::MemRefType>($_self).getElementType()">,
    TypesMatchWith<"result type must match input 'memref' type",
                     "memref", "result",
                     "($_self)">
]> {
    let summary = "Array update operation.";
    let description = [{}];

    let arguments = (ins Arg<AnyMemRef, "the reference to update",[MemRead]>:$memref,
                        AnyType:$value,
                        Variadic<Index>:$indices, Optional<I1>:$we);
    let results = (outs AnyMemRef:$result);

//  let hasCustomAssemblyFormat =1;

    let assemblyFormat = [{
    $we `->` $memref `[` $indices `]` `,` $value attr-dict `:` type($memref)
    }];
}

def SpecHLS_MuOp : SpecHLSOp<"mu", [Pure,SameOperandsAndResultType,
    TypesMatchWith<"mu inputs type must match'",
                     "next", "init",
                     "$_self">
]> {
    let summary = "Array update operation.";
    let description = [{}];

   let arguments = (ins AnyType:$next,AnyType:$init);
   let results = (outs AnyType:$result);

 let assemblyFormat = [{
     $next `,` $init attr-dict `:` type($init)
 }];

}

def SpecHLS_DelayOp : SpecHLSOp<"delay", [AttrSizedOperandSegments,
    TypesMatchWith<"Delay data must match'", "result", "next", "$_self">,
    TypesMatchWith<"Delay data must match'", "result", "init", "$_self">
]> {
    let summary = "Delay operation.";
    let description = [{}];

   let arguments = (ins AnyType:$next,Optional<I1>:$enable,Optional<AnyType>:$init);
   let results = (outs AnyType:$result);

   let assemblyFormat = [{
        $enable `->` $next `(` $init `)` `:` type($result) attr-dict
   }];

//  let hasCanonicalizer = 1;
//  let hasFolder = 1;
//  let hasCustomAssemblyFormat =1;
//  let assemblyFormat = "attr-dict  `(` $array `:` type($array) $value type($we) $we `:` type($value) `,` ($indices^ `:` type($indices))? `)` `:`  type($res)";

}





def SpecHLS_LookUpTableOp : SpecHLSOp<"lookUpTable", [Pure]> {
    let summary = "Array update operation.";
    let description = [{}];

   let arguments = (ins AnySignlessInteger:$input, I32ArrayAttr:$content);
   let results = (outs AnyInteger:$result);

//  let hasCanonicalizer = 1;
//  let hasFolder = 1;
   let hasCustomAssemblyFormat =1;
}



def SpecHLS_DontCareOp : SpecHLSOp<"dontCare", [Pure]> {
    let summary = "Array update operation.";
    let description = [{}];

   let arguments = (ins );
   let results = (outs AnyInteger:$result);

     let assemblyFormat = [{
         `:` type($result)  attr-dict
     }];


}

def SpecHLS_ExitOp : SpecHLSOp<"exit", []> {
    let summary = "Exit.";
    let description = [{}];

   let arguments = (ins I1:$finished, Variadic<AnyType>:$inputs);
   let results = (outs I1:$end );

  // let hasCustomAssemblyFormat =1;
     let assemblyFormat = [{
         $finished `:` `live` $inputs `:` type($inputs)   attr-dict
     }];
}

  


#endif // SpecHLS_OPS
