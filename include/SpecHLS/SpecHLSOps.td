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
/*
// It's common to define a base classes for types in the same dialect. This
// removes the need to pass in the dialect for each type, and can also be used
// to define a few fields ahead of time.
class MyDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<My_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Here is a simple definition of an "integer" type, with a width parameter.
def My_IntegerType : MyDialect_Type<"Integer", "int"> {
  let summary = "Integer type with arbitrary precision up to a fixed limit";
  let description = [{
    Integer types have a designated bit width.
  }];
  /// Here we defined a single parameter for the type, which is the bitwidth.
  let parameters = (ins "unsigned":$width);

  /// Here we define the textual format of the type declaratively, which will
  /// automatically generate parser and printer logic. This will allow for
  /// instances of the type to be output as, for example:
  ///
  ///    !my.int<10> // a 10-bit integer.
  ///
  let assemblyFormat = "`<` $width `>`";

  /// Indicate that our type will add additional verification to the parameters.
  let genVerifyDecl = 1;
}

*/
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

    let arguments = (ins FlatSymbolRefAttr:$name, SignlessIntegerLike:$select, Variadic<AnyType>:$inputs);
    let results = (outs AnyType:$res);

    let hasCanonicalizer = 1;
    let hasFolder = 1;

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

    let arguments = (ins FlatSymbolRefAttr:$name, Arg<AnyMemRef, "the reference to update",[MemRead]>:$memref,
                        AnyType:$value,
                        Variadic<Index>:$indices, Optional<I1>:$we);
    let results = (outs AnyMemRef:$result);

    let assemblyFormat = [{
    $name `:` $we `->` $memref `[` $indices `]` `,` $value attr-dict `:` type($memref)
    }];
}

def SpecHLS_ArrayReadOp : SpecHLSOp<"read", [Pure,
    TypesMatchWith<"value type must match element type of 'memref'","memref", "result","::llvm::cast<mlir::MemRefType>($_self).getElementType()">,
]> {
    let summary = "Array read operation.";
    let description = [{}];

    let arguments = (ins Arg<AnyMemRef, "the array to read",[MemRead]>:$memref, Variadic<Index>:$indices);
    let results = (outs AnyType:$result);


    let assemblyFormat = [{
     $memref `:` type($memref)`[` $indices `]`   attr-dict
    }];
}

def SpecHLS_MuOp : SpecHLSOp<"mu", [Pure,SameOperandsAndResultType,
    TypesMatchWith<"mu inputs type must match'",
                     "next", "init",
                     "$_self">
]> {
    let summary = "Array update operation.";
    let description = [{}];

   let arguments = (ins FlatSymbolRefAttr:$name,AnyType:$next,AnyType:$init);
   let results = (outs AnyType:$result);

 let assemblyFormat = [{
    $name `:` $next `,` $init attr-dict `:` type($init)
 }];

}

def SpecHLS_RollbackOp : SpecHLSOp<"rollback", [Pure,
    TypesMatchWith<"data and result type must match'",
                     "data", "result","$_self">
]> {
    let summary = "Rollback operation.";
    let description = [{}];

   let arguments = (ins AnyType:$data,AnyInteger:$command,I32ArrayAttr:$depths);
   let results = (outs AnyType:$result);

//   let hasCustomAssemblyFormat =1;

 let assemblyFormat = [{
    $data `:` type($data) `by` $command`:` type($command) `in` $depths attr-dict
 }];

}

def SpecHLS_PrintOp : SpecHLSOp<"ioprintf", [Pure]> {
    let summary = "Rollback operation.";
    let description = [{}];

   let arguments = (ins StrAttr:$format,  I1:$state, I1:$enable,Variadic<AnyType>:$items);
   let results = (outs I1:$newstate);

//   let hasCustomAssemblyFormat =1;

 let assemblyFormat = [{
    $format `(` $items `:` type($items) `)` `from` $state `when` $enable attr-dict
 }];

}

def SpecHLS_EncoderOp : SpecHLSOp<"encode", [Pure
//,
//    TypesMatchWith<"data and result type must match'",
//                     "data", "result",
//                     "::llvm::cast<mlir::MemRefType>($_self).getElementType()
//                     std::ceil(std::log2(($_self).getWidth()))
//                     "
//                     >
]> {
    let summary = "encoder operation.";
    let description = [{}];

   let arguments = (ins AnyInteger:$data);
   let results = (outs AnyInteger:$result);



 let assemblyFormat = [{
    $data `:` type($data) `into` type($result)  attr-dict
 }];

}

def SpecHLS_DecoderOp : SpecHLSOp<"decode", [Pure
//,
//    TypesMatchWith<"data and result type must match'",
//                     "data", "result",
//                     "::llvm::cast<mlir::MemRefType>($_self).getElementType()
//                     std::ceil(std::log2(($_self).getWidth()))
//                     "
//                     >
]> {
    let summary = "encoder operation.";
    let description = [{}];

   let arguments = (ins AnyInteger:$data);
   let results = (outs AnyInteger:$result);



 let assemblyFormat = [{
    $data `:` type($data) `into` type($result)  attr-dict
 }];

}


def SpecHLS_DelayOp : SpecHLSOp<"delay", [AttrSizedOperandSegments,
    TypesMatchWith<"Delay data must match'", "result", "next", "$_self">
]> {
    let summary = "Delay operation.";
    let description = [{}];

   let arguments = (ins AnyType:$next,Optional<I1>:$enable,Optional<AnyType>:$init, I32Attr:$depth);
   let results = (outs AnyType:$result);

    let hasCustomAssemblyFormat =1;

}





def SpecHLS_LookUpTableOp : SpecHLSOp<"lookUpTable", [Pure]> {
    let summary = "Array update operation.";
    let description = [{}];

   let arguments = (ins AnySignlessInteger:$input, I32ArrayAttr:$content);
   let results = (outs AnyInteger:$result);

  //let hasCanonicalizer = 1;
    let hasFolder = 1;
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

   let arguments = (ins I1:$finished, Variadic<AnyType>:$liveout);
   let results = (outs I1:$end );

   let hasCustomAssemblyFormat =1;
}

  

def SpecHLS_SyncOp : SpecHLSOp<"sync", [Pure,
    TypesMatchWith<"Sync data/result types must match'", "result", "data", "$_self">]> {
    let summary = "Sync.";
    let description = [{}];

   let arguments = (ins AnyType:$data, Variadic<AnyType>:$others);
   let results = (outs AnyType:$result );

  // let hasCustomAssemblyFormat =1;
     let assemblyFormat = [{
         $data `:` type($result) `,`  $others `:` type($others)     attr-dict
     }];
}

def SpecHLS_StringOp : SpecHLSOp<"string", [Pure]> {
    let summary = "constant.";
    let description = [{}];

   let arguments = (ins StrAttr:$value);
   let results = (outs AnyType:$result );

  // let hasCustomAssemblyFormat =1;
     let assemblyFormat = [{
         $value `:` type($result) attr-dict
     }];
}

def SpecHLS_InitOp : SpecHLSOp<"init", [Pure]> {
    let summary = "Init.";
    let description = [{}];

   let arguments = (ins FlatSymbolRefAttr:$name);
   let results = (outs AnyType:$result );

  // let hasCustomAssemblyFormat =1;
     let assemblyFormat = [{
         $name `:` type($result)      attr-dict
     }];
}

def SpecHLS_DefOp : SpecHLSOp<"def", [
    Pure,
    TypesMatchWith<"Def data/result types must match'", "result", "data", "$_self">
]> {

    let summary = "Def.";
    let description = [{}];

   let arguments = (ins AnyType:$data, FlatSymbolRefAttr:$name);
   let results = (outs AnyType:$result );

  // let hasCustomAssemblyFormat =1;
     let assemblyFormat = [{
          $name $data `:` type($result)      attr-dict
     }];
}

#endif // SpecHLS_OPS
