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
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


// Include the definition of the necessary tablegen constructs for defining
// our attributes.
include "mlir/IR/AttrTypeBase.td"


class SpecHLS_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SpecHLS_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}


def SpecHLS_SpecArrayType : SpecHLS_Type<"SpecArray", "specmemref", [
    ShapedTypeInterface
  ]> {
  let summary = "Shaped reference to a region of memory";

  let parameters = (ins

    "int64_t":$size,
    "Type":$elementType,
    "int64_t":$nbDiscardedWrites,
    ArrayRefParameter<"int64_t">:$nbPendingWrites,
    "int64_t":$maxPendingWrites,
    "int64_t":$maxPendingAddresses

  );

  let hasCustomAssemblyFormat =1;
}


def SpecHLS_SeqBlock : SpecHLSOp<"block", [Pure,SingleBlock]> {
    let arguments = (ins FlatSymbolRefAttr:$name, I1:$enable, Variadic<AnyType>:$inputs, I32Attr:$latency);
    let results = (outs Variadic<AnyType>:$outputs);
    let regions = (region SizedRegion<1>:$region);
    // let hasCanonicalizer = 1;
    // let hasFolder = 1;
    // let hasCustomAssemblyFormat = 1;
}

def SpecHLS_PipeBlock : SpecHLSOp<"block", [Pure,SingleBlock]> {
    let arguments = (ins FlatSymbolRefAttr:$name, I1:$enable, Variadic<AnyType>:$inputs, I32Attr:$II, I32Attr:$depth);
    let results = (outs Variadic<AnyType>:$outputs);
    let regions = (region SizedRegion<1>:$region);
    // let hasCanonicalizer = 1;
    // let hasFolder = 1;
    // let hasCustomAssemblyFormat = 1;
}

def SpecHLS_RootBlock : SpecHLSOp<"block", [Pure,SingleBlock]> {
    let arguments = (ins FlatSymbolRefAttr:$name);
    let results = (outs );

    let regions = (region SizedRegion<1>:$region);
    // let hasCanonicalizer = 1;
    // let hasFolder = 1;
    // let hasCustomAssemblyFormat = 1;
}


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

def SpecHLS_CastOp : SpecHLSOp<"cast", [Pure]> {
    let arguments = (ins AnyType:$input);
    let results = (outs AnyType:$res);

//    let hasCanonicalizer = 1;
    let hasFolder = 1;

    //let hasCustomAssemblyFormat = 1;
    let assemblyFormat = "$input `:` type($input) `to` type($res) attr-dict";
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
    let summary = "printf operation.";
    let description = [{}];

   let arguments = (ins StrAttr:$format,  AnyType:$state, I1:$enable,Variadic<AnyType>:$items);
   let results = (outs AnyType:$newstate);

   let hasCustomAssemblyFormat =1;

// let assemblyFormat = [{
//    $format `(` $items `:` type($items) `)` `from` $state `when` $enable attr-dict
// }];

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

    let hasVerifier = 1;
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

  let hasCustomAssemblyFormat =1;
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
