//===- GatedSSAOps.td - GatedSSA dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GatedSSA_OPS
#define GatedSSA_OPS

include "Dialect/GatedSSA/GatedSSADialect.td"
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
include "mlir/IR/AttrTypeBase.td"

// include "circt/Dialect/HW/HWAttributes.td"
// include "circt/Dialect/HW/HWDialect.td"
// include "circt/Dialect/HW/HWOpInterfaces.td"
// include "circt/Dialect/HW/HWTypes.td"

// Include the definition of the necessary tablegen constructs for defining
// our attributes.
include "Dialect/GatedSSA/GatedSSATypes.td"
include "mlir/IR/AttrTypeBase.td"



def hasCustomSSAName : DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>;



// Include necessary MLIR definitions
include "mlir/IR/OpBase.td"
include "mlir/IR/StandardTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PredicatedRegionOp : Op<"predicated_region",
                            [SingleBlockImplicitTerminator<"cf.br">,
                             NoSideEffect]> {
  let summary = "An operation with a single block region predicated by a boolean value";
  let description = [{
    This operation represents a region that is executed based on a predicate boolean value.
  }];

  let arguments = (ins
    I1:$predicate // Predicate boolean value
  );

  let regions = (region anyregion);

  let results = (outs);

  let builders = [
    OpBuilder<(ins "Value":$predicate), [{
      auto *region = new Region();
      region->push_back(new Block());
      this->addRegion(region);
      build($0, $1, $predicate);
    }], [{
      auto *region = new Region();
      region->push_back(new Block());
      this->addRegion(region);
      build($0, $1, $predicate);
    }], [{
      auto *region = new Region();
      region->push_back(new Block());
      this->addRegion(region);
      build($0, $1, $predicate);
    }], [{
      auto *region = new Region();
      region->push_back(new Block());
      this->addRegion(region);
      build($0, $1, $predicate);
    }], [{
      auto *region = new Region();
      region->push_back(new Block());
      this->addRegion(region);
      build($0, $1, $predicate);
    }]];
}


def GatedSSA_GammaOp : GatedSSAOp<"gamma", [Pure,hasCustomSSAName]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `GatedSSA.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = GatedSSA.foo %0 : i32
        ```
    }];

    let arguments = (ins FlatSymbolRefAttr:$name, AnyInteger   :$select, Variadic<AnyType>:$inputs);
    let results = (outs AnyType:$res);



    let extraClassDefinition = [{
        void $cppClass::getAsmResultNames(llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn)
        {
            setNameFn(getResult(), "gamma");
        }
    }];
}


def GatedSSA_InitOp : GatedSSAOp<"init", [Pure]> {
    let summary = "Init.";
    let description = [{}];

   let arguments = (ins FlatSymbolRefAttr:$name);
   let results = (outs AnyType:$result );

  // let hasCustomAssemblyFormat =1;
     let assemblyFormat = [{
         $name `:` type($result)      attr-dict
     }];
}

def GatedSSA_DefOp : GatedSSAOp<"def", [
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

#endif // GatedSSA_OPS
