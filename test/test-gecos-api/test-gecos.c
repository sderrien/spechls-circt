
#include <mlir-c/IR.h>
#include "mlir-c/Conversion.h"
#include <mlir-c/AffineExpr.h>
#include <mlir-c/AffineMap.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Diagnostics.h>
#include <mlir-c/Dialect/Arith.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/Transform.h>
#include <mlir-c/Dialect/SCF.h>

#include <circt-c/Dialect/SV.h>
#include <circt-c/Dialect/HW.h>
#include <circt-c/Dialect/Seq.h>
#include <circt-c/Dialect/Comb.h>
#include <circt-c/Dialect/HWArith.h>
#include <circt-c/Dialect/FSM.h>

#include "CAPI/SpecHLS.h"

#include <mlir-c/IntegerSet.h>
#include <mlir-c/RegisterEverything.h>
#include <mlir-c/Support.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern MlirModule parseMLIR(const char* mlir);
extern void registerAllUpstreamDialects(MlirContext ctx) ;
char* mlirOperationToString(MlirOperation b);
void printMlirIdentifier (MlirIdentifier a);
const char mlir[]= {
  "module {\n"\
  "  func.func @add(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {\n"
  "    %test = SpecHLS.init \"exit11\" : i1 "\
  "    %res = arith.addi %arg0, %arg0 : i32\n"\
  "    return %res : i32\n"\
  "  }\n"\
  "}"
};


void traverseMLIROperation(MlirOperation op);
void traverseMLIRBlock(MlirBlock blk) ;
void traverseMLIRRegion(MlirRegion blk) ;


void traverseMLIRBlock(MlirBlock blk) {
  if (blk.ptr!=NULL) {
    MlirOperation op = mlirBlockGetFirstOperation(blk);
    while (op.ptr!=NULL) {
      traverseMLIROperation(op);
      op = mlirOperationGetNextInBlock(op);
    }
  }
}

void traverseMLIRRegion(MlirRegion region) {
  if (region.ptr!=NULL) {
    traverseMLIRBlock(mlirRegionGetFirstBlock(region));
  }
}
void traverseMLIROperation(MlirOperation op) {
  if (op.ptr!=NULL) {


    MlirStringRef identStr = mlirIdentifierStr(mlirOperationGetName(op));
    printf("Hello %s\n",identStr.data);
    printf("Numattributes = %ld\n",mlirOperationGetNumAttributes(op));

    for (int i=0;i< mlirOperationGetNumAttributes(op);i++) {
      MlirNamedAttribute namedAttr = mlirOperationGetAttribute(op,i);
      MlirStringRef identStr = mlirIdentifierStr(namedAttr.name);
      printf("attrname = %s\n",identStr.data);
      if (namedAttr.attribute.ptr!=NULL)
        mlirAttributeDump(namedAttr.attribute);
    }
    for (int i=0;i< mlirOperationGetNumRegions(op);i++) {
      MlirRegion region = mlirOperationGetRegion(op,i);
      traverseMLIRRegion(region);
    }

  }

}

int traverseMLIRModule(MlirModule module) {
  // Assuming we are given a module, go to the first operation of the first
  // function.
  if (module.ptr==NULL) printf("Error %p\n",module.ptr);


  //printf("traverse IR %p\n",module.ptr);

  MlirOperation moduleOp = mlirModuleGetOperation(module);
  traverseMLIROperation(moduleOp);
  //printf("traverse IR %p\n",moduleOp.ptr);
  if (moduleOp.ptr!=NULL) {
    MlirIdentifier ident = mlirOperationGetName(moduleOp);

    MlirStringRef identStr = mlirIdentifierStr(ident);
    printf("Hello %s\n",identStr.data);
    printf("Numattributes = %d\n",mlirOperationGetNumAttributes(moduleOp));

    for (int i=0;i< mlirOperationGetNumAttributes(moduleOp);i++) {
      MlirNamedAttribute namedAttr = mlirOperationGetAttribute(moduleOp,i);
      MlirStringRef identStr = mlirIdentifierStr(namedAttr.name);
      printf("attrname = %s\n",identStr.data);
      if (namedAttr.attribute.ptr!=NULL)
        mlirAttributeDump(namedAttr.attribute);
    }
    char* str = mlirOperationToString(moduleOp);
    printf("%s",str);
  }
}



int main(int argc, char **argv) {

  //registerAllUpstreamDialects();
  //fwrite(mlir,strlen(mlir),1,stdout);
  MlirModule m = parseMLIR(mlir);
  traverseMLIRModule(m);
  //char * str = mlirOperationToString()

//
//  try {
//                        loadNative
//
//				}
//			''')
//
//			var op = m.operation
//			println(traverse(op))
//
//  } catch (Throwable t) {
//                        t.printStackTrace();
//  }

}
