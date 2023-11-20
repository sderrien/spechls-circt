
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

#include <CAPI/SpecHLS.h>

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
  "  func.func @add(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {\n"\
  "    %res = arith.addi %arg0, %arg0 : i32\n"\
  "    return %res : i32\n"\
  "  }\n"\
  "}"
};

int traverseMLIR(MlirModule module) {
  // Assuming we are given a module, go to the first operation of the first
  // function.
  if (module.ptr==NULL) printf("Error %p\n",module.ptr);


  //printf("traverse IR %p\n",module.ptr);

  MlirOperation moduleOp = mlirModuleGetOperation(module);
  //printf("traverse IR %p\n",moduleOp.ptr);
  if (moduleOp.ptr!=NULL) {
    MlirIdentifier ident = mlirOperationGetName(moduleOp);
    printMlirIdentifier(ident);


    char* str = mlirOperationToString(moduleOp);
    printf("%s",str);
  }
}



int main(int argc, char **argv) {

  //registerAllUpstreamDialects();
  //fwrite(mlir,strlen(mlir),1,stdout);
  MlirModule m = parseMLIR(mlir);
  traverseMLIR(m);
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
