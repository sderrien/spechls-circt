
#include <mlir-c/Conversion.h>
#include <mlir-c/IR.h>
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

//#include <mlir/Dialect/Utils.h>

#include <mlir-c/IntegerSet.h>
//#include <mlir-c/RegisterEverything.h>
#include <mlir-c/Support.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


  static void registerAllUpstreamDialects(MlirContext ctx) {
    MlirDialectRegistry registry = mlirDialectRegistryCreate();
    //mlirRegisterAllDialects(registry);
    mlirDialectHandleRegisterDialect(mlirGetDialectHandle__transform__(), ctx);
    mlirDialectHandleRegisterDialect(mlirGetDialectHandle__func__(), ctx);
    mlirDialectHandleRegisterDialect(mlirGetDialectHandle__arith__(), ctx);
    mlirDialectHandleRegisterDialect(mlirGetDialectHandle__comb__(), ctx);
    mlirDialectHandleRegisterDialect(mlirGetDialectHandle__seq__(), ctx);
    mlirDialectHandleRegisterDialect(mlirGetDialectHandle__hw__(), ctx);
    mlirDialectHandleRegisterDialect(mlirGetDialectHandle__sv__(), ctx);
    mlirDialectHandleRegisterDialect(mlirGetDialectHandle__hwarith__(), ctx);
    mlirDialectHandleRegisterDialect(mlirGetDialectHandle__scf__(), ctx);
    mlirDialectHandleRegisterDialect(mlirGetDialectHandle__spechls__(), ctx);

    mlirContextAppendDialectRegistry(ctx, registry);
    mlirDialectRegistryDestroy(registry);
}

void traverseMLIR(MlirModule module);

void printMlirIdentifier(MlirIdentifier ident) {
  MlirStringRef identStr = mlirIdentifierStr(ident);
  printf("ident[%d] %s\n",identStr.length,identStr.data);
}


// CHECK-LABEL: Running test 'testSimpleExecution'
MlirModule parseMLIR(const char* mlir) {
  MlirContext ctx = mlirContextCreate();
  registerAllUpstreamDialects(ctx);
  printf("C side : context %p\n", ctx.ptr);

  printf("Input %s", mlir);

  MlirStringRef str = mlirStringRefCreateFromCString(mlir);
  printf("C side : str %p,%d -> %s\n", str.data, str.length,str.data);

  MlirModule module = mlirModuleCreateParse(ctx, str);
  printf("C side : module %p\n", module.ptr);

  return module;
}


void foobar12(MlirModule module) {
  printf("Im there");
}


void traverseRegion( MlirRegion region) {
  printf("traverse region %p\n",region.ptr);

  if (region.ptr!=NULL) {
    MlirBlock block = mlirRegionGetFirstBlock(region);
    printf("block %p\n",block.ptr);
    while (block.ptr !=NULL) {
      MlirOperation op = mlirBlockGetFirstOperation(block);
      printf("operation %p\n",op.ptr);
      while (op.ptr !=NULL) {
        MlirIdentifier ident = mlirOperationGetName(op);
        printf("ident %p\n",ident.ptr);
        printMlirIdentifier(ident);
        printf("\n");
        int num = mlirOperationGetNumRegions(op);
        for (int i=0;i<num;i++) {
          region = mlirOperationGetRegion(op, i);
          traverseRegion(region);
        }
        op = mlirOperationGetNextInBlock(op);
      }
      block = mlirBlockGetNextInRegion(block);
      printf("next block %p\n",block.ptr);
    }
  }

}

void traverseMLIR(MlirModule module) {
  // Assuming we are given a module, go to the first operation of the first
  // function.
  printf("traverse IR %p\n",module.ptr);

  MlirOperation moduleOp = mlirModuleGetOperation(module);
  printf("traverse IR %p\n",moduleOp.ptr);
  if (moduleOp.ptr!=NULL) {
    MlirIdentifier ident = mlirOperationGetName(moduleOp);
    printMlirIdentifier(ident);


    MlirRegion region = mlirOperationGetRegion(moduleOp, 0);
    traverseRegion(region);
  }
}


// CHECK-LABEL: Running test 'testSimpleExecution'
void pass(const char* mlir) {
  MlirContext ctx = mlirContextCreate();
  registerAllUpstreamDialects(ctx);

  MlirModule module = mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(mlir));
  MlirOperation moduleOp = mlirModuleGetOperation(module);
  //printFirstOfEach(ctx,moduleOp);

  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
}

static void printToStderr(MlirStringRef str, void *userData) {
  (void)userData;
  fwrite(str.data, 1, str.length, stderr);
}


static void printFirstOfEach(MlirContext ctx, MlirOperation operation) {
  // Assuming we are given a module, go to the first operation of the first
  // function.
  MlirRegion region = mlirOperationGetRegion(operation, 0);
  MlirBlock block = mlirRegionGetFirstBlock(region);
  while (block.ptr !=NULL) {
    MlirOperation op = mlirBlockGetFirstOperation(block);
    while (op.ptr !=NULL) {

      op = mlirOperationGetNextInBlock(op);
    }
    block = mlirBlockGetNextInRegion(block);
  }

  region = mlirOperationGetRegion(operation, 0);
  MlirOperation parentOperation = operation;
  block = mlirRegionGetFirstBlock(region);
  operation = mlirBlockGetFirstOperation(block);
  assert(mlirModuleIsNull(mlirModuleFromOperation(operation)));

  fprintf(stderr, "Parent operation eq: %d\n",
          mlirOperationEqual(mlirOperationGetParentOperation(operation),
                             parentOperation));

}

