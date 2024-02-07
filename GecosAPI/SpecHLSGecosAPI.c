

#include "mlir-c/Conversion.h"
#include <mlir-c/AffineExpr.h>
#include <mlir-c/AffineMap.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Diagnostics.h>
#include <mlir-c/Dialect/Arith.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/SCF.h>
#include <mlir-c/Dialect/Transform.h>
#include <mlir-c/IR.h>

#include <circt-c/Dialect/Comb.h>
#include <circt-c/Dialect/FSM.h>
#include <circt-c/Dialect/HW.h>
#include <circt-c/Dialect/HWArith.h>
#include <circt-c/Dialect/SV.h>
#include <circt-c/Dialect/Seq.h>

#include <CAPI/SpecHLS.h>

#include <CAPI/SSP.h>

#include <mlir-c/IntegerSet.h>
#include <mlir-c/RegisterEverything.h>
#include <mlir-c/Support.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int isNull(void *p) { return (p == NULL); }

void registerAllUpstreamDialects(MlirContext ctx) {
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  // mlirRegisterAllDialects(registry);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__spechls__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__transform__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__func__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__arith__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__comb__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__seq__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__hw__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__sv__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__hwarith__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__scf__(), ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__ssp__(), ctx);

  mlirContextAppendDialectRegistry(ctx, registry);
  mlirDialectRegistryDestroy(registry);
}

bool mlirAttributeIsAArray(MlirAttribute attr);
bool mlirAttributeIsAString(MlirAttribute attr);

int traverseMLIR(MlirModule module);

void printMlirIdentifier(MlirIdentifier ident) {
  MlirStringRef identStr = mlirIdentifierStr(ident);
  //  printf("ident[%d] %s\n",identStr.length,identStr.data);
}

int getCStringSizeFromMlirIdentifier(MlirIdentifier ident) {
  MlirStringRef identStr = mlirIdentifierStr(ident);
  return identStr.length;
}

char *getCStringDataFromMlirIdentifier(MlirIdentifier ident) {
  MlirStringRef identStr = mlirIdentifierStr(ident);
  return identStr.data;
}

char *getCStringDataFromMlirStringRef(MlirStringRef ident) {
  return ident.data;
}

size_t getCStringSizeFromMlirStringRef(MlirStringRef ident) {
  printf("size =%lu", ident.length);
  return ident.length;
}

MlirIdentifier mlirNamedAttributeGetName(MlirNamedAttribute p) {
  return p.name;
}

MlirAttribute mlirNamedAttributeGetAttribute(MlirNamedAttribute p) {
  if (p.attribute.ptr == NULL)
    fprintf(stderr, "NULL %X->%X", p.attribute, p.attribute.ptr);
  return p.attribute;
}

#define BUFFERLEN 0x800000
char buffer[BUFFERLEN];
int offset = 0;

static void resetBuffer() { offset = 0; }

char getCharAt(char *v, int offset) { return v[offset]; }

int getOffset() { return offset; }

static void printToBuffer(MlirStringRef str, void *userData) {

  strcpy(&buffer[offset], str.data);
  // strncpy(&buffer[offset],str.data, BUFFERLEN-offset);
  offset += str.length;
}

char *mlirBlockToString(MlirBlock b) {
  resetBuffer();

  mlirBlockPrint(b, printToBuffer, NULL);
  return buffer;
}

char *mlirOperationToString(MlirOperation b) {
  resetBuffer();
  mlirOperationPrint(b, printToBuffer, NULL);
  buffer[offset] = '\0';
  return buffer;
}

char *mlirModuleToString(MlirModule b) {
  resetBuffer();
  mlirOperationToString(mlirModuleGetOperation(b));
  buffer[offset] = '\0';
  return buffer;
}

char *mlirValueToString(MlirValue b) {
  resetBuffer();
  mlirValuePrint(b, printToBuffer, NULL);
  buffer[offset] = '\0';
  return buffer;
}

char *mlirAttributeToString(MlirAttribute b) {
  resetBuffer();
  if (b.ptr != NULL) {
    // mlirAttributeDump(b);
    // fprintf(stderr,"Dump  %p\n",b.ptr);
    mlirAttributePrint(b, printToBuffer, NULL);
    // fprintf(stderr,"Print  %p\n",b.ptr);
    buffer[offset] = '\0';
    return buffer;
  } else {
    // fprintf(stderr,"Null  %p\n",b.ptr);
  }
}

MlirIdentifier mlirOperationGetAttributeNameAt(MlirOperation op, intptr_t pos) {
  //  if (pos>=0 && pos<mlirOperationGetNumAttributes(op)) {
  return mlirOperationGetAttribute(op, pos).name;
  //  } else {
  //    fprintf(stderr,"error at pos %d\n",pos);
  //    return (MlirIdentifier) {NULL,NULL};
  //  }
}

MlirAttribute mlirOperationGetAttributeAt(MlirOperation op, intptr_t pos) {

  return mlirOperationGetAttribute(op, pos).attribute;
}

// MlirNamedAttribute mlirOperationGetAttribute(MlirOperation op, intptr_t pos);

char *mlirPrintNamedAttribute(MlirNamedAttribute b) {
  fprintf(stderr, "%s->%s", getCStringDataFromMlirIdentifier(b.name),
          mlirAttributeToString(b.attribute));
}

char *mlirTypeToString(MlirType b) {
  resetBuffer();
  mlirTypePrint(b, printToBuffer, NULL);
  buffer[offset] = '\0';
  return buffer;
}
// In the module we created, the first operation of the first function is
// an "memref.dim", which has an attribute and a single result that we can
// use to test the printing mechanism.

// CHECK-LABEL: Running test 'testSimpleExecution'
MlirModule parseMLIR(const char *mlir) {
  MlirContext ctx = mlirContextCreate();
  registerAllUpstreamDialects(ctx);
  // printf("C side : context %p\n", ctx.ptr);

  // printf("Input %s", mlir);

  MlirStringRef str = mlirStringRefCreateFromCString(mlir);
  printf("C side : str %p,%d -> %s\n", str.data, str.length, str.data);

  MlirModule module = mlirModuleCreateParse(ctx, str);
  printf(stderr, "C side : module %p\n", module.ptr);

  return module;
}

void foobar12(MlirModule module) {
  // printf("Im there");
}

static void printToStderr(MlirStringRef str, void *userData) {
  (void)userData;
  fwrite(str.data, 1, str.length, stderr);
}

void traverseBlock(MlirBlock block);
void traverseOp(MlirOperation operation);
void traverseRegion(MlirRegion region);

int traverseModule(MlirModule m) {
  // Assuming we are given a module, go to the first operation of the first
  // function.
  printf("start of TraverseModule \n ");
  MlirOperation op = mlirModuleGetOperation(m);
  traverseOp(op);
  printf("end of TraverseModule \n ");

  return 0;
}

void traverseOp(MlirOperation op) {
  if (op.ptr != NULL) {
    MlirIdentifier ident = mlirOperationGetName(op);
    if (ident.ptr != NULL) {
      printf("Operation %s\n", getCStringDataFromMlirIdentifier(ident));
      // printMlirIdentifier(ident);
      //   mlirOperationToString(op);
      printf("\n");
    }

    int num = mlirOperationGetNumRegions(op);
    for (int i = 0; i < num; i++) {
      MlirRegion region = mlirOperationGetRegion(op, i);
      traverseRegion(region);
    }
    fflush(stdout);
  } else {
    printf("null op\n");
  }
}

void traverseRegion(MlirRegion region) {
  if (region.ptr != NULL) {
    printf("regions \n");
    MlirBlock block = mlirRegionGetFirstBlock(region);
    while (block.ptr != NULL) {
      traverseBlock(block);
      block = mlirBlockGetNextInRegion(block);
    }
    fflush(stdout);
  }
}

void traverseBlock(MlirBlock block) {
  if (block.ptr != NULL) {
    printf("Block \n");
    MlirOperation op = mlirBlockGetFirstOperation(block);
    while (op.ptr != NULL) {
      traverseOp(op);
      op = mlirOperationGetNextInBlock(op);
    }
    fflush(stdout);
  }
}

void traverseRegionLegacy(MlirRegion region) {

  if (region.ptr != NULL) {
    MlirBlock block = mlirRegionGetFirstBlock(region);
    printf("block %p\n", block.ptr);
    while (block.ptr != NULL) {
      MlirOperation op = mlirBlockGetFirstOperation(block);
      printf("operation %p\n", op.ptr);
      while (op.ptr != NULL) {
        MlirIdentifier ident = mlirOperationGetName(op);
        printf("ident %p\n", ident.ptr);
        printMlirIdentifier(ident);
        printf("\n");
        int num = mlirOperationGetNumRegions(op);
        for (int i = 0; i < num; i++) {
          region = mlirOperationGetRegion(op, i);
          traverseRegion(region);
        }
        op = mlirOperationGetNextInBlock(op);
      }
      block = mlirBlockGetNextInRegion(block);
      printf("next block %p\n", block.ptr);
    }
  }
}

// CHECK-LABEL: Running test 'testSimpleExecution'
void pass(const char *mlir) {
  MlirContext ctx = mlirContextCreate();
  registerAllUpstreamDialects(ctx);

  MlirModule module =
      mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(mlir));
  MlirOperation moduleOp = mlirModuleGetOperation(module);
  // printFirstOfEach(ctx,moduleOp);

  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
}