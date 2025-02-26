//
// Created by Steven on 14/10/2024.
//

#ifndef SPECHLS_DIALECT_TESTCONVERSION_H
#define SPECHLS_DIALECT_TESTCONVERSION_H

#include "AffineToISLConverter.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include <isl/isl.h>

int main() {
  // Initialize MLIR context
  mlir::MLIRContext mlirCtx;
  mlirCtx.loadDialect<mlir::AffineDialect>();

  // Create an example AffineMap: (d0, d1)[s0] -> (d0 + 2*d1 + s0)
  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &mlirCtx);
  mlir::AffineExpr d1 = mlir::getAffineDimExpr(1, &mlirCtx);
  mlir::AffineExpr s0 = mlir::getAffineSymbolExpr(0, &mlirCtx);

  mlir::AffineExpr expr = d0 + 2 * d1 + s0;
  mlir::AffineMap affineMap = mlir::AffineMap::get(2, 1, expr);

  // Initialize ISL context
  isl_ctx *islCtx = isl_ctx_alloc();

  // Create converter
  AffineToISLConverter converter(islCtx);

  // Convert AffineMap to isl_map
  isl_map *islMap = converter.convertAffineMapToISL(affineMap);

  if (islMap) {
    char *islMapStr = isl_map_to_str(islMap);
    std::cout << "Converted isl_map: " << islMapStr << std::endl;

    // Convert back to AffineMap
    mlir::AffineMap convertedAffineMap = converter.convertISLMapToAffineMap(islMap, &mlirCtx);
    std::cout << "Converted back to AffineMap: ";
    convertedAffineMap.print(llvm::outs());
    std::cout << std::endl;

    isl_map_free(islMap);
    free(islMapStr);
  } else {
    std::cout << "Conversion to isl_map failed." << std::endl;
  }

  // Create an example IntegerSet: { [d0, d1] : d0 - d1 >= 0 and d0 + d1 = 5 }
  mlir::AffineExpr c1 = d0 - d1;
  mlir::AffineExpr c2 = d0 + d1 - 5;

  mlir::IntegerSet affineSet = mlir::IntegerSet::get(
      /*dimCount=*/2, /*symbolCount=*/0,
      /*constraints=*/{c1, c2},
      /*eqFlags=*/{false, true} // c1 is inequality, c2 is equality
  );

  // Convert IntegerSet to isl_set
  isl_set *islSet = converter.convertAffineSetToISL(affineSet);

  if (islSet) {
    char *islSetStr = isl_set_to_str(islSet);
    std::cout << "Converted isl_set: " << islSetStr << std::endl;

    // Convert back to IntegerSet
    mlir::IntegerSet convertedAffineSet = converter.convertISLSetToAffineSet(islSet, &mlirCtx);
    std::cout << "Converted back to IntegerSet: ";
    convertedAffineSet.print(llvm::outs());
    std::cout << std::endl;

    isl_set_free(islSet);
    free(islSetStr);
  } else {
    std::cout << "Conversion to isl_set failed." << std::endl;
  }

  // Clean up
  isl_ctx_free(islCtx);

  return 0;
}

#endif // SPECHLS_DIALECT_TESTCONVERSION_H
