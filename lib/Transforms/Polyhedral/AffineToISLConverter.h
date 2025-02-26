//
// Created by Steven on 14/10/2024.
//

#ifndef AFFINE_TO_ISL_CONVERTER_H
#define AFFINE_TO_ISL_CONVERTER_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "isl/isl.h"
#include <string>
#include <iostream>
#include <vector>

// Debug mode: Set to 1 to enable verbose output, 0 to disable
#define DEBUG_MODE 1

#if DEBUG_MODE
#define DEBUG(x) do { std::cout << x << std::endl; } while (0)
#else
#define DEBUG(x)
#endif

class AffineToISLConverter {
public:
  AffineToISLConverter(isl_ctx *ctx) : ctx(ctx) {}

  // Conversion functions
  isl_map *convertAffineMapToISL(mlir::AffineMap affineMap);
  isl_set *convertAffineSetToISL(mlir::IntegerSet affineSet);

  mlir::AffineMap convertISLMapToAffineMap(isl_map *islMap, mlir::MLIRContext *mlirCtx);
  mlir::IntegerSet convertISLSetToAffineSet(isl_set *islSet, mlir::MLIRContext *mlirCtx);

private:
  std::string convertAffineExprToISLString(mlir::AffineExpr expr);
  mlir::AffineExpr convertISLAffineExpr(isl_aff *aff, mlir::MLIRContext *mlirCtx);
  void collectAffineExprFromISLConstraint(isl_constraint *constraint,
                                          mlir::MLIRContext *mlirCtx,
                                          std::vector<mlir::AffineExpr> &exprs,
                                          std::vector<bool> &eqFlags);

  isl_ctx *ctx;
};

#endif // AFFINE_TO_ISL_CONVERTER_H

