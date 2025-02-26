//
// Created by Steven on 14/10/2024.
//

#include "AffineToISLConverter.h"

#include <isl/aff.h>
#include <isl/constraint.h>
#include <isl/set.h>
#include <isl/map.h>

// Helper function to convert an MLIR AffineExpr to an ISL string representation
std::string AffineToISLConverter::convertAffineExprToISLString(mlir::AffineExpr expr) {
  using namespace mlir;
  std::string islExprStr;

  if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
    islExprStr = std::to_string(constExpr.getValue());
  } else if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
    islExprStr = "d" + std::to_string(dimExpr.getPosition());
  } else if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>()) {
    islExprStr = "s" + std::to_string(symExpr.getPosition());
  } else if (auto binaryExpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
    std::string lhs = convertAffineExprToISLString(binaryExpr.getLHS());
    std::string rhs = convertAffineExprToISLString(binaryExpr.getRHS());
    switch (binaryExpr.getKind()) {
    case AffineExprKind::Add:
      islExprStr = "(" + lhs + " + " + rhs + ")";
      break;
    case AffineExprKind::Mul:
      islExprStr = "(" + lhs + " * " + rhs + ")";
      break;
    case AffineExprKind::Mod:
      islExprStr = "(" + lhs + " % " + rhs + ")";
      break;
    case AffineExprKind::FloorDiv:
      islExprStr = "floor((" + lhs + ") / (" + rhs + "))";
      break;
    case AffineExprKind::CeilDiv:
      islExprStr = "ceil((" + lhs + ") / (" + rhs + "))";
      break;
    default:
      DEBUG("Unsupported binary operation in AffineExpr.");
      islExprStr = "0";
    }
  } else {
    DEBUG("Unsupported AffineExpr encountered.");
    islExprStr = "0";
  }

  DEBUG("AffineExpr to ISL String: " << islExprStr);
  return islExprStr;
}

// Converts an MLIR AffineMap to an ISL map
isl_map *AffineToISLConverter::convertAffineMapToISL(mlir::AffineMap affineMap) {
  using namespace mlir;
  std::string islMapStr = "{ [";

  // Input dimensions
  for (unsigned i = 0; i < affineMap.getNumDims(); ++i) {
    if (i != 0)
      islMapStr += ", ";
    islMapStr += "d" + std::to_string(i);
  }

  // Symbols
  if (affineMap.getNumSymbols() > 0) {
    islMapStr += "][";
    for (unsigned i = 0; i < affineMap.getNumSymbols(); ++i) {
      if (i != 0)
        islMapStr += ", ";
      islMapStr += "s" + std::to_string(i);
    }
    islMapStr += "]";
  } else {
    islMapStr += "]";
  }

  // Results
  islMapStr += " -> [";
  for (unsigned i = 0; i < affineMap.getNumResults(); ++i) {
    if (i != 0)
      islMapStr += ", ";
    islMapStr += convertAffineExprToISLString(affineMap.getResult(i));
  }
  islMapStr += "] }";

  DEBUG("AffineMap ISL String: " << islMapStr);

  // Convert to isl_map
  isl_map *map = isl_map_read_from_str(ctx, islMapStr.c_str());
  if (!map) {
    DEBUG("Failed to create isl_map from string.");
  }
  return map;
}

// Converts an MLIR IntegerSet to an ISL set
isl_set *AffineToISLConverter::convertAffineSetToISL(mlir::IntegerSet affineSet) {
  using namespace mlir;
  std::string islSetStr = "{ [";

  // Input dimensions
  for (unsigned i = 0; i < affineSet.getNumDims(); ++i) {
    if (i != 0)
      islSetStr += ", ";
    islSetStr += "d" + std::to_string(i);
  }

  // Symbols
  if (affineSet.getNumSymbols() > 0) {
    islSetStr += "][";
    for (unsigned i = 0; i < affineSet.getNumSymbols(); ++i) {
      if (i != 0)
        islSetStr += ", ";
      islSetStr += "s" + std::to_string(i);
    }
    islSetStr += "]";
  } else {
    islSetStr += "]";
  }

  // Constraints
  islSetStr += " : ";
  bool firstConstraint = true;
  for (unsigned i = 0; i < affineSet.getNumConstraints(); ++i) {
    if (!firstConstraint)
      islSetStr += " and ";
    else
      firstConstraint = false;

    auto expr = affineSet.getConstraint(i);
    std::string exprStr = convertAffineExprToISLString(expr);
    if (affineSet.isEq(i)) {
      islSetStr += "(" + exprStr + " = 0)";
    } else {
      islSetStr += "(" + exprStr + " >= 0)";
    }
  }

  islSetStr += " }";

  DEBUG("AffineSet ISL String: " << islSetStr);

  // Convert to isl_set
  isl_set *set = isl_set_read_from_str(ctx, islSetStr.c_str());
  if (!set) {
    DEBUG("Failed to create isl_set from string.");
  }
  return set;
}

// Helper function to convert an ISL affine expression to MLIR AffineExpr
mlir::AffineExpr AffineToISLConverter::convertISLAffineExpr(isl_aff *aff, mlir::MLIRContext *mlirCtx) {
  using namespace mlir;

  isl_ctx *islCtx = isl_aff_get_ctx(aff);
  isl_local_space *ls = isl_aff_get_domain_local_space(aff);
  int totalDims = isl_local_space_dim(ls, isl_dim_all);

  // Coefficients for dimensions and symbols
  std::vector<int64_t> coeffs(totalDims);
  isl_val *c;

  for (int i = 0; i < totalDims; ++i) {
    c = isl_aff_get_coefficient_val(aff, isl_dim_in, i);
    coeffs[i] = isl_val_get_num_si(c);
    isl_val_free(c);
  }

  // Constant term
  c = isl_aff_get_constant_val(aff);
  int64_t constant = isl_val_get_num_si(c);
  isl_val_free(c);

  // Build AffineExpr
  AffineExpr expr = getAffineConstantExpr(constant, mlirCtx);
  for (int i = 0; i < totalDims; ++i) {
    if (coeffs[i] != 0) {
      AffineExpr dimExpr = getAffineDimExpr(i, mlirCtx);
      expr = expr + coeffs[i] * dimExpr;
    }
  }

  isl_local_space_free(ls);
  return expr;
}

// Collects MLIR affine expressions and equality flags from an ISL constraint
void AffineToISLConverter::collectAffineExprFromISLConstraint(isl_constraint *constraint,
                                                              mlir::MLIRContext *mlirCtx,
                                                              std::vector<mlir::AffineExpr> &exprs,
                                                              std::vector<bool> &eqFlags) {
  using namespace mlir;

  isl_aff *aff = isl_constraint_get_aff(constraint);
  AffineExpr expr = convertISLAffineExpr(aff, mlirCtx);

  if (isl_constraint_is_equality(constraint)) {
    eqFlags.push_back(true);
  } else {
    eqFlags.push_back(false);
  }

  exprs.push_back(expr);
  isl_aff_free(aff);
}

// Converts an ISL map to an MLIR AffineMap
mlir::AffineMap AffineToISLConverter::convertISLMapToAffineMap(isl_map *islMap, mlir::MLIRContext *mlirCtx) {
  using namespace mlir;

  // For simplicity, we assume that the map is single-valued and convex
  isl_basic_map *bmap = isl_map_simple_hull(islMap);
  isl_aff_list *affList = isl_basic_map_get_div_list(bmap);
  unsigned numDims = isl_map_dim(islMap, isl_dim_in);
  unsigned numSymbols = 0; // ISL does not have symbols in the same way

  std::vector<AffineExpr> results;

  // For each output dimension, extract the affine expression
  for (unsigned i = 0; i < isl_map_dim(islMap, isl_dim_out); ++i) {
    isl_aff *aff = isl_basic_map_get_aff(bmap, i);
    AffineExpr expr = convertISLAffineExpr(aff, mlirCtx);
    results.push_back(expr);
    isl_aff_free(aff);
  }

  // Create the AffineMap
  AffineMap affineMap = AffineMap::get(numDims, numSymbols, results, mlirCtx);

  isl_aff_list_free(affList);
  isl_basic_map_free(bmap);

  return affineMap;
}

// Converts an ISL set to an MLIR IntegerSet
mlir::IntegerSet AffineToISLConverter::convertISLSetToAffineSet(isl_set *islSet, mlir::MLIRContext *mlirCtx) {
  using namespace mlir;

  isl_basic_set *bset = isl_set_simple_hull(islSet);
  unsigned numDims = isl_set_dim(islSet, isl_dim_set);
  unsigned numSymbols = 0; // ISL does not have symbols in the same way

  std::vector<AffineExpr> constraints;
  std::vector<bool> eqFlags;

  isl_constraint_list *constraintList = isl_basic_set_get_constraint_list(bset);
  int nConstraints = isl_constraint_list_n_constraint(constraintList);

  for (int i = 0; i < nConstraints; ++i) {
    isl_constraint *constraint = isl_constraint_list_get_constraint(constraintList, i);
    collectAffineExprFromISLConstraint(constraint, mlirCtx, constraints, eqFlags);
    isl_constraint_free(constraint);
  }

  // Create the IntegerSet
  IntegerSet intSet = IntegerSet::get(numDims, numSymbols, constraints, eqFlags);

  isl_constraint_list_free(constraintList);
  isl_basic_set_free(bset);

  return intSet;
}
