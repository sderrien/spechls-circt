#ifndef SIMULATEURCOMB_H
#define SIMULATEURCOMB_H

#pragma once

#include <map>
#include <string>
#include <vector>

#include "llvm/ADT/APInt.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

// CIRCT / MLIR (adaptez selon votre arborescence)
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"

/// ---------------------------------------------------------------------------
/// Fonctions utilitaires : gestion des largeurs, extensions, etc.
/// ---------------------------------------------------------------------------
unsigned getValueBitWidth(mlir::Value val);

llvm::APInt adjustToWidth(const llvm::APInt &in,
                          unsigned targetWidth,
                          bool isSigned);

llvm::APInt getOperandValueAdjusted(
    mlir::Value operand,
    const std::map<mlir::Value, llvm::APInt> &valueMap,
    unsigned targetWidth,
    bool isSignedExtension);

unsigned computeMaxWidthForArithmetic(mlir::Operation *op);

/// ---------------------------------------------------------------------------
/// Fonctions d’évaluation intermédiaires : une par type d'opération Comb
/// ---------------------------------------------------------------------------

// 1) Add, And, Or, Xor
llvm::APInt evaluateAddOp(circt::comb::AddOp combOp,
                          const std::map<mlir::Value, llvm::APInt> &valueMap);

llvm::APInt evaluateAndOp(circt::comb::AndOp combOp,
                          const std::map<mlir::Value, llvm::APInt> &valueMap);

llvm::APInt evaluateOrOp(circt::comb::OrOp combOp,
                         const std::map<mlir::Value, llvm::APInt> &valueMap);

llvm::APInt evaluateXorOp(circt::comb::XorOp combOp,
                          const std::map<mlir::Value, llvm::APInt> &valueMap);

// 2) Sub, Mul
llvm::APInt evaluateSubOp(circt::comb::SubOp combOp,
                          const std::map<mlir::Value, llvm::APInt> &valueMap);

llvm::APInt evaluateMulOp(circt::comb::MulOp combOp,
                          const std::map<mlir::Value, llvm::APInt> &valueMap);

// 3) DivU, DivS, ModU, ModS
llvm::APInt evaluateDivUOp(circt::comb::DivUOp combOp,
                           const std::map<mlir::Value, llvm::APInt> &valueMap);

llvm::APInt evaluateDivSOp(circt::comb::DivSOp combOp,
                           const std::map<mlir::Value, llvm::APInt> &valueMap);

llvm::APInt evaluateModUOp(circt::comb::ModUOp combOp,
                           const std::map<mlir::Value, llvm::APInt> &valueMap);

llvm::APInt evaluateModSOp(circt::comb::ModSOp combOp,
                           const std::map<mlir::Value, llvm::APInt> &valueMap);

// 4) Shl, ShrU, ShrS
llvm::APInt evaluateShlOp(circt::comb::ShlOp combOp,
                          const std::map<mlir::Value, llvm::APInt> &valueMap);

llvm::APInt evaluateShrUOp(circt::comb::ShrUOp combOp,
                           const std::map<mlir::Value, llvm::APInt> &valueMap);

llvm::APInt evaluateShrSOp(circt::comb::ShrSOp combOp,
                           const std::map<mlir::Value, llvm::APInt> &valueMap);

// 5) SExt, ZExt
llvm::APInt evaluateSExtOp(circt::comb::SExtOp combOp,
                           const std::map<mlir::Value, llvm::APInt> &valueMap);

llvm::APInt evaluateZExtOp(circt::comb::ZExtOp combOp,
                           const std::map<mlir::Value, llvm::APInt> &valueMap);

// 6) Parity, Concat, Extract, Replicate, Mux
llvm::APInt evaluateParityOp(circt::comb::ParityOp combOp,
                             const std::map<mlir::Value, llvm::APInt> &valueMap);

llvm::APInt evaluateConcatOp(circt::comb::ConcatOp combOp,
                             const std::map<mlir::Value, llvm::APInt> &valueMap);

llvm::APInt evaluateExtractOp(circt::comb::ExtractOp combOp,
                              const std::map<mlir::Value, llvm::APInt> &valueMap);

llvm::APInt evaluateReplicateOp(circt::comb::ReplicateOp combOp,
                                const std::map<mlir::Value, llvm::APInt> &valueMap);

llvm::APInt evaluateMuxOp(circt::comb::MuxOp combOp,
                          const std::map<mlir::Value, llvm::APInt> &valueMap);

// 7) ICmp, LUT
llvm::APInt evaluateICmpOp(circt::comb::ICmpOp combOp,
                           const std::map<mlir::Value, llvm::APInt> &valueMap);

llvm::APInt evaluateLUTOp(circt::comb::LUTOp combOp,
                          const std::map<mlir::Value, llvm::APInt> &valueMap);

/// ---------------------------------------------------------------------------
/// Fonction de dispatch : appelle la bonne fonction evaluateXXXOp(...).
/// ---------------------------------------------------------------------------
void evaluateCombOperation(mlir::Operation *op,
                           std::map<mlir::Value, llvm::APInt> &valueMap);

/// ---------------------------------------------------------------------------
/// Détection de cycles + tri topologique (DAG combinatoire).
/// ---------------------------------------------------------------------------
bool topologicalSortAndCheckCycles(const std::vector<mlir::Operation *> &ops,
                                   std::vector<mlir::Operation *> &sortedOps);

/// ---------------------------------------------------------------------------
/// Classe SimulateurComb : simule un HwModuleOp combinatoire en assignant
/// des valeurs d’entrée (map) et en retournant les valeurs de sortie.
/// ---------------------------------------------------------------------------
class SimulateurComb {
public:
  /// Simule le module combinatoire 'hwModule' en utilisant 'inputValues'
  /// comme valeurs pour les ports d'entrée. Retourne les valeurs de sortie
  /// dans une map <portName -> APInt>.
  std::map<std::string, llvm::APInt> simulate(
      circt::hw::HWModuleOp hwModule,
      const std::map<std::string, llvm::APInt> &inputValues);
};

#endif // SIMULATEURCOMB_H
