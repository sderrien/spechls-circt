/*
* DFGMergePass.cpp
*
* This file implements the DFGMergePass which merges DFGs by detecting and merging maximum weighted cliques.
* The main logic of the pass involves building a compatibility graph, detecting maximum weighted cliques,
* and merging operations within these cliques. This process reduces hardware resource usage.
*
* This version of the file includes detailed comments for every function, explains the role of each argument,
* and provides debug information at every key step of the algorithm.
*/
#include "Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"

#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>

using namespace mlir;
using namespace circt;

namespace SpecHLS {

/// Flag to enable debug information.
bool debug = false;

/// DFGMergePass: A pass to optimize DFGs (Data Flow Graphs) by merging operations.
struct DFGMergePass : public impl::DFGMergePassBase<DFGMergePass> {
public:
 enum CliqueAlgorithm { BronKerbosch = 0, Grasp = 1 };

 DFGMergePass() = default;

 // User-configurable option to select the algorithm used for maximum weighted clique detection.
 Option<CliqueAlgorithm> cliqueAlgorithm{
     *this, "clique-algorithm",
     llvm::cl::desc("Algorithm to use for maximum weighted clique"),
     llvm::cl::init(BronKerbosch),
     llvm::cl::values(clEnumValN(BronKerbosch, "bron-kerbosch", "Use Bron-Kerbosch algorithm"),
                      clEnumValN(Grasp, "grasp", "Use GRASP heuristic"))};

 /// Main entry point for the pass. Applies the DFG merge process to the HWModuleOp.
 void runOnOperation() override {

   // PROMPT : corrige le code car cette de passe doit fonctionner sur un ModuleOp Qui est supposé contenir une série de HwModuleOp (Si pas le cas, La place de renvoyer une erreur).

   // La passe fait les choses suivantes :
   // 1) elle récupère les deux premières instances de HwModuleOp du ModuleOp, et les fusionne
   // 2) elle les remplace dans le ModuleOp par le module fusionné,
   // 3) elle recommencer en 1) jusqu'à ce qu'il n'y ait plus qu'un seul HwModule dans le ModuleOp

   // Pour faire la fusion, la passe construit un Graphe de compatibilité entre les deux HwModulop à fusionner.
   // Ce graphe de compatibilité va lier les opérations internes, les inputs et les output de chacun de ces deux HWModuleOp.
   // Deux inputs (ou output) peuvent-être fusionnés si même nom et même type
   // Deux operation peuvent-être fusionnées si même nom (type différent possible)

   // Une fois la clique maximal obtenue il faut fusionner les deux modules. Pour cela :
   // 1) on crée un nouveau HwModuleOp avec les inputs des deux modules (en s'assurant que les input fusionnée n'apparaissent qu'une fois afin de refléter la fusion) avec un signe de controle select:i32.
   // 2) on ajoute les operations non-fusionnées de deux HwModuleOp telles quels.
   // 3) on ajoute les operations fusionnées de la manière suivante :
   //   3.1) pour chaque operation fusionnée, on ajoute un mux sur chacune des opérandes de l'oération
   //   3.2) ce mux est commandé par le signal select, et qui choisi entre les opérandes des deux HwModuleOp fusionnés

   if (debug) llvm::outs() << "Running DFGMergePass on HWModuleOp.\n";
   hw::HWModuleOp module = getOperation();
   OpBuilder builder(module.getContext());

   module.walk([&](Operation *op) {
     if (auto addOp = dyn_cast<comb::AddOp>(op)) {
       if (debug) llvm::outs() << "Processing comb::AddOp.\n";
       tryToMergeDFG(addOp, builder);
     } else if (auto mulOp = dyn_cast<comb::MulOp>(op)) {
       if (debug) llvm::outs() << "Processing comb::MulOp.\n";
       tryToMergeDFG(mulOp, builder);
     }
   });
 }

 /// Attempts to merge the Data Flow Graph (DFG) rooted at `op`.
 /// @param op The root operation of the DFG to be merged.
 /// @param builder The OpBuilder used to create new operations.
 void tryToMergeDFG(Operation *op, OpBuilder &builder) {
   if (debug) llvm::outs() << "Building compatibility graph.\n";

   std::vector<Operation *> dfgOps;
   std::unordered_map<Operation *, std::vector<std::pair<Operation *, int>>> compatibilityGraph;

   op->getParentRegion()->walk([&](Operation *innerOp) {
     if (isa<comb::AddOp>(innerOp) || isa<comb::MulOp>(innerOp)) {
       dfgOps.push_back(innerOp);
     }
   });

   if (debug) llvm::outs() << "Building compatibility graph.\n";
   buildCompatibilityGraph(dfgOps, compatibilityGraph);

   if (debug) llvm::outs() << "Detecting maximum weighted clique.\n";
   std::vector<Operation *> maxClique;
   if (cliqueAlgorithm == BronKerbosch) {
     maxClique = bronKerboschMaxWeightClique(compatibilityGraph);
   } else {
     maxClique = graspMaxWeightClique(compatibilityGraph);
   }

   if (debug) llvm::outs() << "Creating merged HW module for the clique.\n";
   createMergedHWModule(op, maxClique, builder);
 }

 /// Builds the compatibility graph.
 void buildCompatibilityGraph(const std::vector<Operation *> &dfgOps,
                              std::unordered_map<Operation *, std::vector<std::pair<Operation *, int>>> &compatibilityGraph) {
   for (size_t i = 0; i < dfgOps.size(); ++i) {
     for (size_t j = i + 1; j < dfgOps.size(); ++j) {
       if (canMergeOperations(dfgOps[i], dfgOps[j])) {
         int weight = calculateMergeWeight(dfgOps[i], dfgOps[j]);
         compatibilityGraph[dfgOps[i]].emplace_back(dfgOps[j], weight);
         compatibilityGraph[dfgOps[j]].emplace_back(dfgOps[i], weight);
       }
     }
   }
 }

 /// Checks if two operations can be merged.
 bool canMergeOperations(Operation *op1, Operation *op2) {
   return op1->getName() == op2->getName();
 }

 /// Calculates the weight of merging two operations.
 int calculateMergeWeight(Operation *op1, Operation *op2) {
   return 10; // Dummy weight function
 }

 /// Detects the maximum weighted clique using the GRASP heuristic.
 std::vector<Operation *> graspMaxWeightClique(const std::unordered_map<Operation *, std::vector<std::pair<Operation *, int>>> &graph) {
   return {}; // Dummy implementation
 }

 /// Detects the maximum weighted clique using the Bron-Kerbosch algorithm.
 std::vector<Operation *> bronKerboschMaxWeightClique(const std::unordered_map<Operation *, std::vector<std::pair<Operation *, int>>> &graph) {
   return {}; // Dummy implementation
 }

 /// Creates a new HW module that merges the operations in the maximum clique.
 void createMergedHWModule(Operation *rootOp, std::vector<Operation *> &maxClique, OpBuilder &builder) {
   // Dummy implementation
 }
};

} // end namespace SpecHLS

/// Factory method to create a new instance of DFGMergePass
std::unique_ptr<Pass> createDFGMergePass() {
 return std::make_unique<SpecHLS::DFGMergePass>();
}

/// Register the pass using the factory method
static PassRegistration<SpecHLS::DFGMergePass> pass("dfg-merge", "Merge DFGs using the Comb dialect to optimize area and resource usage.", createDFGMergePass);
