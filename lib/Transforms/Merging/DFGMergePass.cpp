//
// Created by Steven on 13/12/2024.
//

#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"

#include <algorithm>
#include <fstream> // Added this to define std::ofstream
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace mlir;
using namespace circt;

namespace SpecHLS {
// #define DFGMERGE_GPT

/**
 * DFGMergePass: A pass to optimize DFGs (Data Flow Graphs) by merging
 * operations. The merging is achieved by identifying maximum weighted cliques
 * in a compatibility graph using one of two algorithms:
 * 1. **Bron-Kerbosch**: An exact algorithm for maximum clique detection.
 * 2. **GRASP**: A heuristic approach that provides a faster, approximate
 * solution.
 *
 * The goal is to reduce hardware resource usage by merging compatible
 * operations.
 */

extern std::vector<Operation *> bronKerboschMaxWeightClique(const std::unordered_map<Operation *,
                             std::vector<std::pair<Operation *, int>>> &graph);
extern std::vector<Operation *> graspMaxWeightClique(
    const std::unordered_map<Operation *,
                             std::vector<std::pair<Operation *, int>>> &graph);

struct DFGMergePass : public impl::DFGMergePassBase<DFGMergePass> {
  bool debug = false; // Flag to enable debug information.

public:
  // User-configurable option to select the algorithm used for maximum
  // weighted clique detection. Enum to represent the algorithm selection for
  // maximum weighted clique
  enum CliqueAlgorithm { BronKerbosch = 0, Grasp = 1 };

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    if (!module) {
      module.emitError("The operation is not a valid ModuleOp.");
      return;
    }

    std::vector<hw::HWModuleOp> hwModules;
    module.walk(
        [&](hw::HWModuleOp hwModule) { hwModules.push_back(hwModule); });

    if (debug)
      llvm::outs() << "Number of HWModuleOps found: " << hwModules.size()
                   << "\n";

    while (hwModules.size() > 1) {
      hw::HWModuleOp moduleA = hwModules.back();
      hwModules.pop_back();
      hw::HWModuleOp moduleB = hwModules.back();
      hwModules.pop_back();

      if (debug)
        llvm::outs() << "Merging HWModuleOp instances.\n";

      hw::HWModuleOp mergedModule = mergeHWModules(moduleA, moduleB, builder);
      hwModules.push_back(mergedModule);
    }
  }

  hw::HWModuleOp mergeHWModules(hw::HWModuleOp moduleA, hw::HWModuleOp moduleB,
                                OpBuilder &builder) {
    if (debug)
      llvm::outs() << "Building compatibility graph for two HWModuleOps.\n";

    std::unordered_map<Operation *, std::vector<std::pair<Operation *, int>>>
        compatibilityGraph;
    buildCompatibilityGraph(moduleA, moduleB, compatibilityGraph);

    if (debug)
      llvm::outs() << "Finding maximum clique in the compatibility graph.\n";
    std::vector<Operation *> maxClique = findMaxClique(compatibilityGraph);

    if (debug)
      llvm::outs() << "Creating new HWModuleOp for the merged modules.\n";
    return createMergedHWModule(moduleA, moduleB, maxClique, builder);
  }

  void buildCompatibilityGraph(
      hw::HWModuleOp moduleA, hw::HWModuleOp moduleB,
      std::unordered_map<Operation *, std::vector<std::pair<Operation *, int>>>
          &graph) {

    /*
     * must create dummy node to capture inputs
     * for (auto &argA : moduleA.getBody().getArguments()) {
      for (auto &argB : moduleB.getBody().getArguments()) {
        if (argA.getType() == argB.getType()) {
          graph[&argA].emplace_back(&argB, 1);
          graph[&argB].emplace_back(&argA, 1);
        }
      }
    }
*/
    for (auto &opA : moduleA.getBody().getOps()) {
      for (auto &opB : moduleB.getBody().getOps()) {
        if (opA.getName() == opB.getName()) {
          graph[&opA].emplace_back(&opB, 1);
          graph[&opB].emplace_back(&opA, 1);
        }
      }
    }
  }

  std::vector<Operation *>
  findMaxClique(const std::unordered_map<
                Operation *, std::vector<std::pair<Operation *, int>>> &graph) {
    std::vector<Operation *> maxClique;
    if (cliqueAlgorithm.getValue() == "grasp") {
      return graspMaxWeightClique(graph);
    } else {
      return bronKerboschMaxWeightClique(graph);
    }
  }

  hw::HWModuleOp createMergedHWModule(hw::HWModuleOp moduleA,
                                      hw::HWModuleOp moduleB,
                                      std::vector<Operation *> &maxClique,
                                      OpBuilder &builder) {
    if (debug)
      llvm::outs()
          << "Creating new HWModuleOp with merged inputs and operations.\n";

    std::vector<hw::PortInfo> mergedPorts;
    size_t inputOffset = 0;
    size_t outputOffset = 0;
    for (auto &port : moduleA.getPortList()) {
      if (port.isInput()) {
        hw::PortInfo copy{port.name, port.type, port.dir, inputOffset++};
        mergedPorts.push_back(copy);
      } else {
        hw::PortInfo copy{port.name, port.type, port.dir, outputOffset++};
        mergedPorts.push_back(copy);
      }
    }

    for (auto &port : moduleA.getPortList()) {
      // only add inputs that are not shared
      if (std::none_of(mergedPorts.begin(), mergedPorts.end(),
                       [&](hw::PortInfo val) {
                         return val.name == port.name && val.type == port.type;
                       })) {
        if (port.isInput()) {
          hw::PortInfo copy{port.name, port.type, port.dir, inputOffset++};
          mergedPorts.push_back(copy);
        } else {
          hw::PortInfo copy{port.name, port.type, port.dir, outputOffset++};
          mergedPorts.push_back(copy);
        }
      }
    }

    // PROMPT : print the new inputs here

    std::string newModuleName =
        moduleA.getName().str() + "_merged_" + moduleB.getName().str();

    auto newModule = builder.create<hw::HWModuleOp>(
        moduleA.getLoc(), builder.getStringAttr(newModuleName), mergedPorts);

    builder.setInsertionPointToStart(&newModule.getBody().front());

    for (auto &op : moduleA.getBody().getOps()) {
      builder.clone(op);
    }
    for (auto &op : moduleB.getBody().getOps()) {
      // only add inputs that are not shared
      if (std::find(maxClique.begin(), maxClique.end(), &op) ==
          maxClique.end()) {
        builder.clone(op);
      }
    }

    for (auto *mergedOp : maxClique) {
      // Merge logic for the operations in the clique

      // builder.create<circt::comb::MuxOp>()
      //  Placeholder for merging logic
    }

    return newModule;
  }
}
;

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createDFGMergePass() {
  return std::make_unique<DFGMergePass>();
}


} // namespace SpecHLS
