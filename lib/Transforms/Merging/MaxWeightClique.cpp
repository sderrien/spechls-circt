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

namespace SpecHLS {
int calculateCliqueWeight(
    const std::vector<Operation *> &clique,
    const std::unordered_map<Operation *,
                             std::vector<std::pair<Operation *, int>>> &graph) {
  int totalWeight = 0;
  for (auto *op : clique) {
    for (auto &neighbor : graph.at(op)) {
      if (std::find(clique.begin(), clique.end(), neighbor.first) !=
          clique.end()) {
        totalWeight += neighbor.second;
      }
    }
  }
  // Each edge is counted twice, so divide the total weight by 2
  return totalWeight / 2;
}

/**
   * GRASP heuristic for finding a maximum weighted clique.
   * The method performs multiple iterations, each time randomly selecting initial vertices to build a clique. At the end of each iteration, the total weight of the clique is calculated, and the heaviest clique across all iterations is stored as the best result.
 */
std::vector<Operation *> graspMaxWeightClique(
    const std::unordered_map<Operation *,
                             std::vector<std::pair<Operation *, int>>> &graph) {
  std::vector<Operation *> maxClique;
  int maxWeight = 0;
  std::default_random_engine generator;

  // Perform multiple iterations to increase the likelihood of finding a near-optimal clique
  for (int i = 0; i < 10; ++i) {
    std::vector<Operation *> candidateVertices;

    // Step 1: Collect all vertices in the graph
    for (const auto &node : graph) {
      candidateVertices.push_back(node.first);
    }

    // Step 2: Randomly shuffle vertices to introduce variability in the clique construction process
    std::shuffle(candidateVertices.begin(), candidateVertices.end(), generator);

    std::vector<Operation *> clique;

    // Step 3: Greedily construct a clique by selecting vertices from candidate set
    while (!candidateVertices.empty()) {
      // Pick the last vertex from the shuffled list and add it to the current clique
      Operation *v = candidateVertices.back();
      candidateVertices.pop_back();
      clique.push_back(v);

      // Step 4: Filter the candidate set to only include neighbors of the current vertex
      std::vector<Operation *> newCandidates;
      for (Operation *neighbor : candidateVertices) {
        if (std::any_of(graph.at(v).begin(), graph.at(v).end(),
                        [&](auto &pair) { return pair.first == neighbor; })) {
          newCandidates.push_back(neighbor);
        }
      }
      candidateVertices = newCandidates;
    }

    // Step 5: Calculate the total weight of the clique and update the maximum if necessary
    int weight = calculateCliqueWeight(clique, graph);
    if (weight > maxWeight) {
      maxWeight = weight;
      maxClique = clique;
    }
  }
  return maxClique;
}
void bronKerboschRecursive(
    std::vector<Operation *> &R, std::vector<Operation *> &P,
    std::vector<Operation *> &X,
    const std::unordered_map<Operation *,
                             std::vector<std::pair<Operation *, int>>> &graph,
    std::vector<Operation *> &maxClique, int &maxWeight) {
  if (P.empty() && X.empty()) {
    int currentWeight = calculateCliqueWeight(R, graph);
    if (currentWeight > maxWeight) {
      maxWeight = currentWeight;
      maxClique = R;
    }
    return;
  }
  for (auto it = P.begin(); it != P.end();) {
    Operation *v = *it;
    std::vector<Operation *> newR = R;
    newR.push_back(v);

    std::vector<Operation *> newP;
    std::vector<Operation *> newX;
    for (auto &neighbor : graph.at(v)) {
      if (std::find(P.begin(), P.end(), neighbor.first) != P.end()) {
        newP.push_back(neighbor.first);
      }
      if (std::find(X.begin(), X.end(), neighbor.first) != X.end()) {
        newX.push_back(neighbor.first);
      }
    }

    bronKerboschRecursive(newR, newP, newX, graph, maxClique, maxWeight);

    X.push_back(v);
    it = P.erase(it);
  }
}
std::vector<Operation *> bronKerboschMaxWeightClique(
    const std::unordered_map<Operation *,
                             std::vector<std::pair<Operation *, int>>> &graph) {
  std::vector<Operation *> maxClique;
  int maxWeight = 0;
  std::vector<Operation *> R, P, X;
  for (const auto &node : graph) {
    P.push_back(node.first);
  }
  bronKerboschRecursive(R, P, X, graph, maxClique, maxWeight);
  return maxClique;
}



}