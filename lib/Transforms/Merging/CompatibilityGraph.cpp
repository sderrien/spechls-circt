#include "mlir/IR/Operation.h"
#include "mlir/IR/Argument.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <fstream>
#include <random>

// Enum to distinguish node types: OperationPair or ArgumentPair
enum class NodeType {
  OperationPair,
  ArgumentPair
};

// GraphNode represents a node in the graph with a weight and node type (either OperationPair or ArgumentPair)
struct GraphNode {
  int32_t weight;  // The weight of the node
  NodeType nodeType;  // Type of node: OperationPair or ArgumentPair

  // Use llvm::PointerUnion to hold either a pair of Operations or Arguments
  llvm::PointerUnion<std::pair<mlir::Operation*, mlir::Operation*>,
                     std::pair<mlir::Argument*, mlir::Argument*>> nodePair;

  // Constructor for a node with a pair of Operations
  GraphNode(int32_t w, mlir::Operation* op1, mlir::Operation* op2)
      : weight(w), nodeType(NodeType::OperationPair) {
    nodePair = std::make_pair(op1, op2);
  }

  // Constructor for a node with a pair of Arguments
  GraphNode(int32_t w, mlir::Argument* arg1, mlir::Argument* arg2)
      : weight(w), nodeType(NodeType::ArgumentPair) {
    nodePair = std::make_pair(arg1, arg2);
  }
};

// Graph class representing the undirected graph
class Graph {
public:
  llvm::SmallVector<GraphNode, 4> nodes; // List of nodes in the graph
  llvm::DenseMap<int, llvm::SmallVector<int, 4>> adjacencyMap;  // Adjacency map (node index -> list of adjacent nodes)

  // Add a node with a pair of Operations
  void addOperationNode(int32_t weight, mlir::Operation* op1, mlir::Operation* op2) {
    int nodeIndex = nodes.size();
    nodes.emplace_back(weight, op1, op2);
    updateAdjacencyMap(nodeIndex);  // Update adjacency map after adding the node
  }

  // Add a node with a pair of Arguments
  void addArgumentNode(int32_t weight, mlir::Argument* arg1, mlir::Argument* arg2) {
    int nodeIndex = nodes.size();
    nodes.emplace_back(weight, arg1, arg2);
    updateAdjacencyMap(nodeIndex);  // Update adjacency map after adding the node
  }

  // Method to update the adjacency map after adding a node
  void updateAdjacencyMap(int nodeIndex) {
    // Add edges to the graph (logic simplified here)
    for (int i = 0; i < nodeIndex; ++i) {
      adjacencyMap[nodeIndex].push_back(i);  // Example: each node is connected to all others
      adjacencyMap[i].push_back(nodeIndex);
    }
  }

  // Bron-Kerbosch algorithm to find all maximal cliques
  void bronKerbosch(llvm::SmallVector<int, 4>& R, llvm::SmallVector<int, 4>& P, llvm::SmallVector<int, 4>& X, llvm::SmallVector<llvm::SmallVector<int, 4>, 4>& cliques) {
    // If both P and X are empty, R is a maximal clique
    if (P.empty() && X.empty()) {
      cliques.push_back(R);  // Add the clique to the list of cliques
      return;
    }

    // Select a pivot from the nodes in P ∪ X
    int pivot = P.front();
    llvm::SmallVector<int, 4> P_temp = P;
    for (int neighbor : adjacencyMap[pivot]) {
      P_temp.erase(std::remove(P_temp.begin(), P_temp.end(), neighbor), P_temp.end());  // Remove neighbors of pivot from P
    }

    // Explore each node v in P \ N(pivot) (nodes in P that are not neighbors of the pivot)
    for (int v : P_temp) {
      R.push_back(v);  // Add v to R, the set of nodes in the clique
      bronKerbosch(R, P, X, cliques);  // Recursive call with the updated R
      R.pop_back();  // Remove v from R

      P.erase(std::remove(P.begin(), P.end(), v), P.end());  // Remove v from P
      X.push_back(v);  // Add v to X, because it can no longer be added to the clique
    }
  }

  // GRASP heuristic to find a maximum clique (approximate solution)
  void graspClique(llvm::SmallVector<int, 4>& clique) {
    llvm::SmallVector<int, 4> nodesRemaining(nodes.size());
    std::iota(nodesRemaining.begin(), nodesRemaining.end(), 0);  // Initialize remaining nodes

    std::random_device rd;
    std::mt19937 gen(rd());

    while (!nodesRemaining.empty()) {
      // Randomly select a node from the remaining nodes
      std::shuffle(nodesRemaining.begin(), nodesRemaining.end(), gen);
      int node = nodesRemaining.front();

      // Add this node to the clique if it forms a valid clique
      bool isValid = true;
      for (int existingNode : clique) {
        if (std::find(adjacencyMap[existingNode].begin(), adjacencyMap[existingNode].end(), node) == adjacencyMap[existingNode].end()) {
          isValid = false;
          break;
        }
      }

      if (isValid) {
        clique.push_back(node);  // Add the node to the clique
      }

      // Remove the node from the remaining nodes
      nodesRemaining.erase(std::remove(nodesRemaining.begin(), nodesRemaining.end(), node), nodesRemaining.end());
    }
  }

  // Method to dump the graph in Graphviz DOT format
  void dumpGraph(const std::string& filename) {
    // Open a file stream to write the DOT format to a file
    std::ofstream file(filename);
    if (!file.is_open()) {
      llvm::errs() << "Error opening file for writing.\n";
      return;
    }

    // Start the Graphviz representation
    file << "graph G {\n";

    // Iterate over each node and add it to the DOT format
    for (size_t i = 0; i < nodes.size(); ++i) {
      const GraphNode& node = nodes[i];
      file << "  " << i << " [label=\"Weight: " << node.weight;

      // Add additional information based on node type
      if (node.nodeType == NodeType::OperationPair) {
        file << ", Type: OperationPair\"];\n";
      } else if (node.nodeType == NodeType::ArgumentPair) {
        file << ", Type: ArgumentPair\"];\n";
      }
    }

    // Iterate over each edge and add it to the DOT format
    for (size_t i = 0; i < nodes.size(); ++i) {
      for (size_t j = i + 1; j < nodes.size(); ++j) {
        if (std::find(adjacencyMap[i].begin(), adjacencyMap[i].end(), j) != adjacencyMap[i].end()) {
          file << "  " << i << " -- " << j << ";\n";  // Undirected edge
        }
      }
    }

    // End the Graphviz representation
    file << "}\n";

    // Close the file stream
    file.close();
  }

  // Method to print the graph nodes for debugging
  void print() {
    for (const auto& node : nodes) {
      llvm::outs() << "Node weight: " << node.weight << ", ";
      if (node.nodeType == NodeType::OperationPair) {
        llvm::outs() << "Edge type: OperationPair\n";
      } else if (node.nodeType == NodeType::ArgumentPair) {
        llvm::outs() << "Edge type: ArgumentPair\n";
      }
    }
  }
};
