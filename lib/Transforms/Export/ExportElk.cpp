//===- ExportVitisHLS.cpp - Arith-to-comb mapping pass ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the ExportVitisHLS pass.
//
//===----------------------------------------------------------------------===//
// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/Pass/Pass.h"

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"

#include "Transforms/Passes.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"

#include "Transforms/VitisExport/CFileContent.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

using namespace std;

using namespace mlir;
using namespace circt;
using namespace SpecHLS;
using namespace circt::hw;

namespace SpecHLS {

struct ExportElkPass : public impl::ExportElkBase<ExportElkPass> {

private:
  llvm::DenseMap<Operation *, int> operationIDMap;
  int idCounter = 100;
  int depth=0;

  std::string content;

  int getOperationId(Operation *op) {
    if (!operationIDMap.contains(op)) {
      operationIDMap[op] = idCounter++;
    }
    return operationIDMap[op];
  }

public:

  std::string generateDefUseEdge(OpResult *def, OpOperand *use) {
    auto id = property("id", quote("e_"+std::to_string(idCounter++)));
    auto srcOpId = getOperationId(def->getDefiningOp());
    auto tgtOpId = getOperationId(use->getOwner());
    auto src = property("sources",llvm::formatv("[\"n{0}.op{1}\"]",srcOpId,def->getResultNumber()));
    auto tgt = property("targets",llvm::formatv("[\"n{0}.ip{1}\"]",tgtOpId,use->getOperandNumber()));
    return llvm::formatv("{{ {0},{1},{2} }\n", id,src,tgt);
  }

  std::string quote(std::string s) { return "\"" + s + "\""; }

  std::string property(std::string name, std::string value) {
    return (quote(name) + ":" + value);
  }

  std::string port(std::string name, std::string value) {
    return llvm::formatv("{{\"id\" : \"{0}\"}, {{\"side\": \"{1}}\" }\n", name,
                         value);
  }

  std::string label(std::string id, std::string label) {
    return llvm::formatv("\"labels\": [{{ {0} }]",
                         property("text", label));
  }



void edgeList(Operation* op) {

  if (op->getNumResults()>0) {
    append("\"edges\" : [");
    pushIndent();
    for (auto arg : op->getResults()) {
      for (auto use : llvm::enumerate(arg.getUses())) {
        auto au_se = &use.value();
        if (use.index() > 0)
          append(",");
        append(generateDefUseEdge(&arg, &use.value()));
      }
    }
    popIndent();
    append("]");
  }
}
  void portList(std::string nodeId, int nbInput, int nbOutput) {

    if ((nbInput+nbOutput)>0) {
      append("\"ports\":[");
      pushIndent();
      for (auto k = 0; k < nbInput; k++) {
        if (k > 0)
          append(",");
        append(llvm::formatv("{{ {0},{1} }",
                                 property("id", quote(nodeId+".ip" + std::to_string(k))),
                                 property("side", quote("WEST"))));
      }

      for (auto k = 0; k < nbOutput; k++) {
        if (k > 0 || nbInput > 0)
          append(",");
        append(llvm::formatv("{{ {0},{1} }",
                                 property("id", quote(nodeId+".op" + std::to_string(k))),
                                 property("side", quote("EAST"))));
      }
      popIndent();
      append(llvm::formatv("]"));
    }

  }

  void childrenTask(HTaskOp* op) {
    append(llvm::formatv("\"children\": ["));
    for (auto child : llvm::enumerate(op->getBody(0)->getOperations())) {
      if (child.index() > 0) append(",");
      llvm::outs() << "start visiting child " << child.value() << "\n";
      pushIndent();
      generate(&child.value());
      popIndent();
      llvm::outs() << "end visiting child " << child.value() << "\n";
    }
    append(llvm::formatv("]"));
  }

  void append(std::string line) {
    std:string res="";
      for (int i=0;i<depth;i++) {
         res+="\t";
      }
      content += res + line +"\n";
  }


  void pushIndent() {
    depth ++;
  }

  void popIndent() {
    depth --;
  }

  void generate(Operation *op) {
    llvm::errs() << "Generating Elk for " << *op << "\n";

    auto opname = op->getName().getStringRef().str();
    std::string  nodeId = llvm::formatv("n{0}",getOperationId(op));

    append("{");
    append(property("id", quote(nodeId))+",");
    append(property("width", std::to_string(80))+",");
    append(property("height", std::to_string(80))+",");
    append(label(nodeId+"_l", quote(opname)));

    if (op->getNumResults()>0|| op->getNumOperands()>0 || op->getNumRegions()>0) {
      append(",");

      portList(nodeId,op->getNumOperands(),op->getNumResults());
      if (auto hwop = dyn_cast<HTaskOp>(op)) {
        childrenTask(&hwop);
      }
      if (op->getNumResults()>0) append(",");

      edgeList(op);
    }
    append("}\n");
  }

  void runOnOperation() {
    auto *ctx = &getContext();
    auto module = this->getOperation();
    llvm::errs() << "Module   " << module << "\n";

    for (auto kernel : module.getOps<SpecHLS::HKernelOp>()) {
      // Create an output file stream object
      llvm::errs() << "Generating ElkJS file for  " << kernel.getName() << "\n";
      auto filename = kernel.getName() + ".json";
      std::ofstream outFile(filename.str());
      // Check if the file was opened successfully
      if (!outFile) {
        llvm::errs() << "Error: Could not open the file " << filename << "\n";
      } else {
        // Write the string to the file
        content="";
        generate(kernel);
        outFile << content;
        llvm::outs() << "Content written to " << filename << "\n";
        // Close the file stream
        outFile.close();
      }
    }
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createExportElkPass() {
  return std::make_unique<ExportElkPass>();
}
} // namespace SpecHLS