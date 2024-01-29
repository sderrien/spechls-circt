
#include "LUTImporter.h"
#include "SpecHLS/SpecHLSOps.h"
#include "circt/Dialect/Comb/CombOps.h"

#include "mlir/IR/ImplicitLocOpBuilder.h" // from @llvm-project
#include "mlir/IR/Operation.h"            // from @llvm-project
#include "mlir/IR/Value.h"                // from @llvm-project
#include "mlir/Support/LLVM.h"            // from @llvm-project
#include "llvm/ADT/ArrayRef.h"            // from @llvm-project
#include "llvm/Support/FormatVariadic.h"  // from @llvm-project
#include <cassert>
#include <kernel/rtlil.h> // from @at_clifford_yosys

// namespace mlir {
// namespace heir {

mlir::Operation *
LUTImporter::createOp(Yosys::RTLIL::Cell *cell,
                      llvm::SmallVector<mlir::Value, 4> &inputs,
                      circt::ImplicitLocOpBuilder &b) const {

  llvm::outs() << "Cell " << cell << ":" << cell->type.str() <<"\n";
  //assert(cell->type.begins_with("$lut"));

  // Create truth table from cell attributes.

  if(cell->type.begins_with("$lut")) {
    int lutBits;
    llvm::StringRef(cell->type.substr(4, 1)).getAsInteger(10, lutBits);
    uint64_t lutValue = 0;
    int lutSize = 1 << lutBits;
    for (int i = 0; i < lutSize; i++) {
      auto lutStr =
          cell->getPort(Yosys::RTLIL::IdString(llvm::formatv("\\P{0}", i)));
      lutValue |= (lutStr.as_bool() ? 1 : 0) << i;
    }
    llvm::outs() << "Extracting LUT value  ";
    auto lookupTable =
        b.getIntegerAttr(b.getIntegerType(lutSize, /*isSigned=*/false), lutValue);

    llvm::SmallVector<int, 1024> newcontent;
    circt::ArrayAttr attr = b.getI32ArrayAttr(newcontent);
    llvm::outs() << "LUT  "<< lookupTable <<"   " << attr <<"\n";

    return b.create<SpecHLS::LookUpTableOp>(b.getIntegerType(1), inputs.front(),
                                            attr);
  } else {
    auto type = cell->type.str();
    if (type=="$_AND_") {
      return b.create<circt::comb::AndOp>(b.getIntegerType(1), inputs);
    } else if (type=="$_OR_") {
      return b.create<circt::comb::OrOp>(b.getIntegerType(1), inputs);
    } else if (type=="$_XOR_") {
      return b.create<circt::comb::XorOp>(b.getIntegerType(1), inputs);
    } else if (type=="$_NOT_") {
      llvm::SmallVector<circt::Value> args = {inputs[0], inputs[1]};
      return b.create<circt::comb::XorOp>(b.getIntegerType(1), args);
    } else if (type=="$and") {
        return b.create<circt::comb::AndOp>(b.getIntegerType(1), inputs);
    } else if (type=="$or") {
        return b.create<circt::comb::OrOp>(b.getIntegerType(1), inputs);
      } else if (type=="$xor") {
        return b.create<circt::comb::XorOp>(b.getIntegerType(1), inputs);
      } else if (type=="$not") {
        llvm::SmallVector<circt::Value> args= {inputs[0],inputs[1]};
        return b.create<circt::comb::XorOp>(b.getIntegerType(1), args);
    } else {
      llvm::errs() << "Error : unsupported cell type " <<cell->type.str() << "\n";
      return NULL;
    }
  }

}

llvm::SmallVector<Yosys::RTLIL::SigSpec, 4>

LUTImporter::getInputs(Yosys::RTLIL::Cell *cell) const {

  // Return all non-P, non-Y named attributes.
  llvm::SmallVector<Yosys::RTLIL::SigSpec, 4> inputs;
  for (auto &conn : cell->connections()) {
    if (conn.first.contains("P") || conn.first.contains("Y")) {
      continue;
    }
    inputs.push_back(conn.second);
  }
  return inputs;
}

Yosys::RTLIL::SigSpec LUTImporter::getOutput(Yosys::RTLIL::Cell *cell) const {
  return cell->getPort(Yosys::RTLIL::IdString("\\Y"));
}

//}  // namespace heir
//}  // namespace mlir