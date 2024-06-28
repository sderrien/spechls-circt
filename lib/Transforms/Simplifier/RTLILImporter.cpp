#include "RTLILImporter.h"

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Dialect/SpecHLS/SpecHLSUtils.h"
#include "Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/Namespace.h"
#include "kernel/rtlil.h"                  // from @at_clifford_yosys
#include "mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h" // from @llvm-project
#include "mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/Transforms/FoldUtils.h"     // from @llvm-project
#include "llvm/ADT/MapVector.h"            // from @llvm-project
#include "llvm/Support/FormatVariadic.h"   // from @llvm-project
namespace mlir {

using ::Yosys::RTLIL::Module;
using ::Yosys::RTLIL::SigSpec;
using ::Yosys::RTLIL::Wire;

#define VERBOSE false
namespace {

// getTypeForWire gets the MLIR type corresponding to the RTLIL wire. If the
// wire is an integer with multiple bits, then the MLIR type is a tensor of
// bits.
Type getTypeForWire(OpBuilder &b, Wire *wire) {
  return b.getIntegerType(wire->width);
}

} // namespace

llvm::SmallVector<std::string, 10>
getTopologicalOrder(std::stringstream &torderOutput) {
  llvm::SmallVector<std::string, 10> cells;
  std::string line;
  while (std::getline(torderOutput, line)) {
    auto lineCell = line.find("cell ");
    if (lineCell != std::string::npos) {
      cells.push_back(line.substr(lineCell + 5, std::string::npos));
    }
  }
  return cells;
}

mlir::Operation *
RTLILImporter::createOp(Yosys::RTLIL::Cell *cell,
                        llvm::SmallVector<mlir::Value, 4> &inputs,
                        circt::ImplicitLocOpBuilder &b) {

  if (VERBOSE)
    llvm::outs() << "Cell " << cell << ":" << cell->type.str() << "\n";
  // assert(cell->type.begins_with("$lut"));

  // Create truth table from cell attributes.

  if (cell->type.begins_with("$lut")) {
    int lutBits;
    llvm::StringRef(cell->type.substr(4, 1)).getAsInteger(10, lutBits);
    uint64_t lutValue = 0;
    int lutSize = 1 << lutBits;
    for (int i = 0; i < lutSize; i++) {
      auto lutStr =
          cell->getPort(Yosys::RTLIL::IdString(llvm::formatv("\\P{0}", i)));
      lutValue |= (lutStr.as_bool() ? 1 : 0) << i;
    }
    if (VERBOSE)
      llvm::outs() << "Extracting LUT value  ";
    auto lookupTable = b.getIntegerAttr(
        b.getIntegerType(lutSize, /*isSigned=*/false), lutValue);

    llvm::SmallVector<int, 1024> newcontent;
    circt::ArrayAttr attr = b.getI32ArrayAttr(newcontent);
    if (VERBOSE)
      llvm::outs() << "LUT  " << lookupTable << "   " << attr << "\n";

    return b.create<SpecHLS::LookUpTableOp>(b.getIntegerType(1), inputs.front(),
                                            attr);
  } else {
    auto type = cell->type.str();
    if (type == "$_AND_") {
      return b.create<circt::comb::AndOp>(b.getIntegerType(1), inputs);
    } else if (type == "$_OR_") {
      return b.create<circt::comb::OrOp>(b.getIntegerType(1), inputs);
    } else if (type == "$_XOR_") {
      return b.create<circt::comb::XorOp>(b.getIntegerType(1), inputs);
    } else if (type == "$_NOT_") {
        auto allOnes = b.create<circt::hw::ConstantOp>(b.getIntegerType(1), -1);
      llvm::SmallVector<circt::Value> args = {inputs[0], allOnes};
      return b.create<circt::comb::XorOp>(b.getIntegerType(1), args);
    } else if (type == "$and") {
      return b.create<circt::comb::AndOp>(b.getIntegerType(1), inputs);
    } else if (type == "$or") {
      return b.create<circt::comb::OrOp>(b.getIntegerType(1), inputs);
    } else if (type == "$xor") {
      return b.create<circt::comb::XorOp>(b.getIntegerType(1), inputs);
    } else if (type == "$not") {
      llvm::SmallVector<circt::Value> args = {inputs[0], inputs[1]};
      return b.create<circt::comb::XorOp>(b.getIntegerType(1), args);
    } else {
      llvm::errs() << "Error : unsupported cell type " << cell->type.str()
                   << "\n";
      return NULL;
    }
  }
}

llvm::SmallVector<Yosys::RTLIL::SigSpec, 4>

RTLILImporter::getInputs(Yosys::RTLIL::Cell *cell) {

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

Yosys::RTLIL::SigSpec RTLILImporter::getOutput(Yosys::RTLIL::Cell *cell) {
  return cell->getPort(Yosys::RTLIL::IdString("\\Y"));
}

void RTLILImporter::addWireValue(Wire *wire, Value value) {
  wireNameToValue.insert(std::make_pair(wire->name.str(), value));
}

Value RTLILImporter::getWireValue(Wire *wire) {
  auto wireName = wire->name.str();
  assert(wireNameToValue.contains(wireName));
  return wireNameToValue.at(wireName);
}

Value RTLILImporter::getBit(
    const SigSpec &conn, ImplicitLocOpBuilder &b,
    llvm::MapVector<Wire *, SmallVector<Value>> &retBitValues) {
  // Because the cells are in topological order, and Yosys should have
  // removed redundant wire-wire mappings, the cell's inputs must be a bit
  // of an input wire, in the map of already defined wires (which are
  // bits), or a constant bit.
  if (!(conn.is_wire() || conn.is_fully_const() || conn.is_bit())) {
    if (VERBOSE)
      llvm::outs() << " connection " << conn.as_string() << "\n";
  }
  assert(conn.is_wire() || conn.is_fully_const() || conn.is_bit());
  if (conn.is_wire()) {
    auto name = conn.as_wire()->name.str();
    assert(wireNameToValue.contains(name));
    return wireNameToValue[name];
  }
  if (conn.is_fully_const()) {
    auto bit = conn.as_const();
    auto constantOp = b.createOrFold<circt::hw::ConstantOp>(
        b.getIntegerAttr(b.getIntegerType(1), bit.as_int()));
    return constantOp;
  }
  // Extract the bit of the multi-bit input or output wire.
  assert(conn.as_bit().is_wire());
  auto bit = conn.as_bit();
  if (retBitValues.contains(bit.wire)) {
    auto offset = retBitValues[bit.wire].size() - bit.offset - 1;
    return retBitValues[bit.wire][offset];
  }
  auto argA = getWireValue(bit.wire);
  auto extractOp = b.create<circt::comb::ExtractOp>(argA, bit.offset, 1);
  return extractOp;
}

void RTLILImporter::addResultBit(
    const SigSpec &conn, Value result,
    llvm::MapVector<Wire *, SmallVector<Value>> &retBitValues) {
  assert(conn.is_wire() || conn.is_bit());
  if (conn.is_wire()) {
    addWireValue(conn.as_wire(), result);
    return;
  }
  // This must be a bit of the multi-bit output wire.
  auto bit = conn.as_bit();
  assert(bit.is_wire() && retBitValues.contains(bit.wire));
  auto offset = retBitValues[bit.wire].size() - bit.offset - 1;
  retBitValues[bit.wire][offset] = result;
}

circt::hw::HWModuleOp
RTLILImporter::importModule(circt::hw::HWModuleOp op, Module *module,
                            const SmallVector<std::string, 10> &cellOrdering) {
  // Gather input and output wires of the module to match up with the block
  // arguments.
  SmallVector<Type, 4> argTypes;
  SmallVector<Wire *, 4> wireArgs;
  SmallVector<Type, 4> retTypes;
  SmallVector<Wire *, 4> wireRet;

  OpBuilder builder(context);
  // Maintain a map from RTLIL output wires to the Values that comprise it
  // in order to reconstruct the multi-bit output.
  llvm::MapVector<Wire *, SmallVector<Value>> retBitValues;
  for (auto *wire : module->wires()) {
    // The RTLIL module may also have intermediate wires that are neither inputs
    // nor outputs.
    if (wire->port_input) {
      argTypes.push_back(getTypeForWire(builder, wire));
      wireArgs.push_back(wire);
    } else if (wire->port_output) {
      retTypes.push_back(getTypeForWire(builder, wire));
      wireRet.push_back(wire);
      retBitValues[wire].resize(wire->width);
    }
  }

  SmallVector<circt::hw::PortInfo> ports;
  size_t id = 0;
  for (auto argtype : argTypes) {
    // auto port = ;
    auto label = op.getPort(id).getName();
    if (!op.getPort(id).isInput()) {
      // expects input port followed by output ports
      llvm::errs() << "Interface mismatch\n";
    }
    auto portName = builder.getStringAttr(label);
    ports.push_back(
        {{portName, argtype, circt::hw::ModulePort::Direction::Input}, id});
    id++;
  }

  for (auto restype : retTypes) {
    auto label = op.getPort(id).getName();
    if (!op.getPort(id).isOutput()) {
      // expects input port followed by output ports
      llvm::errs() << "Interface mismatch\n";
    }
    auto portName = builder.getStringAttr(label);
    ports.push_back(
        {{portName, restype, circt::hw::ModulePort::Direction::Output}, id});
    id++;
  }
  StringAttr nameAttr = builder.getStringAttr(
      module->name.str().replace(0, 1, "") + std::string("_opt"));

  auto submodule = builder.create<circt::hw::HWModuleOp>(
      builder.getUnknownLoc(), nameAttr, ports);

  mlir::Block *block = submodule.getBodyBlock();
  // Map the RTLIL wires to the block arguments' Values.
  for (auto i = 0; i < wireArgs.size(); i++) {
    addWireValue(wireArgs[i], block->getArgument(i));
  }
  auto b = ImplicitLocOpBuilder::atBlockBegin(submodule.getLoc(), block);

  // Convert cells to Operations according to topological order.
  for (const auto &cellName : cellOrdering) {

    assert(module->cells_.count(cellName) != 0 &&
           "expected cell in RTLIL design");
    auto *cell = module->cells_[cellName];

    SmallVector<Value, 4> inputValues;

    for (const auto &conn : getInputs(cell)) {
      if (conn.is_wire()) {
        auto wire = conn.as_wire();
        auto name = wire->name.str();
        auto bit = getBit(conn, b, retBitValues);
        inputValues.push_back(bit);
      } else {
        auto bit = getBit(conn, b, retBitValues);
        inputValues.push_back(bit);
      }
    }
    auto *op = createOp(cell, inputValues, b);
    if (VERBOSE)
      llvm::outs() << "op created " << *op << "\n";
    auto resultConn = getOutput(cell);
    addResultBit(resultConn, op->getResult(0), retBitValues);
  }

  // Wire up remaining connections.
  for (const auto &conn : module->connections()) {
    auto output = conn.first;
    // These must be output wire connections (either an output bit or a bit of a
    // multi-bit output wire).
    assert(output.is_wire() || output.as_chunk().is_wire() ||
           output.as_bit().is_wire());
    if ((output.is_chunk() && !output.is_wire()) ||
        ((conn.second.is_chunk() && !conn.second.is_wire()) ||
         conn.second.chunks().size() > 1)) {
      // If one of the RHS or LHS is a chunk of a wire (and not a whole wire) OR
      // contains multiple chunks, then iterate bit by bit to assign the result
      // bits.
      for (auto i = 0; i < output.size(); i++) {
        Value connValue = getBit(conn.second.bits().at(i), b, retBitValues);
        addResultBit(output.bits().at(i), connValue, retBitValues);
      }
    } else {
      // This may be a single bit, a chunk of a wire, or a whole wire.
      Value connValue = getBit(conn.second, b, retBitValues);
      addResultBit(output, connValue, retBitValues);
    }
  }

  // Concatenate result bits if needed, and return result.
  SmallVector<Value, 4> returnValues;
  for (const auto &[resultWire, retBits] : retBitValues) {
    // If we are returning a whole wire as is (e.g. the input wire) or a single
    // bit, we do not need to concat any return bits.
    if (wireNameToValue.contains(resultWire->name.str())) {
      returnValues.push_back(getWireValue(resultWire));
    } else {
      // We are in a multi-bit scenario.
      assert(retBits.size() > 1);
      auto concatOp = b.create<circt::comb::ConcatOp>(retBits);
      returnValues.push_back(concatOp.getResult());
    }
  }

  circt::hw::OutputOp outOp = cast<circt::hw::OutputOp>(block->getTerminator());

  for (auto retVal : llvm::enumerate(returnValues)) {
    // if (VERBOSE) llvm::outs() << "out_" + std::to_string(retVal.index()) << "
    // " << retVal.value() << "\n";
    outOp->insertOperands(retVal.index(), retVal.value());
  }
  return submodule;
}

} // namespace mlir
