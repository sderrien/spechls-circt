//
// Created by Steven on 19/01/2024.
//

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Transforms/SpecHLSConversion.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
/*
 *
 * Operations
seq.clock_div (::circt::seq::ClockDividerOp)
seq.clock_gate (::circt::seq::ClockGateOp)
seq.clock_inv (::circt::seq::ClockInverterOp)
seq.clock_mux (::circt::seq::ClockMuxOp)
seq.compreg (::circt::seq::CompRegOp)
seq.compreg.ce (::circt::seq::CompRegClockEnabledOp)
seq.const_clock (::circt::seq::ConstClockOp)
seq.fifo (::circt::seq::FIFOOp)
seq.firmem (::circt::seq::FirMemOp)
seq.firmem.read_port (::circt::seq::FirMemReadOp)
seq.firmem.read_write_port (::circt::seq::FirMemReadWriteOp)
seq.firmem.write_port (::circt::seq::FirMemWriteOp)
seq.firreg (::circt::seq::FirRegOp)
seq.from_clock (::circt::seq::FromClockOp)
seq.hlmem (::circt::seq::HLMemOp)
seq.read (::circt::seq::ReadPortOp)
seq.shiftreg (::circt::seq::ShiftRegOp)
seq.to_clock (::circt::seq::ToClockOp)
seq.write (::circt::seq::WritePortOp)
Attributes
ClockConstAttr
FirMemInitAttr
Types
ClockType
FirMemType
HLMemType
 */

static MuOp* findMuSource(Value *value) {
  auto defOp = value->getDefiningOp();
  if (defOp) {
    TypeSwitch<Operation *, bool>(defOp)
        .Case<GammaOp>([&](auto op) {
          llvm::outs() << "- node " << op << "\n";
          auto operand = op->getOperand(1);
          return findMuSource(&operand);
        })
        .Case<AlphaOp>([&](auto op) {
          llvm::outs() << "- node " << op << "\n";
          auto operand = op.getMemref();
          return findMuSource(&operand);
        })
        .Case<SyncOp>([&](auto op) {
          llvm::outs() << "- node " << op << "\n";
          auto operand = defOp->getOperand(0);
          return findMuSource(&operand);
        })
        .Case<MuOp>([&](auto op) { return op; })
        .Default([&](auto op) {
          llvm::errs() << "Operation " << *op << "is not unexpecetd\n";
          return NULL;
        });
  } else {
    return NULL;
  }
}

DelayOpToShiftRegOpConversion::DelayOpToShiftRegOpConversion(
    MLIRContext *context1, Value *clock, Value *reset)
    : OpRewritePattern(context1) {
  this->clock = clock;
  this->reset = reset;
}

LogicalResult DelayOpToShiftRegOpConversion::matchAndRewrite(
    DelayOp op, PatternRewriter &rewriter) const {
  InnerSymAttr innerSymAttr;
  auto _true = rewriter.create<circt::hw::ConstantOp>(op.getLoc(),
                                                      rewriter.getI1Type(), 0);
  auto _false = rewriter.create<circt::hw::ConstantOp>(op.getLoc(),
                                                       rewriter.getI1Type(), 0);

  auto shiftRegOp = rewriter.create<circt::seq::ShiftRegOp>(

      op.getLoc(), 10,
      op.getNext(), // Input signal to be shifted
      *this->clock, // Clock signal
      op.getEnable(), rewriter.getStringAttr("Delay"), *this->reset, _false,
      _false, innerSymAttr);

  shiftRegOp->dump();
  auto value = op->getResult(0);
  value.replaceAllUsesWith(shiftRegOp->getResult(0));
  rewriter.replaceOp(op, shiftRegOp);

  return success();
}

AlphaOpToHLWriteConversion::AlphaOpToHLWriteConversion(
    MLIRContext *context1, Value *clock, Value *reset,
    llvm::DenseMap<MuOp, circt::seq::HLMemOp> memMap)
    : OpRewritePattern(context1) {
  this->clock = clock;
  this->reset = reset;
  this->memMap = memMap;

  llvm::errs() << "AlphaOpToHLWriteConversion";
}

LogicalResult
AlphaOpToHLWriteConversion::matchAndRewrite(AlphaOp op,
                                            PatternRewriter &rewriter) const {
  InnerSymAttr innerSymAttr;
  auto _true = rewriter.create<circt::hw::ConstantOp>(op.getLoc(),
                                                      rewriter.getI1Type(), 0);
  auto _false = rewriter.create<circt::hw::ConstantOp>(op.getLoc(),
                                                       rewriter.getI1Type(), 0);

  llvm::errs() << "AlphaOpToHLWriteConversion:" << op << "\n";
  //  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
  //  &odsState, Value memory, ValueRange addresses, Value rdEn, unsigned
  //  latency); static void build(::mlir::OpBuilder &odsBuilder,
  //  ::mlir::OperationState &odsState, ::mlir::Type readData, ::mlir::Value
  //  memory, ::mlir::ValueRange addresses, /*optional*/::mlir::Value rdEn,
  //  ::mlir::IntegerAttr latency); static void build(::mlir::OpBuilder
  //  &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange
  //  resultTypes, ::mlir::Value memory, ::mlir::ValueRange addresses,
  //  /*optional*/::mlir::Value rdEn, ::mlir::IntegerAttr latency); static void
  //  build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
  //  ::mlir::Type readData, ::mlir::Value memory, ::mlir::ValueRange addresses,
  //  /*optional*/::mlir::Value rdEn, uint64_t latency); static void
  //  build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
  //  ::mlir::TypeRange resultTypes, ::mlir::Value memory, ::mlir::ValueRange
  //  addresses, /*optional*/::mlir::Value rdEn, uint64_t latency); static void
  //  build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
  //  ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
  //  ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  //
  auto memref =op.getOperand(0);
  llvm::outs() << memref << "\n";
  if (auto mu= findMuSource(&memref)) {

    if (auto hlmem = this->memMap.at(*mu)) {

      llvm::errs() << "Associated memory :" << hlmem << "\n";

      auto readop = rewriter.create<circt::seq::ReadPortOp>(
          op.getLoc(), hlmem.getResult(), op.getIndices(), _true, 1u);


      auto value = op->getResult(0);
      value.replaceAllUsesWith(readop->getResult(0));
      rewriter.replaceOp(op, readop);
      llvm::errs() << "New op  :" << readop << "\n";
      return success();
    }
  }
  return failure();

}

ArrayReadOpToHLReadConversion::ArrayReadOpToHLReadConversion(
    MLIRContext *context1, Value *clock, Value *reset,
    llvm::DenseMap<MuOp, circt::seq::HLMemOp> memMap)
    : OpRewritePattern(context1) {
  this->clock = clock;
  this->reset = reset;
  this->memMap = memMap;
}

LogicalResult ArrayReadOpToHLReadConversion::matchAndRewrite(
    ArrayReadOp op, PatternRewriter &rewriter) const {
  InnerSymAttr innerSymAttr;
  auto _true = rewriter.create<circt::hw::ConstantOp>(op.getLoc(),
                                                      rewriter.getI1Type(), 0);
  auto _false = rewriter.create<circt::hw::ConstantOp>(op.getLoc(),
                                                       rewriter.getI1Type(), 0);

  //  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
  //  &odsState, Value memory, ValueRange addresses, Value rdEn, unsigned
  //  latency); static void build(::mlir::OpBuilder &odsBuilder,
  //  ::mlir::OperationState &odsState, ::mlir::Type readData, ::mlir::Value
  //  memory, ::mlir::ValueRange addresses, /*optional*/::mlir::Value rdEn,
  //  ::mlir::IntegerAttr latency); static void build(::mlir::OpBuilder
  //  &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange
  //  resultTypes, ::mlir::Value memory, ::mlir::ValueRange addresses,
  //  /*optional*/::mlir::Value rdEn, ::mlir::IntegerAttr latency); static void
  //  build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
  //  ::mlir::Type readData, ::mlir::Value memory, ::mlir::ValueRange addresses,
  //  /*optional*/::mlir::Value rdEn, uint64_t latency); static void
  //  build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
  //  ::mlir::TypeRange resultTypes, ::mlir::Value memory, ::mlir::ValueRange
  //  addresses, /*optional*/::mlir::Value rdEn, uint64_t latency); static void
  //  build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
  //  ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
  //  ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  //
  auto mvalue= op.getMemref();
  auto mu = findMuSource(&mvalue);
  auto hlmem = this->memMap.at(*mu);

  auto readop = rewriter.create<circt::seq::ReadPortOp>(
      op.getLoc(), hlmem.getResult(), op.getIndices(), _true, 1u);

  readop->dump();
  auto value = op->getResult(0);
  value.replaceAllUsesWith(readop->getResult(0));
  rewriter.replaceOp(op, readop);

  return success();
}

MuOpToRegConversion::MuOpToRegConversion(
    MLIRContext *context1, Value *clock, Value *reset,
    llvm::DenseMap<MuOp, circt::seq::HLMemOp> &memMap)
    : OpRewritePattern(context1) {
  this->clock = clock;
  this->reset = reset;
  this->memMap = &memMap;
}

LogicalResult
MuOpToRegConversion::matchAndRewrite(MuOp op, PatternRewriter &rewriter) const {
  InnerSymAttr innerSymAttr;
  auto _true = rewriter.create<circt::hw::ConstantOp>(op.getLoc(),
                                                      rewriter.getI1Type(), 0);
  auto _false = rewriter.create<circt::hw::ConstantOp>(op.getLoc(),
                                                       rewriter.getI1Type(), 0);
  llvm::errs() << "Associated MuOp :" << op << "\n";
  if (!clock ) {
    llvm::errs() << "no registerd clock\n";
    return failure();
  }


  if (auto memRefType = dyn_cast<MemRefType>(op.getType())) {

//    auto reg = rewriter.create<circt::seq::CompRegOp>(op.getLoc(), op.getNext(),
//                                                      *clock);

    auto hlMemType = rewriter.getType<seq::HLMemType>(memRefType.getShape(),memRefType.getElementType());
    auto mem = rewriter.create<circt::seq::HLMemOp>(
        op.getLoc(), hlMemType, *clock, *reset, op.getNameAttr().getValue());
    memMap->insert(std::make_pair(op, mem));

    llvm::errs() << "created memory " << mem;
//    reg->dump();
    auto value = op->getResult(0);
    value.replaceAllUsesWith(mem->getResult(0));
    rewriter.replaceOp(op, mem);
//
    return success();

  } else {

    auto reg = rewriter.create<circt::seq::CompRegOp>(op.getLoc(), op.getNext(),
                                                      *clock);

    reg->dump();
    auto value = op->getResult(0);
    value.replaceAllUsesWith(reg->getResult(0));
    rewriter.replaceOp(op, reg);

    return success();
  }
}