//
// Created by Steven on 19/01/2024.
//

#include "Dialect/SpecHLS/SpecHLSDialect.h"
#include "Dialect/SpecHLS/SpecHLSOps.h"
#include "Transforms/SpecHLSConversion.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "circt/Dialect/HW/HWTypes.h"
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


static MuOp* findMuSource(Operation* op)  {


  if (op) {
    if (auto mu = dyn_cast<MuOp>(op)) {
      return &mu;
    }
    if (auto gamma = dyn_cast<GammaOp>(op)) {
      if (auto operand = op -> getOperand(1)) {
        auto defOp = operand.getDefiningOp();
        if (defOp) {
          return findMuSource(defOp);
        }
      }
      return NULL;
    } else {
      if (auto operand = op -> getOperand(0)) {
        auto defOp= operand.getDefiningOp();
        if (defOp) {
          findMuSource(defOp);
        }
      }

    }
  } else {
    return NULL;
  }
}

DelayOpToShiftRegOpConversion::DelayOpToShiftRegOpConversion(
    MLIRContext *context1, Value *clock, Value *reset) : OpRewritePattern(context1) {
  this->clock=clock;
  this->reset=reset;

}

LogicalResult DelayOpToShiftRegOpConversion::matchAndRewrite(DelayOp op, PatternRewriter &rewriter) const {
    InnerSymAttr innerSymAttr ;
    auto _true = rewriter.create<circt::hw::ConstantOp>(op.getLoc(),rewriter.getI1Type(),0);
    auto _false = rewriter.create<circt::hw::ConstantOp>(op.getLoc(),rewriter.getI1Type(),0);

    auto shiftRegOp = rewriter.create<circt::seq::ShiftRegOp>(

        op.getLoc(),
        10,
        op.getNext(),             // Input signal to be shifted
        *this->clock,            // Clock signal
        op.getEnable(),
        rewriter.getStringAttr("Delay"),
        *this->reset,
        _false,
        _false,
        innerSymAttr
    );

    shiftRegOp->dump();
    auto value = op->getResult(0);
    value.replaceAllUsesWith(shiftRegOp->getResult(0));
    rewriter.replaceOp(op, shiftRegOp);

    return success();

}

AlphaOpToHLWriteConversion::AlphaOpToHLWriteConversion(
    MLIRContext *context1, Value *clock, Value *reset,llvm::DenseMap<MuOp,circt::seq::HLMemOp> memMap) : OpRewritePattern(context1) {
  this->clock=clock;
  this->reset=reset;
  this->memMap=memMap;

  llvm::errs() << "AlphaOpToHLWriteConversion";
}

LogicalResult AlphaOpToHLWriteConversion::matchAndRewrite(AlphaOp op, PatternRewriter &rewriter) const {
  InnerSymAttr innerSymAttr ;
  auto _true = rewriter.create<circt::hw::ConstantOp>(op.getLoc(),rewriter.getI1Type(),0);
  auto _false = rewriter.create<circt::hw::ConstantOp>(op.getLoc(),rewriter.getI1Type(),0);


  llvm::errs() << "AlphaOpToHLWriteConversion:" << op << "\n";
//  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memory, ValueRange addresses, Value rdEn, unsigned latency);
//  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type readData, ::mlir::Value memory, ::mlir::ValueRange addresses, /*optional*/::mlir::Value rdEn, ::mlir::IntegerAttr latency);
//  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memory, ::mlir::ValueRange addresses, /*optional*/::mlir::Value rdEn, ::mlir::IntegerAttr latency);
//  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type readData, ::mlir::Value memory, ::mlir::ValueRange addresses, /*optional*/::mlir::Value rdEn, uint64_t latency);
//  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memory, ::mlir::ValueRange addresses, /*optional*/::mlir::Value rdEn, uint64_t latency);
//  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
//

  auto hlmem = this->memMap.at(*findMuSource(op));

  llvm::errs() << "Associated memory :" << hlmem << "\n";


  auto readop = rewriter.create<circt::seq::ReadPortOp>(
      op.getLoc(),
      hlmem.getResult(),
      op.getIndices(),
      _true,
      1u
  );

  readop->dump();
  auto value = op->getResult(0);
  value.replaceAllUsesWith(readop->getResult(0));
  rewriter.replaceOp(op, readop);

  llvm::errs() << "New op  :" << readop << "\n";
  return success();

}

ArrayReadOpToHLReadConversion::ArrayReadOpToHLReadConversion(
    MLIRContext *context1, Value *clock, Value *reset,llvm::DenseMap<MuOp,circt::seq::HLMemOp> memMap) : OpRewritePattern(context1) {
  this->clock=clock;
  this->reset=reset;
  this->memMap=memMap;

}


LogicalResult ArrayReadOpToHLReadConversion::matchAndRewrite(ArrayReadOp op, PatternRewriter &rewriter) const {
  InnerSymAttr innerSymAttr ;
  auto _true = rewriter.create<circt::hw::ConstantOp>(op.getLoc(),rewriter.getI1Type(),0);
  auto _false = rewriter.create<circt::hw::ConstantOp>(op.getLoc(),rewriter.getI1Type(),0);

  //  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memory, ValueRange addresses, Value rdEn, unsigned latency);
  //  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type readData, ::mlir::Value memory, ::mlir::ValueRange addresses, /*optional*/::mlir::Value rdEn, ::mlir::IntegerAttr latency);
  //  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memory, ::mlir::ValueRange addresses, /*optional*/::mlir::Value rdEn, ::mlir::IntegerAttr latency);
  //  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type readData, ::mlir::Value memory, ::mlir::ValueRange addresses, /*optional*/::mlir::Value rdEn, uint64_t latency);
  //  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memory, ::mlir::ValueRange addresses, /*optional*/::mlir::Value rdEn, uint64_t latency);
  //  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  //

  auto mu = findMuSource(op);
  auto hlmem = this->memMap.at(*mu);

  auto readop = rewriter.create<circt::seq::ReadPortOp>(
      op.getLoc(),
      hlmem.getResult(),
      op.getIndices(),
      _true,
      1u
  );

  readop->dump();
  auto value = op->getResult(0);
  value.replaceAllUsesWith(readop->getResult(0));
  rewriter.replaceOp(op, readop);

  return success();

}

MuOpToRegConversion::MuOpToRegConversion(
    MLIRContext *context1, Value *clock, Value *reset,llvm::DenseMap<MuOp,circt::seq::HLMemOp> memMap) : OpRewritePattern(context1) {
  this->clock=clock;
  this->reset=reset;
  this->memMap=memMap;

}

LogicalResult MuOpToRegConversion::matchAndRewrite(MuOp op, PatternRewriter &rewriter) const {
  InnerSymAttr innerSymAttr ;
  auto _true = rewriter.create<circt::hw::ConstantOp>(op.getLoc(),rewriter.getI1Type(),0);
  auto _false = rewriter.create<circt::hw::ConstantOp>(op.getLoc(),rewriter.getI1Type(),0);


  if (op.getType().isa<MemRefType>()) {

    auto reg = rewriter.create<circt::seq::CompRegOp>(
        op.getLoc(),
        op.getNext(),
        *clock
    );

    reg->dump();
    auto value = op->getResult(0);
    value.replaceAllUsesWith(reg->getResult(0));
    rewriter.replaceOp(op, reg);

    return success();

  } else {

    auto reg = rewriter.create<circt::seq::CompRegOp>(
        op.getLoc(),
        op.getNext(),
        *clock
    );

    reg->dump();
    auto value = op->getResult(0);
    value.replaceAllUsesWith(reg->getResult(0));
    rewriter.replaceOp(op, reg);

    return success();

  }

}