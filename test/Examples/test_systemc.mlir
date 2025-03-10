module {
  emitc.include <"systemc.h">
  systemc.module @CTRL0 (%in0: !systemc.in<i1>, %in1: !systemc.in<i1>, %in2: !systemc.in<i1>, %out0: !systemc.out<i1>) {
    systemc.ctor {
      systemc.method %innerLogic
      systemc.sensitive %in0, %in1, %in2 : !systemc.in<i1>, !systemc.in<i1>, !systemc.in<i1>
    }
    %innerLogic = systemc.func {
      %0 = systemc.signal.read %in0 : !systemc.in<i1>
      %1 = systemc.convert %0 : (i1) -> i1
      %2 = systemc.signal.read %in1 : !systemc.in<i1>
      %3 = systemc.convert %2 : (i1) -> i1
      %4 = systemc.signal.read %in2 : !systemc.in<i1>
      %5 = systemc.convert %4 : (i1) -> i1
      %6 = comb.or %3, %1 : i1
      %7 = comb.and %6, %5 : i1
      %8 = comb.xor %6, %7 : i1
      %9 = comb.and %3, %8 : i1
      %10 = systemc.convert %9 : (i1) -> i1
      systemc.signal.write %out0, %10 : !systemc.out<i1>
    }
  }
  systemc.module @top (%a: !systemc.in<i1>, %b: !systemc.in<i1>, %c: !systemc.in<i1>, %d: !systemc.in<i1>, %e: !systemc.in<!systemc.uint<32>>, %out0: !systemc.out<i1>) {
    %c0 = systemc.instance.decl  @CTRL0 : !systemc.module<CTRL0(in0: !systemc.in<i1>, in1: !systemc.in<i1>, in2: !systemc.in<i1>, out0: !systemc.out<i1>)>
    %c0_in0 = systemc.signal  : !systemc.signal<i1>
    %c0_in1 = systemc.signal  : !systemc.signal<i1>
    %c0_in2 = systemc.signal  : !systemc.signal<i1>
    %c0_out0 = systemc.signal  : !systemc.signal<i1>
    systemc.ctor {
      systemc.method %innerLogic
      systemc.sensitive %a, %b, %c, %d, %e : !systemc.in<i1>, !systemc.in<i1>, !systemc.in<i1>, !systemc.in<i1>, !systemc.in<!systemc.uint<32>>
      systemc.instance.bind_port %c0["in0"] to %c0_in0 : !systemc.module<CTRL0(in0: !systemc.in<i1>, in1: !systemc.in<i1>, in2: !systemc.in<i1>, out0: !systemc.out<i1>)>, !systemc.signal<i1>
      systemc.instance.bind_port %c0["in1"] to %c0_in1 : !systemc.module<CTRL0(in0: !systemc.in<i1>, in1: !systemc.in<i1>, in2: !systemc.in<i1>, out0: !systemc.out<i1>)>, !systemc.signal<i1>
      systemc.instance.bind_port %c0["in2"] to %c0_in2 : !systemc.module<CTRL0(in0: !systemc.in<i1>, in1: !systemc.in<i1>, in2: !systemc.in<i1>, out0: !systemc.out<i1>)>, !systemc.signal<i1>
      systemc.instance.bind_port %c0["out0"] to %c0_out0 : !systemc.module<CTRL0(in0: !systemc.in<i1>, in1: !systemc.in<i1>, in2: !systemc.in<i1>, out0: !systemc.out<i1>)>, !systemc.signal<i1>
    }
    %innerLogic = systemc.func {
      %0 = systemc.signal.read %a : !systemc.in<i1>
      %1 = systemc.convert %0 : (i1) -> i1
      %2 = systemc.signal.read %b : !systemc.in<i1>
      %3 = systemc.convert %2 : (i1) -> i1
      %4 = systemc.signal.read %c : !systemc.in<i1>
      %5 = systemc.convert %4 : (i1) -> i1
      %6 = systemc.signal.read %d : !systemc.in<i1>
      %7 = systemc.convert %6 : (i1) -> i1
      %8 = systemc.signal.read %e : !systemc.in<!systemc.uint<32>>
      %9 = systemc.convert %8 : (!systemc.uint<32>) -> i32
      %10 = comb.extract %9 from 6 : (i32) -> i1
      %11 = comb.extract %9 from 7 : (i32) -> i1
      %12 = comb.extract %9 from 8 : (i32) -> i1
      %13 = comb.extract %9 from 9 : (i32) -> i1
      systemc.signal.write %c0_in0, %10 : !systemc.signal<i1>
      systemc.signal.write %c0_in1, %11 : !systemc.signal<i1>
      systemc.signal.write %c0_in2, %12 : !systemc.signal<i1>
      %14 = systemc.signal.read %c0_out0 : !systemc.signal<i1>
      %15 = systemc.convert %14 : (i1) -> i1
      systemc.signal.write %out0, %15 : !systemc.out<i1>
    }
  }
}

