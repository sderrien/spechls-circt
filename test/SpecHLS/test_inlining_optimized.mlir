module {
  hw.module @top(in %a : i1, in %b : i1, in %c : i1, in %d : i1, in %e : i32, out out0 : i1) {
    %0 = comb.extract %e from 6 : (i32) -> i1
    %1 = comb.extract %e from 7 : (i32) -> i1
    %2 = comb.extract %e from 8 : (i32) -> i1
    %3 = comb.extract %e from 9 : (i32) -> i1
    %c0.out0 = hw.instance "c0" @CTRL0_opt(in0: %0: i1, in1: %1: i1, in2: %2: i1) -> (out0: i1)
    hw.output %c0.out0 : i1
  }
  hw.module private @CTRL0_opt(in %in0 : i1, in %in1 : i1, in %in2 : i1, out out0 : i1) {
    %false = hw.constant false
    hw.output %false : i1
  }
}
