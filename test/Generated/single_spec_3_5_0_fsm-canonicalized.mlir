module {
  %false = hw.constant false
  %c1_i8 = hw.constant 1 : i8
  %c0_i8 = hw.constant 0 : i8
  %true = hw.constant true
  fsm.machine @SpecSCC_13_fsm(%arg0: i8, %arg1: i8) -> (i8, i8, i8, i1, i1, i1, i8, i8, i1, i8, i8, i1) attributes {initialState = "Init0"} {
    fsm.state @Proceed output {
      fsm.output %c0_i8, %c0_i8, %c0_i8, %true, %true, %true, %c0_i8, %c0_i8, %false, %c1_i8, %c0_i8, %false : i8, i8, i8, i1, i1, i1, i8, i8, i1, i8, i8, i1
    } transitions {
      fsm.transition @x1__Rollback guard {
        %0 = comb.icmp eq %arg0, %c1_i8 : i8
        fsm.return %0
      }
      fsm.transition @x00__Rollback guard {
        %0 = comb.icmp eq %arg1, %c0_i8 : i8
        fsm.return %0
      }
    }
    fsm.state @x1__Rollback output {
      fsm.output %c0_i8, %c0_i8, %c0_i8, %true, %false, %true, %c0_i8, %c0_i8, %false, %c1_i8, %c0_i8, %false : i8, i8, i8, i1, i1, i1, i8, i8, i1, i8, i8, i1
    } transitions {
      fsm.transition @x1__Fill0 guard {
        fsm.return %true
      }
    }
    fsm.state @x1__Fill0 output {
      fsm.output %c0_i8, %c0_i8, %c0_i8, %true, %false, %true, %c0_i8, %c0_i8, %false, %c1_i8, %c0_i8, %false : i8, i8, i8, i1, i1, i1, i8, i8, i1, i8, i8, i1
    } transitions {
      fsm.transition @x1__Fill1 guard {
        fsm.return %true
      }
    }
    fsm.state @x1__Fill1 output {
      fsm.output %c0_i8, %c0_i8, %c0_i8, %true, %true, %true, %c0_i8, %c0_i8, %false, %c1_i8, %c0_i8, %false : i8, i8, i8, i1, i1, i1, i8, i8, i1, i8, i8, i1
    } transitions {
      fsm.transition @Proceed guard {
        fsm.return %true
      }
      fsm.transition @x1_x00__Rollback guard {
        %0 = comb.icmp eq %arg1, %c0_i8 : i8
        fsm.return %0
      }
    }
    fsm.state @x00__Rollback output {
      fsm.output %c0_i8, %c0_i8, %c0_i8, %true, %true, %false, %c0_i8, %c0_i8, %false, %c1_i8, %c0_i8, %false : i8, i8, i8, i1, i1, i1, i8, i8, i1, i8, i8, i1
    } transitions {
      fsm.transition @x00__Fill0 guard {
        fsm.return %true
      }
    }
    fsm.state @x00__Fill0 output {
      fsm.output %c0_i8, %c0_i8, %c0_i8, %true, %true, %false, %c0_i8, %c0_i8, %false, %c1_i8, %c0_i8, %false : i8, i8, i8, i1, i1, i1, i8, i8, i1, i8, i8, i1
    } transitions {
      fsm.transition @x00__Fill1 guard {
        fsm.return %true
      }
    }
    fsm.state @x00__Fill1 output {
      fsm.output %c0_i8, %c0_i8, %c0_i8, %true, %true, %true, %c0_i8, %c0_i8, %false, %c1_i8, %c0_i8, %false : i8, i8, i8, i1, i1, i1, i8, i8, i1, i8, i8, i1
    } transitions {
      fsm.transition @Proceed guard {
        fsm.return %true
      }
      fsm.transition @x1__Rollback guard {
        %0 = comb.icmp eq %arg0, %c1_i8 : i8
        fsm.return %0
      }
    }
    fsm.state @x1_x00__Rollback output {
      fsm.output %c0_i8, %c0_i8, %c0_i8, %true, %true, %false, %c0_i8, %c0_i8, %false, %c1_i8, %c0_i8, %false : i8, i8, i8, i1, i1, i1, i8, i8, i1, i8, i8, i1
    } transitions {
      fsm.transition @x1_x00__Fill0 guard {
        fsm.return %true
      }
    }
    fsm.state @x1_x00__Fill0 output {
      fsm.output %c0_i8, %c0_i8, %c0_i8, %true, %true, %false, %c0_i8, %c0_i8, %false, %c1_i8, %c0_i8, %false : i8, i8, i8, i1, i1, i1, i8, i8, i1, i8, i8, i1
    } transitions {
      fsm.transition @x1_x00__Fill1 guard {
        fsm.return %true
      }
    }
    fsm.state @x1_x00__Fill1 output {
      fsm.output %c0_i8, %c0_i8, %c0_i8, %true, %true, %true, %c0_i8, %c0_i8, %false, %c1_i8, %c0_i8, %false : i8, i8, i8, i1, i1, i1, i8, i8, i1, i8, i8, i1
    } transitions {
      fsm.transition @Proceed guard {
        fsm.return %true
      }
    }
    fsm.state @Init0 output {
      fsm.output %c0_i8, %c0_i8, %c0_i8, %true, %false, %false, %c0_i8, %c0_i8, %false, %c1_i8, %c0_i8, %false : i8, i8, i8, i1, i1, i1, i8, i8, i1, i8, i8, i1
    } transitions {
      fsm.transition @Init1 guard {
        fsm.return %true
      }
    }
    fsm.state @Init1 output {
      fsm.output %c0_i8, %c0_i8, %c0_i8, %true, %false, %false, %c0_i8, %c0_i8, %false, %c1_i8, %c0_i8, %false : i8, i8, i8, i1, i1, i1, i8, i8, i1, i8, i8, i1
    } transitions {
      fsm.transition @Proceed guard {
        fsm.return %true
      }
    }
  }
}

