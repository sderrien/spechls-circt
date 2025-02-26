
module {
  hw.module @SCC_0(out out_0 : i32) {
    %sel = SpecHLS.init @sel : i2

    %c = SpecHLS.init @c : i32

    %addr = SpecHLS.init @addr : i4

    %tab = SpecHLS.init @tab : memref<16xi32>
    %index = arith.index_cast %addr : i4 to index
    %read = SpecHLS.read %tab : memref<16xi32>[%index]

    %gamma = SpecHLS.gamma @dummy %sel ? %read,%c,%read,%c :i32
    hw.output %gamma :i32
  }

}

