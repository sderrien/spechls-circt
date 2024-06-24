// RUN: spechls-opt --canonicalize %s | spechls-opt | FileCheck %s
module {

  hw.module @SCC_0(
     in %enable : i1, in %value : i32, in %address : i32, in %array:memref<16xi32>,
   out result : memref<16xi32>)

   {
    %index = arith.index_cast %address : i32 to index
    %31 = SpecHLS.alpha @x : %enable -> %array[%index], %value : memref<16xi32>
    hw.output %31 :memref<16xi32>
  }
}
