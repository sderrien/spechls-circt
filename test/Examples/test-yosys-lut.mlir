// RUN: spechls-opt --yosys-optimizer="replace-with-optimized-module=true"  %s | spechls-opt | FileCheck %s
module {


  hw.module @test_LUT1(in %in_0 : i3, out out_0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
    %1 = SpecHLS.lookUpTable [%in_0:i3] : i1= {0,1,1,0,1,0,1,0 }
    hw.output %1 : i1
  }
//
//   hw.module @test_LUT2(in %in_0 : i3, out out_0 : i1) attributes {"#pragma" = "CONTROL_NODE"} {
//    %1 = SpecHLS.lookUpTable [%in_0:i3] : i1= {1,0,0,0,1,0,0,0 }
//    hw.output %1 : i1
//  }
//
//   hw.module @test_LUT3(in %in_0 : i3, out out_0 : i3) attributes {"#pragma" = "CONTROL_NODE"} {
//    %1 = SpecHLS.lookUpTable [%in_0:i3] : i3= {0,1,2,3,4,5,6,7 }
//    hw.output %1 : i3
//  }
//
//   hw.module @test_LUT4(in %in_0 : i3, out out_0 : i3) attributes {"#pragma" = "CONTROL_NODE"} {
//    %1 = SpecHLS.lookUpTable [%in_0:i3] : i3= {7,6,5,4,3,2,1,0 }
//    hw.output %1 : i3
//  }

   hw.module @test_LUT4(in %in_0 : i2, out out_0 : i2) attributes {"#pragma" = "CONTROL_NODE"} {
    %1 = SpecHLS.lookUpTable [%in_0:i2] : i2= {0,1,2,3 }
    hw.output %1 : i2
  }

   hw.module @WTF_LUT5(in %in_0 : i2, out out_0 : i2) attributes {"#pragma" = "CONTROL_NODE"} {
    %1 = SpecHLS.lookUpTable [%in_0:i2] : i2= {3,2,1,0 }
    hw.output %1 : i2
  }
}
