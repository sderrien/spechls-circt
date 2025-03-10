module {

  hw.module @SCC_0(in %in_0 : i3, in %in_1 : i2,in %in_2 : i1, in %in_3 : i3, in %in_4 : i2, in %in_5 : i1) attributes {"#pragma" = "toplevel"} {
    
    %35 = comb.concat %in_0, %in_1 : i3, i2
    %LUT = SpecHLS.lookUpTable [%35 : i5] :i3= {0,0,0,0,1,1,1,1,2,3,4,7,5,5,5,5,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7 }
    %LUT_4 = SpecHLS.lookUpTable [%LUT : i3] :i3= {0,1,2,3,4,3,5,6 }

    %55 = comb.concat %in_2, %in_3 : i1, i3
    %LUT_5 = SpecHLS.lookUpTable [%55 : i4] :i3= {0,0,0,0,0,0,0,0,1,2,3,4,5,6,6,6 }
    %56 = comb.concat %LUT_5, %in_4 : i3, i2
    %LUT_6 = SpecHLS.lookUpTable [%56 : i5] :i4= {0,0,0,0,1,1,1,1,2,2,2,2,3,4,5,8,6,6,6,6,7,7,7,7,8,8,8,8,8,8,8,8 }
    %LUT_7 = SpecHLS.lookUpTable [%LUT_6 : i4] :i3= {0,1,2,3,4,5,4,6,7,7,7,7,7,7,7,7 }

    %67 = SpecHLS.exit %in_5 live  %LUT_7:i3 ,%LUT_4:i3
    hw.output
  }
}
