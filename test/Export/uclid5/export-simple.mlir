module {

SpecHLS.hkernel @IDG_SRC_0 -> {
    %x = SpecHLS.init @x : i1
	%t82:1 = SpecHLS.htask @SpecSCC_13(%x: i1) -> (i1) {
	^body(%arg0: i1):

	        %t1 = SpecHLS.init @__guard : i1
	        %t2 = comb.xor %t1,%arg0 : i1
	        SpecHLS.commit (%arg0:i1) when %t2
		}
	SpecHLS.htask @SCC_14(%t82#0: i1) -> () {
			^body(%arg0: i1):

			SpecHLS.commit when %arg0
	}
	SpecHLS.exit %t82 live  %t82:i1
}
}
