add_mlir_library(SpecHLSTransforms
		MergeGammas.cpp
		MergeLUTs.cpp
		DEPENDS

		MLIRSpecHLSOpsIncGen
		SpecHLSTransformsPassIncGen
		LINK_COMPONENTS
		Core

		LINK_LIBS PUBLIC
		MLIRSpecHLS
		MLIRTransforms
		)
target_link_libraries(SpecHLSTransforms

		PUBLIC
		CIRCTComb
		CIRCTHW
		CIRCTSupport
		)