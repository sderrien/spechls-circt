add_mlir_dialect_library(MLIRSpecHLS
        SpecHLSDialect.cpp
        SpecHLSOps.cpp
		SpecHLSTypes.cpp
		SpecHLSUtils.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/SpecHLS

        DEPENDS
        MLIRSpecHLSOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)
