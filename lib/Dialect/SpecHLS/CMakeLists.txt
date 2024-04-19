add_mlir_dialect_library(MLIRSpecHLS
        SpecHLSDialect.cpp
        SpecHLSOps.cpp
        SpecHLSTypes.cpp
        SpecHLSUtils.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../include/Dialect/SpecHLS

        DEPENDS
        MLIRSpecHLSOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
)