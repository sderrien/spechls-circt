add_mlir_dialect_library(SpecHLSScheduleDialect
        ScheduleDialectDialect.cpp
        ScheduleDialectOps.cpp

        ADDITIONAL_HEADER_DIRS
        ../../../include/Dialect/SpecHLS

        DEPENDS
        MLIRSpecHLSOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
)