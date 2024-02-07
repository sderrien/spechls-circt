add_mlir_library(SpecHLSScheduling
        MobilityPass.cpp
        DEPENDS
        SpecHLSSchedulingTransformsPassIncGen

        MLIRSpecHLSOpsIncGen
        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRSpecHLS
        MLIRTransforms
)
target_link_libraries(SpecHLSScheduling

        PUBLIC
        CIRCTComb
        CIRCTHW
        CIRCTSupport
)