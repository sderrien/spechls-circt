add_mlir_library(SpecHLSToComb
        SpecHLSToSeq.cpp
        DelayConversion.cpp
        DEPENDS

        MLIRSpecHLSOpsIncGen
        SpecHLSConversionIncGen
  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRSpecHLS
  MLIRTransforms
)

target_link_libraries(SpecHLSToSeq

        PUBLIC
        CIRCTComb
        CIRCTHW
        CIRCTSupport
        )