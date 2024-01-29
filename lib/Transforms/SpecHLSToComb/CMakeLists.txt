add_mlir_library(SpecHLSToComb
        SpecHLSToComb.cpp
        EncoderToComb.cpp
        LUTConversion.cpp
        GammaConversion.cpp
        SpecHLSLUTToComb.cpp
        SpecHLSToComb.cpp
        DEPENDS

        MLIRSpecHLSOpsIncGen
        SpecHLSConversionIncGen
  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRSpecHLS
  MLIRTransforms
)

target_link_libraries(SpecHLSToComb

        PUBLIC
        CIRCTComb
        CIRCTHW
        CIRCTSupport
        )