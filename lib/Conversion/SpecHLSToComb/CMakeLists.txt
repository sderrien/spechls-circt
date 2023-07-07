add_mlir_library(SpecHLSToComb
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
        )