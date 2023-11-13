add_mlir_library(SpecHLSToArc
        SpecHLSToArc.cpp

        DEPENDS

        MLIRSpecHLSOpsIncGen
        SpecHLSConversionIncGen
  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRSpecHLS
  MLIRTransforms
)

target_link_libraries(SpecHLSToArc

        PUBLIC
        CIRCTArc
        CIRCTComb
        CIRCTHW
        CIRCTSupport
        )