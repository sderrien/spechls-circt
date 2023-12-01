set(LLVM_OPTIONAL_SOURCES
  RecMIIAnalysis.cpp
#  DependenceAnalysis.cpp
#  SchedulingAnalysis.cpp
#  TestPasses.cpp
  )

add_circt_library(SPECHLSRecMIIAnalysis
  RecMIIAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
#  MLIRAffineUtils
  MLIRTransformUtils
  )

#add_circt_library(CIRCTSchedulingAnalysis
#  SchedulingAnalysis.cpp
#
#  LINK_LIBS PUBLIC
#  MLIRAffineDialect
#  MLIRIR
#  CIRCTDependenceAnalysis
#  CIRCTScheduling
#  )
#
#add_circt_library(CIRCTRecMIIAnalysis
#        SchedulingAnalysis.cpp
#
#        LINK_LIBS PUBLIC
#        MLIRAffineDialect
#        MLIRIR
#        CIRCTDependenceAnalysis
#        CIRCTScheduling
#        )
#
#add_circt_library(CIRCTAnalysisTestPasses
#  TestPasses.cpp
#
#  LINK_LIBS PUBLIC
#  CIRCTDependenceAnalysis
#  CIRCTSchedulingAnalysis
#  CIRCTHW
#  MLIRPass
#  )
#
#add_circt_library(CIRCTDebugInfoAnalysis
#  DebugInfo.cpp
#
#  LINK_LIBS PUBLIC
#  CIRCTHW
#  MLIRIR
#)
