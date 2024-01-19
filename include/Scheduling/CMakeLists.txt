add_subdirectory(Transforms)

set(LLVM_TARGET_DEFINITIONS SchedulingProperty.td)
mlir_tablegen(SchedulingProperty.h.inc -gen-attrdef-decls)
mlir_tablegen(SchedulingProperty.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(SpecHLSSchedulingAttributesIncGen)