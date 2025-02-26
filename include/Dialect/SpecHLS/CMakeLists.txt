add_mlir_dialect(SpecHLSOps SpecHLS)
add_mlir_doc(SpecHLSDialect SpecHLSDialect SpecHLS/ -gen-dialect-doc)
add_mlir_doc(SpecHLSOps SpecHLSOps SpecHLS/ -gen-op-doc)

add_mlir_doc(SpecHLSTypes SpecHLSTypes SpecHLS/ -gen-type-doc)
set(LLVM_TARGET_DEFINITIONS SpecHLSPatterns.td)
mlir_tablegen(SpecHLSPatterns.h.inc  -gen-rewriters)
add_public_tablegen_target(SpecHLSPatternsIncGen)

