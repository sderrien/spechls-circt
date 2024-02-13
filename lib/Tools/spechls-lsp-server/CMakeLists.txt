function(llvm_update_compile_flags_custom name)
    get_property(sources TARGET ${name} PROPERTY SOURCES)
    if ("${sources}" MATCHES "\\.c(;|$)")
        set(update_src_props ON)
    endif ()

    list(APPEND LLVM_COMPILE_CFLAGS " ${LLVM_COMPILE_FLAGS}")

    # LLVM_REQUIRES_EH is an internal flag that individual targets can use to
    # force EH
    if (LLVM_REQUIRES_EH OR LLVM_ENABLE_EH)
        if (NOT (LLVM_REQUIRES_RTTI OR LLVM_ENABLE_RTTI))
            message(AUTHOR_WARNING "Exception handling requires RTTI. Enabling RTTI for ${name}")
            set(LLVM_REQUIRES_RTTI ON)
        endif ()
        if (MSVC)
            list(APPEND LLVM_COMPILE_FLAGS "/EHsc")
        endif ()
    else ()
        if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
            if (LLVM_ENABLE_UNWIND_TABLES)
                list(APPEND LLVM_COMPILE_FLAGS "-funwind-tables")
            else ()
                list(APPEND LLVM_COMPILE_FLAGS "-fno-unwind-tables")
                list(APPEND LLVM_COMPILE_FLAGS "-fno-asynchronous-unwind-tables")
            endif ()
        elseif (MSVC)
            list(APPEND LLVM_COMPILE_DEFINITIONS _HAS_EXCEPTIONS=0)
            list(APPEND LLVM_COMPILE_FLAGS "/EHs-c-")
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "XL")
            list(APPEND LLVM_COMPILE_FLAGS "-qnoeh")
        endif ()
    endif ()

    # LLVM_REQUIRES_RTTI is an internal flag that individual
    # targets can use to force RTTI
    set(LLVM_CONFIG_HAS_RTTI YES CACHE INTERNAL "")
    if (NOT (LLVM_REQUIRES_RTTI OR LLVM_ENABLE_RTTI))
        set(LLVM_CONFIG_HAS_RTTI NO CACHE INTERNAL "")
        list(APPEND LLVM_COMPILE_DEFINITIONS GTEST_HAS_RTTI=0)
        if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
            list(APPEND LLVM_COMPILE_FLAGS "-fno-rtti")
        elseif (MSVC)
            list(APPEND LLVM_COMPILE_FLAGS "/GR-")
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "XL")
            list(APPEND LLVM_COMPILE_FLAGS "-qnortti")
        endif ()
    elseif (MSVC)
        list(APPEND LLVM_COMPILE_FLAGS "/GR")
    endif ()

    # Assume that;
    #   - LLVM_COMPILE_FLAGS is list.
    #   - PROPERTY COMPILE_FLAGS is string.
    string(REPLACE ";" " " target_compile_flags " ${LLVM_COMPILE_FLAGS}")
    string(REPLACE ";" " " target_compile_cflags " ${LLVM_COMPILE_CFLAGS}")

    if (update_src_props)
        foreach (fn ${sources})
            get_filename_component(suf ${fn} EXT)
            if ("${suf}" STREQUAL ".cpp")
                set_property(SOURCE ${fn} APPEND_STRING PROPERTY
                        COMPILE_FLAGS "${target_compile_flags}")
            endif ()
            if ("${suf}" STREQUAL ".c")
                set_property(SOURCE ${fn} APPEND_STRING PROPERTY
                        COMPILE_FLAGS "${target_compile_cflags}")
            endif ()
        endforeach ()
    else ()
        # Update target props, since all sources are C++.
        set_property(TARGET ${name} APPEND_STRING PROPERTY
                COMPILE_FLAGS "${target_compile_flags}")
    endif ()

    set_property(TARGET ${name} APPEND PROPERTY COMPILE_DEFINITIONS ${LLVM_COMPILE_DEFINITIONS})
endfunction()


get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(circt_dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)

get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        CIRCTExportVerilog
        MLIROptLib
        MLIRSpecHLS
        MLIRAffineAnalysis
        MLIRAnalysis
        MLIRDialect
        MLIRFuncAllExtensions
        MLIRLspServerLib
        MLIRParser
        MLIRPass
        MLIRTransforms
        MLIRTransformUtils
        MLIRSupport
        MLIRIR
        SpecHLSTransforms
        CIRCTSeq
        CIRCTComb
        CIRCTHW
        CIRCTDC
        CIRCTFIRRTL
        CIRCTHWArith
        CIRCTComb
        CIRCTSeq
        CIRCTSSP
        CIRCTCombTransforms
        CIRCTFIRRTL
        CIRCTFSM
        CIRCTHW
        CIRCTHWArith
        CIRCTScheduling
        CIRCTHWArith
        CIRCTSystemC
        CIRCTTransforms
        CIRCTSupport
        CIRCTArc
        CIRCTCalyx
        CIRCTHandshake
        CIRCTInteropDialect
        CIRCTIbis
        CIRCTLLHD
        CIRCTLoopSchedule
        CIRCTMoore
        MLIRLspServerLib
        ${mlir_dialect_libs}
        ${circt_dialect_libs}

)
add_llvm_executable(spechls-lsp-server SpecHLSLspServerMain.cpp)


llvm_update_compile_flags(spechls-lsp-server)

target_link_libraries(spechls-lsp-server PRIVATE ${LIBS})
target_compile_options(spechls-lsp-server PUBLIC -fexceptions)
install(TARGETS spechls-lsp-server DESTINATION bin)