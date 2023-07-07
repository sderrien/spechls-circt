configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SpecHLS_OPT_TEST_DEPENDS
        FileCheck count not
        spechls-opt
        )

add_lit_testsuite(check-spechls-opt "Running the spechls-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${SpecHLS_OPT_TEST_DEPENDS}
        )
set_target_properties(check-spechls-opt PROPERTIES FOLDER "Tests")

add_lit_testsuites(SpecHLS_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${SpecHLS_OPT_TEST_DEPENDS})
