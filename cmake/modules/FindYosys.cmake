if (UNIX)
    find_path(YOSYS_INCLUDE_DIR NAMES kernel/rtlil.h
              HINTS "/usr/share/yosys/include" "/usr/include"
              REQUIRED)
    find_path(YOSYS_LIBRARY_DIR NAMES libyosys.so
              HINTS "/usr/lib/yosys"
              REQUIRED)
else ()
    message(ERROR "Not implemented yet")
endif ()
