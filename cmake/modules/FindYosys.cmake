if (UNIX)
    find_path(YOSYS_INCLUDE_DIR NAMES kernel/rtlil.h
              HINTS "/opt/yosys/share/include" "/usr/include"
              REQUIRED)
    find_path(YOSYS_LIBRARY_DIR NAMES libyosys.so
              HINTS "/opt/yosys"
              REQUIRED)
else ()
    message(ERROR "Not implemented yet")
endif ()
