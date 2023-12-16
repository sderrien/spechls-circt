# Autogenerated from /Users/steven/Documents/gecos-gitlab/circt/mlir-standalone-template/test/lit.site.cfg.py.in
# Do not edit!

# Allow generated file to be relocatable.
import os
import platform
def path(p):
    if not p: return ''
    # Follows lit.util.abs_path_preserve_drive, which cannot be imported here.
    if platform.system() == 'Windows':
        return os.path.abspath(os.path.join(os.path.dirname(__file__), p))
    else:
        return os.path.realpath(os.path.join(os.path.dirname(__file__), p))


config.llvm_tools_dir = lit_config.substitute("/opt/circt/llvm/build/./bin")
config.lit_tools_dir = ""
config.mlir_obj_dir = "/Users/steven/Documents/gecos-gitlab/circt/mlir-standalone-template"
config.python_executable = ""
config.enable_bindings_python = 0
config.SpecHLS_src_root = "/Users/steven/Documents/gecos-gitlab/circt/mlir-standalone-template"
config.SpecHLS_obj_root = "/Users/steven/Documents/gecos-gitlab/circt/mlir-standalone-template"

import lit.llvm
lit.llvm.initialize(lit_config, config)

# Let the main config do the real work.
lit_config.load_config(config, "/Users/steven/Documents/gecos-gitlab/circt/mlir-standalone-template/test/lit.cfg.py")
