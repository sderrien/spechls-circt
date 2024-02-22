# An out-of-tree dialect for MLIR targeted at Speculative HLS

This repository contains a template for an out-of-tree [MLIR](https://mlir.llvm.org/) dialect as well as a
SpecHLS `opt`-like tool to operate on that dialect.

## How to build

### Step 1 : MLIR

```sh
cd circt
$ mkdir llvm/build
$ cd llvm/build
$ cmake -G Ninja ../llvm \
    -DCMAKE_INSTALL_PREFIX=/opt/circt-prefix/ \rld
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_BUILD_UTILS:BOOL=ON \
    -DLLVM_INCLUDE_UTILS:BOOL=ON \
    -DLLVM_INSTALL_UTILS:BOOL=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_EH=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="X86;RISCV" 
    
$ ninja
$ ninja check-mlir
```
cmake -G Ninja ../llvm -DCMAKE_INSTALL_PREFIX=/opt/circt-prefix/ -DCMAKE_BUILD_TYPE=Debug -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_ASSERTIONS=ON \
### Step 2 : CIRCT

```sh
$ cd circt
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
     -DCMAKE_INSTALL_PREFIX=/opt/circt-prefix/ \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_BUILD_UTILS:BOOL=ON \
    -DLLVM_INCLUDE_UTILS:BOOL=ON \
    -DLLVM_INSTALL_UTILS:BOOL=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_EH=ON \
    -DCMAKE_BUILD_TYPE=DEBUG
$ ninja
$ ninja check-circt
$ ninja check-circt-integration # Run the integration tests.
```
### Setp 3 : Install yosys

```
git clone https://github.com/YosysHQ/yosys.git
cd yosys
```

Edit the following lines from ```Makefile```  

```sh
ENABLE_TCL := 0
ENABLE_ABC := 1
ENABLE_GLOB := 0
ENABLE_PLUGINS := 0
ENABLE_READLINE := 0
ENABLE_EDITLINE := 0
ENABLE_GHDL := 0
ENABLE_VERIFIC := 0
ENABLE_VERIFIC_EDIF := 0
ENABLE_VERIFIC_LIBERTY := 0
DISABLE_VERIFIC_EXTENSIONS := 0
DISABLE_VERIFIC_VHDL := 0
ENABLE_COVER := 0
ENABLE_LIBYOSYS := 1
ENABLE_ZLIB := 0
```

```sh
$ cd circt
$ mkdir build
$ cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_EH=ON \
    -DCMAKE_BUILD_TYPE=DEBUG
$ ninja
$ ninja check-circt
$ ninja check-circt-integration # Run the integration tests.
```



This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. 

To build and launch the tests, run
```sh


mkdir build && cd build
cmake -G Ninja .. \
-DYOSYS_LIBRARY_DIRS=/usr/local/lib/yosys/ \ 
-DYOSYS_INCLUDE_DIRS=/opt/yosys/share/include/  \
-DMLIR_DIR=$PREFIX/lib/cmake/mlir  \
-DCIRCT_DIR=$PREFIX/lib/cmake/mlir  \
-DLLVM_EXTERNAL_LIT=/opt/circt/llvm/build/bin/llvm-lit  \
-DUSE_ALTERNATE_LINKER=/usr/local/opt/llvm/bin/ld64.lld  \
-DLLVM_DIR:PATH=$PREFIX \

cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-spechls-opt
```
To build the documentation from the TableGen description of the dialect
operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with
CMake so that it installs `FileCheck` to the chosen installation prefix.

## License

This dialect template is made available under the Apache License 2.0 with LLVM Exceptions. See the `LICENSE.txt` file for more details.
