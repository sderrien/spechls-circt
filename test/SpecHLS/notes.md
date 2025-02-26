mlir-opt -convert-vector-to-spirv $1 | grep "failed to materialize"
if [[ $? -eq 1 ]]; then
exit 1
else
exit 0
fi

The sample usage will be like, note that the test argument is part of the mode argument.

mlir-reduce $INPUT -reduction-tree='traversal-mode=0 test=$TEST_SCRIPT'
