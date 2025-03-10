include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "circt/Dialect/HW/HW.td"
include "circt/Dialect/Comb/Comb.td"
include "Dialect/SpecHLS/SpecHLSOps.td"

include "mlir/Dialect/Arith/IR/ArithOps.td"

// Create zero attribute of type matching the argument's type.
def GetZeroAttr : NativeCodeCall<"$_builder.getZeroAttr($0.getType())">;

// Add two integer attributes and create a new one with the result.
def AddIntAttrs : NativeCodeCall<"addIntegerAttrs($_builder, $0, $1, $2)">;

def narrowCompare : NativeCodeCall<"narrowCompare($_builder, $0.getDefiningOp(), $1, $2,$3)">;
def expandAndMergeLUT : NativeCodeCall<"expandAndMergeLUT($_builder, $0.getDefiningOp(), $1, $2,$3, $4)">;

def ICmpConstant :
    Pat<(ICmpOp:$res $p, $x, (ConstantLikeMatcher APIntAttr:$c), $s ), (narrowCompare $p, $x, $c)>;

def ICmpConstantWithCastRight :
    Pat<(ICmpOp:$res $p, (SpecHLS_CastOp $x), (ConstantLikeMatcher APIntAttr:$c), $s ), (narrowCompare $p, $x, $c,$s)>;

def ICmpConstantWithCastLeft :
    Pat<(ICmpOp:$res $p, (ConstantLikeMatcher APIntAttr:$c),(SpecHLS_CastOp $x), $s ), (narrowCompare $p, $x, $c,$s)>;

def LUTSimplifyPattern :
    Pat<(SpecHLS_LookUpTableOp:$res (ConcatOp (variadic $x, (SpecHLS_LookUpTableOp $y ,$inner_lut)) ), $outer_lut), (expandAndMergeLUT $res, $x, $y,$outer_lut, $inner_lut)>;

