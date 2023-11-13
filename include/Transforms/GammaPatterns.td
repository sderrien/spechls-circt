def : Pat<(AOp (BOp), $attr), ...>;


// fmadd: rs1 * rs2 + rs3
def : Pat<(GammaOp FPR64:$rs1, FPR64:$rs2, FPR64:$rs3),
          (FMADD_D $rs1, $rs2, $rs3, FRM_DYN)>;
