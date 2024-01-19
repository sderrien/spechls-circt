#IR Example using the CF Dialect

# Import the Standard and ControlFlow dialects
# The control flow operations are defined in the CF dialect
module {
  func @simple_loop(%arg0: index) -> () {
    %entry = ^bb(%arg0: index):
      br ^bb1(%arg0: index)

    %loop_header = ^bb1(%arg1: index):
      %cmp = cmpi "slt", %arg1, 10 : index
      cond_br %cmp, ^bb2(%arg1: index), ^bb3


    %loop_body = ^bb2(%arg2: index):
      %inc = addi %arg2, 1 : index
        switch %inc : i32, [
          default: ^bb1(%inc : i32),
          42: ^exit(%inc : i32),
          43: ^loop_header(%inc : i32)
        ]
      br ^bb1(%inc : index)

    %exit = ^bb3:
        return
  }


  func @simple_loop(%arg0: index) -> () {
    %entry = ^bb(%arg0: index):
      br ^bb1(%arg0: index)

    %loop_header = ^bb1(%arg1: index):
      %cmp = cmpi "slt", %arg1, 10 : index
      cond_br %cmp, ^bb2(%arg1: index), ^bb3

    %loop_body = ^bb2(%arg2: index):
      %inc = addi %arg2, 1 : index
      br ^bb1(%inc : index)

    %exit = ^bb3:
        return
  }
}
