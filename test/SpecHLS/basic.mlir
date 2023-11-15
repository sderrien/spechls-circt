hw.module @mycse(%a: i16, %b: i16, %sel:i2) -> (o: i16) {
   %c2 = hw.constant 1 : i1

  %c0 = comb.extract %sel from 0 : (i2)->i1
  %c1 = comb.extract %sel from 1 : (i2)->i1
  %0 = comb.mul %a, %b : i16
  %1 = comb.mul %a, %b : i16
  %2 = comb.mux %c2, %1, %0 : i16
  hw.output %2 : i16
}