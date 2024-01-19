#ifndef SCHEDULING_PROPERTY_TD
#define SCHEDULING_PROPERTY_TD


include "circt/Dialect/SSP/PropertyBase.td"
include "SpecHLS/SpecHLSDialect.td"


def MinPositionProp : OperationProperty<SpecHLS_Dialect,
  "MinPosition", "unsigned", "::SpecHLS::GammaMobilityProblem"> {
  let mnemonic = "minPosition";
}

#endif // SCHEDULING_PROPERTY_TD