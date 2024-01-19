//===- InitAllPasses.h - CIRCT Global Pass Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all passes to the
// system.
//
//===----------------------------------------------------------------------===//

#ifndef SPECHLS_INITALLPASSES_H_
#define SPECHLS_INITALLPASSES_H_

#include "SpecHLS/SpecHLSDialect.h"
#include "Conversion/Passes.h"
#include "Scheduling/Transforms/SchedulePass.h"
#include "Transforms/Passes.h"
#include "circt/Transforms/Passes.h"

namespace SpecHLS {

inline void registerAllPasses() {

  static bool initOnce = []() {
    registerSpecHLSToCombPass();
    registerMergeGammasPass();
    registerMergeLookUpTablesPass();
    registerFactorGammaInputsPass();
    registerGroupControlNodePass();
    registerInlineModules();
    registerYosysOptimizerPass();
    registerGecosSchedulePass();
    return true;
  }();
  (void)initOnce;
}

} // namespace SpecHLS

#endif // SPECHLS_INITALLPASSES_H_
