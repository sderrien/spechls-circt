#include "Transforms/Passes.h"
#include "circt/Dialect/SSP/SSPOps.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/StringExtras.h"
#include <circt/Dialect/SSP/SSPAttributes.h>
#include <iostream>

namespace SpecHLS {

namespace LocalMobility {

class Operator {
  llvm::StringRef name;
  unsigned latency;
  float incDelay;
  float outDelay;

public:
  Operator(llvm::StringRef name, int latency, float incDelay, float outDelay)
      : name(name), latency(latency), incDelay(incDelay), outDelay(outDelay) {}

  unsigned getLatency(void) { return latency; }

  float getIncDelay(void) { return incDelay; }

  float getOutDelay(void) { return outDelay; }

  void dump(void) {
    std::cerr << "operator @" << name.str() << "[latency=" << latency
              << "; incDelay=" << incDelay << "; outDelay=" << outDelay << "]"
              << std::endl;
  }
};

class Operation {
private:
  llvm::StringRef name;
  unsigned latency;
  float incDelay;
  float outDelay;
  llvm::DenseMap<Operation *, unsigned> dependences;
  Operation *controlPredecessor = nullptr;
  bool gamma;
  bool computeLocalMobility;
  unsigned mobility = 0;
  mlir::Operation *ptr = nullptr;
  llvm::SmallVector<Operation *> &gammas;
  llvm::DenseMap<Operation *, llvm::SmallVector<unsigned>> asapCycle;
  llvm::DenseMap<Operation *, llvm::SmallVector<float>> asapTimeInCycle;
  llvm::SmallVector<unsigned> alapCycle;
  llvm::SmallVector<float> alapTimeInCycle;

public:
  Operation(llvm::StringRef name, int latency, float incDelay, float outDelay,
            bool gamma, bool computeLocalMobility,
            llvm::SmallVector<Operation *> &gammas)
      : name(name), latency(latency), incDelay(incDelay), outDelay(outDelay),
        gamma(gamma), computeLocalMobility(computeLocalMobility),
        gammas(gammas) {
    for (auto *g : gammas) {
      asapCycle[g] = llvm::SmallVector<unsigned>();
      asapTimeInCycle[g] = llvm::SmallVector<float>();
    }
  }

  void resize(unsigned sumDistance) {
    for (auto &g : gammas) {
      asapCycle[g].resize(2 * sumDistance);
      asapTimeInCycle[g].resize(2 * sumDistance);
    }
    alapCycle.resize(2 * sumDistance);
    alapTimeInCycle.resize(2 * sumDistance);
  }

  void setControl(Operation *op) { controlPredecessor = op; }

  Operation *getControl(void) { return controlPredecessor; }

  llvm::SmallVector<Operation *> &getGammas(void) { return gammas; }

  void addDependences(Operation *op) { dependences[op] = 0; }

  void addDependences(Operation *op, unsigned distance) {
    dependences[op] = distance;
  }

  void setMlirOperation(mlir::Operation *operation) { ptr = operation; }

  mlir::Operation *getMlirOperation(void) { return ptr; }

  llvm::DenseMap<Operation *, unsigned> &getDependences(void) {
    return dependences;
  }

  void getInfos(unsigned &lat, float &incomingDelay, float &outgoingDelay) {
    lat = latency;
    incomingDelay = incDelay;
    outgoingDelay = outDelay;
  }

  bool isGamma(void) { return gamma; }

  bool needLocalMobility(void) { return computeLocalMobility; }

  void getAlap(unsigned iteration, unsigned &cycle, float &timeInCycle) {
    cycle = alapCycle[iteration];
    timeInCycle = alapTimeInCycle[iteration];
  }

  void getAsap(Operation *g, unsigned iteration, unsigned &cycle,
               float &timeInCycle) {
    cycle = asapCycle[g][iteration];
    timeInCycle = asapTimeInCycle[g][iteration];
  }

  void dump(void) {
    std::cerr << "operation @" << name.str() << "[lat=" << latency
              << "; incDelay=" << incDelay << "; outDelay=" << outDelay << "] "
              << "dependences={";
    for (auto op : dependences)
      llvm::errs() << op.first->name.str() << ":" << op.second << "; ";
    llvm::errs() << "}";
    if (gamma)
      llvm::errs() << "; mobility=" << mobility;
    llvm::errs() << "\n";
  }

  void setAsapCycle(Operation *g, unsigned iteration, unsigned cycle) {
    asapCycle[g][iteration] = cycle;
  }

  void setAsapTimeInCycle(Operation *g, unsigned iteration, float timeInCycle) {
    asapTimeInCycle[g][iteration] = timeInCycle;
  }

  void setAlapCycle(unsigned iteration, unsigned cycle) {
    alapCycle[iteration] = cycle;
  }

  void setAlapTimeInCycle(unsigned iteration, float timeInCycle) {
    alapTimeInCycle[iteration] = timeInCycle;
  }

  void computeMobility(unsigned sumDistance) {
    for (unsigned iteration = sumDistance; iteration < 2 * sumDistance;
         ++iteration) {
      unsigned diff1 = alapCycle[iteration] - alapCycle[iteration - 1];
      unsigned diff2 =
          asapCycle[this][iteration] - asapCycle[this][iteration - 1];
      unsigned localMobility = (diff1 < diff2) ? 0 : (diff1 - diff2);
      mobility = std::max(mobility, localMobility);
    }
    mlir::MLIRContext *context = ptr->getContext();
    mlir::Attribute mobilityAttr =
        mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32), mobility);
    ptr->setAttr("SpecHLS.mobility", mobilityAttr);
  }
};

void computeEnd(unsigned &cycle, float &timeInCycle, unsigned startCycle,
                float startTimeInCycle, unsigned latency, float outDelay) {
  if (latency == 0) {
    cycle = startCycle;
    timeInCycle = startTimeInCycle + outDelay;
  } else {
    cycle = startCycle + latency;
    timeInCycle = outDelay;
  }
}

void computeNextStart(unsigned &cycle, float &timeInCycle, float incDelay,
                      unsigned distance, float period) {
  if ((distance != 0) || (timeInCycle + incDelay >= period)) {
    ++cycle;
    timeInCycle = 0.0f;
  }
}

void compare(unsigned &aCycle, float &aTimeInCycle, unsigned bCycle,
             float bTimeInCycle, bool min) {
  if (min) {
    if (aCycle > bCycle) {
      aCycle = bCycle;
      aTimeInCycle = bTimeInCycle;
    } else if (aCycle == bCycle) {
      aTimeInCycle = std::min(aTimeInCycle, bTimeInCycle);
    }
  } else {
    if (aCycle < bCycle) {
      aCycle = bCycle;
      aTimeInCycle = bTimeInCycle;
    } else if (aCycle == bCycle) {
      aTimeInCycle = std::max(aTimeInCycle, bTimeInCycle);
    }
  }
}

void computeNext(Operation *op,
                 llvm::DenseMap<Operation *, unsigned> &nextAsapCycle,
                 llvm::DenseMap<Operation *, float> &nextAsapTimeInCycle,
                 unsigned &nextAlapCycle, float &nextAlapTimeInCycle,
                 Operation *pred, unsigned iteration, unsigned distance,
                 float period) {
  if (iteration < distance) {
    for (Operation *g : op->getGammas()) {
      nextAsapCycle[g] = 0;
      nextAsapTimeInCycle[g] = 0.0f;
    }
    compare(nextAlapCycle, nextAlapTimeInCycle, 0, 0.0f, false);
    return;
  }
  unsigned lat;
  float incDelay, outDelay;
  pred->getInfos(lat, incDelay, outDelay);

  // ALAP
  unsigned predStartCycle, nextStartCycle = 0;
  float predStartTimeInCycle, nextStartTimeInCycle = 0.0f;
  pred->getAlap(iteration - distance, predStartCycle, predStartTimeInCycle);
  computeEnd(nextStartCycle, nextStartTimeInCycle, predStartCycle,
             predStartTimeInCycle, lat, outDelay);
  computeNextStart(nextStartCycle, nextStartTimeInCycle, incDelay, distance,
                   period);
  compare(nextAlapCycle, nextAlapTimeInCycle, nextStartCycle,
          nextStartTimeInCycle, false);

  // ASAPs
  for (Operation *g : op->getGammas()) {
    bool computeAsap = g->isGamma() && g->needLocalMobility() && (op == g);
    if (pred != op->getControl()) {
      nextStartCycle = 0;
      nextStartTimeInCycle = 0.0f;
      pred->getAsap(g, iteration - distance, predStartCycle,
                    predStartTimeInCycle);
      computeEnd(nextStartCycle, nextStartTimeInCycle, predStartCycle,
                 predStartTimeInCycle, lat, outDelay);
      computeNextStart(nextStartCycle, nextStartTimeInCycle, incDelay, distance,
                       period);
      unsigned localNextAsapCycle = nextAsapCycle[g];
      float localNextAsapTimeInCycle = nextAsapTimeInCycle[g];
      compare(localNextAsapCycle, localNextAsapTimeInCycle, nextStartCycle,
              nextStartTimeInCycle, computeAsap);
      nextAsapCycle[g] = localNextAsapCycle;
      nextAsapTimeInCycle[g] = localNextAsapTimeInCycle;
    }
  }
}

}; // namespace LocalMobility

struct LocalMobilityPass
    : public impl::LocalMobilityPassBase<LocalMobilityPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();

    llvm::SmallVector<circt::ssp::InstanceOp> instanceOps;
    for (auto instOp : moduleOp.getOps<circt::ssp::InstanceOp>()) {
      float period;
      if (mlir::FloatAttr periodAttr =
              instOp->getAttrOfType<mlir::FloatAttr>("SpecHLS.period")) {
        period = periodAttr.getValue().convertToFloat();
      } else {
        llvm::errs() << "Error: ssp instance must have a 'period' attribute.\n";
        exit(1);
      }

      llvm::SmallVector<LocalMobility::Operator *> operators;
      llvm::SmallVector<LocalMobility::Operation *> operations;
      llvm::SmallVector<LocalMobility::Operation *> gammas;
      llvm::DenseMap<mlir::Operation *, LocalMobility::Operator *>
          translationOperator;
      llvm::DenseMap<mlir::Operation *, LocalMobility::Operation *>
          translationOperation;

      for (auto &op : instOp.getOperatorLibrary()) {
        mlir::StringAttr nameAttr =
            op.getAttrOfType<mlir::StringAttr>("sym_name");
        llvm::StringRef name = nameAttr.getValue();
        mlir::ArrayAttr array =
            op.getAttrOfType<mlir::ArrayAttr>("sspProperties");
        unsigned lat = 0;
        float incDelay = 0.0f, outDelay = 0.0f;
        for (auto &attr : array) {
          if (auto latencyAttr =
                  llvm::dyn_cast<circt::ssp::LatencyAttr>(attr)) {
            lat = latencyAttr.getValue();
          } else if (auto incomingDelay =
                         llvm::dyn_cast<circt::ssp::IncomingDelayAttr>(attr)) {
            incDelay = incomingDelay.getValue().getValue().convertToFloat();
          } else if (auto outcommingDelay =
                         llvm::dyn_cast<circt::ssp::OutgoingDelayAttr>(attr)) {
            outDelay = outcommingDelay.getValue().getValue().convertToFloat();
          }
        }
        LocalMobility::Operator *ptr =
            new LocalMobility::Operator(name, lat, incDelay, outDelay);
        operators.push_back(ptr);
        translationOperator[&op] = ptr;
      }

      for (auto &operation : instOp.getDependenceGraph()) {
        mlir::StringAttr nameAttr =
            operation.getAttrOfType<mlir::StringAttr>("sym_name");
        llvm::StringRef name = nameAttr.getValue();
        bool isGamma =
            operation.hasAttrOfType<mlir::ArrayAttr>("SpecHLS.gamma");
        bool computeLocalMobility = operation.hasAttr("SpecHLS.localMobility");
        auto properties =
            operation.getAttrOfType<mlir::ArrayAttr>("sspProperties");
        unsigned lat = 0;
        float incDelay = 0.0f, outDelay = 0.0f;
        for (auto prop : properties) {
          if (auto opr =
                  llvm::dyn_cast<circt::ssp::LinkedOperatorTypeAttr>(prop)) {
            auto *operatorType =
                instOp.getOperatorLibrary().lookupSymbol(opr.getValue());
            auto *opType = translationOperator.lookup(operatorType);
            lat = opType->getLatency();
            incDelay = opType->getIncDelay();
            outDelay = opType->getOutDelay();
          }
        }
        auto *ptr =
            new LocalMobility::Operation(name, lat, incDelay, outDelay, isGamma,
                                         computeLocalMobility, gammas);
        translationOperation[&operation] = ptr;
        operations.push_back(ptr);
        if (isGamma)
          gammas.push_back(ptr);
        ptr->setMlirOperation(&operation);
        for (auto &op : operation.getOpOperands()) {
          ptr->addDependences(
              translationOperation.lookup(op.get().getDefiningOp()));
        }
      }

      unsigned sumDistances = 0;
      for (auto *operation : operations) {
        auto *mlirOperation = operation->getMlirOperation();
        mlir::ArrayAttr controlNodeArray =
            mlirOperation->getAttrOfType<mlir::ArrayAttr>("SpecHLS.gamma");
        if (controlNodeArray) {
          for (mlir::Attribute attr : controlNodeArray) {
            if (auto control = llvm::dyn_cast<mlir::SymbolRefAttr>(attr)) {
              mlir::Operation *mlirControlOperation =
                  instOp.getDependenceGraph().lookupSymbol(control);
              LocalMobility::Operation *controlOperation =
                  translationOperation.lookup(mlirControlOperation);
              operation->setControl(controlOperation);
            }
          }
        }

        if (auto dependences =
                mlirOperation->getAttrOfType<mlir::ArrayAttr>("dependences")) {
          for (auto depAttr : dependences) {
            if (auto dep =
                    llvm::dyn_cast<circt::ssp::DependenceAttr>(depAttr)) {
              auto *op =
                  instOp.getDependenceGraph().lookupSymbol(dep.getSourceRef());
              mlir::ArrayAttr properties = dep.getProperties();
              for (auto prop : properties) {
                if (auto dist =
                        llvm::dyn_cast<circt::ssp::DistanceAttr>(prop)) {
                  sumDistances += dist.getValue();
                  operation->addDependences(translationOperation.lookup(op),
                                            dist.getValue());
                }
              }
            }
          }
        }
      }
      for (auto *op : operations) {
        op->resize(sumDistances);
      }

      for (auto *op : operators)
        delete op;

      for (unsigned iteration = 0; iteration < 2 * sumDistances; ++iteration)
        for (LocalMobility::Operation *op : operations) {
          // unsigned nextAsapCycle =
          //     op->isGamma() ? std::numeric_limits<unsigned>::max() : 0;
          llvm::DenseMap<LocalMobility::Operation *, unsigned> nextAsapCycle;
          // float nextAsapTimeInCycle = 0.0f;
          llvm::DenseMap<LocalMobility::Operation *, float> nextAsapTimeInCycle;
          for (auto *g : gammas) {
            nextAsapCycle[g] =
                op->isGamma() ? std::numeric_limits<unsigned>::max() : 0;
            nextAsapTimeInCycle[g] = 0.0f;
          }
          unsigned nextAlapCycle = 0;
          float nextAlapTimeInCycle = 0.0f;
          for (auto dep : op->getDependences()) {
            LocalMobility::Operation *pred = dep.first;
            unsigned dist = dep.second;
            LocalMobility::computeNext(op, nextAsapCycle, nextAsapTimeInCycle,
                                       nextAlapCycle, nextAlapTimeInCycle, pred,
                                       iteration, dist, period);
          }
          for (auto *g : gammas) {
            op->setAsapCycle(g, iteration, nextAsapCycle[g]);
            op->setAsapTimeInCycle(g, iteration, nextAsapTimeInCycle[g]);
          }
          op->setAlapCycle(iteration, nextAlapCycle);
          op->setAlapTimeInCycle(iteration, nextAlapTimeInCycle);
        }

      for (LocalMobility::Operation *g : gammas) {
        g->computeMobility(sumDistances);
      }

      for (LocalMobility::Operation *op : operations)
        delete op;
    }

    llvm::for_each(instanceOps, [](circt::ssp::InstanceOp op) { op.erase(); });
  }
};

[[maybe_unused]] std::unique_ptr<mlir::Pass> createLocalMobilityPass() {
  return std::make_unique<LocalMobilityPass>();
}

} // namespace SpecHLS