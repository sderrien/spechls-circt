#include "Scheduling/Algorithms.h"
#include "circt/Dialect/SSP/Utilities.h"
#include "circt/Scheduling/Utilities.h"
#include "mlir/IR/Builders.h"
#include <iostream>
#include <map>

using namespace mlir;
using namespace SpecHLS;

static constexpr bool debug = false;

struct StartTime {
public:
  unsigned first;
  float second;

  StartTime() : first(0), second(0.0) {}

  template <typename T1, typename T2>
  StartTime(std::pair<T1, T2> pair) : first(pair.first), second(pair.second) {}

  template <typename T1, typename T2>
  StartTime(std::pair<T1, T2> &pair) : first(pair.first), second(pair.second) {}

  friend std::ostream &operator<<(std::ostream &os,
                                  const StartTime &startTime) {
    os << "(" << startTime.first << ", " << startTime.second << ")";
    return os;
  }
};

void put(std::map<Operation *, unsigned> &map, Operation *key, unsigned value) {
  auto oldValue = map[key];
  map[key] = (oldValue > value) ? oldValue : value;
}

unsigned getSumDistance(GammaMobilityProblem &prob) {
  unsigned distance = 0;
  for (auto *op : prob.getOperations())
    for (auto &dep : prob.getDependences(op))
      if (dep.isAuxiliary())
        distance += *prob.getDistance(dep);
  return distance;
}

StartTime getEndTime(GammaMobilityProblem &prob, StartTime startTime,
                     Operation *op) {
  auto opr = *prob.getLinkedOperatorType(op);
  if (*prob.getLatency(opr) == 0) {
    return std::make_pair(startTime.first,
                          startTime.second + *prob.getOutgoingDelay(opr));
  }
  return std::make_pair(startTime.first + *prob.getLatency(opr),
                        *prob.getOutgoingDelay(opr));
}

StartTime min(StartTime a, StartTime b) {
  if (a.first < b.first)
    return a;
  if (a.first > b.first)
    return b;
  if (a.second < b.second)
    return a;
  return b;
}

StartTime max(StartTime a, StartTime b) {
  if (a.first < b.first)
    return b;
  if (a.first > b.first)
    return a;
  if (a.second < b.second)
    return b;
  return a;
}

StartTime getCompared(StartTime first, StartTime second, bool isGamma) {
  if (isGamma) {
    return min(first, second);
  }
  return max(first, second);
}

StartTime getNextStart(GammaMobilityProblem &prob, StartTime time,
                       Operation *op, float cycleTime) {
  auto opr = *prob.getLinkedOperatorType(op);
  if (time.second + *prob.getIncomingDelay(opr) < cycleTime)
    return time;
  return std::make_pair(time.first + 1, 0.0);
}

bool fixpointReached(
    GammaMobilityProblem &prob,
    std::vector<std::unordered_map<Operation *, StartTime>> &scheduled,
    unsigned iteration, unsigned maxDep) {
  if (debug) {
    std::cerr << "computing fixpoint for iteration " << iteration << "."
              << std::endl;
    for (auto *op : prob.getOperations()) {
      op->dump();
      std::cerr << scheduled[iteration - 1][op] << std::endl;
    }
  }
  if ((iteration < maxDep) || (iteration < 3))
    return false;
  for (auto *op : prob.getOperations()) {
    auto t1 = scheduled[iteration - 3][op];
    auto t2 = scheduled[iteration - 2][op];
    auto t3 = scheduled[iteration - 1][op];
    if (t2.second != t3.second) {
      if (debug)
        std::cerr << "iteration " << iteration << ": fixpoint not reached."
                  << std::endl;
      return false;
    }
    if ((t3.first - t2.first) != (t2.first - t1.first)) {
      if (debug)
        std::cerr << "iteration " << iteration << ": fixpoint not reached."
                  << std::endl;
      return false;
    }
  }
  if (debug)
    std::cerr << "iteration " << iteration << ": fixpoint reached."
              << std::endl;
  return true;
}

StartTime ceil(StartTime t) {
  if (t.second == 0.0)
    return t;
  return std::make_pair(t.first + 1, 0.0);
}

LogicalResult SpecHLS::scheduleASAP(GammaMobilityProblem &prob,
                                    float cycleTime) {
  std::vector<std::unordered_map<Operation *, StartTime>> scheduled;
  unsigned maxDep = 0;
  for (auto *op : prob.getOperations())
    for (auto &dep : prob.getDependences(op)) {
      if (dep.isAuxiliary())
        maxDep = std::max(maxDep, *prob.getDistance(dep));
    }
  unsigned iteration;
  unsigned sumDistance = getSumDistance(prob);
  for (iteration = 0; iteration < 2 * sumDistance; ++iteration) {
    if (failed(handleOperationsInTopologicalOrder(prob, [&](Operation *op) {
          if (prob.getDependences(op).empty()) {
            prob.setStartTimeInCycle(op, 0.0);
            prob.setStartTime(op, 0);
            return success();
          }

          StartTime startTime = std::make_pair(0, 0.0);

          bool isGamma = op->hasAttr("SpecHLS.gamma");
          bool first = true;
          for (auto &dep : prob.getDependences(op)) {
            auto *pred = dep.getSource();
            StartTime potentialStartTime;
            if (dep.isAuxiliary()) {
              if (iteration < prob.getDistance(dep)) {
                potentialStartTime = std::make_pair(0, 0.0);

              } else {
                auto predStartTime =
                    scheduled[iteration - *prob.getDistance(dep)][pred];
                auto endTime = getEndTime(prob, predStartTime, pred);
                auto nextStart = getNextStart(prob, endTime, op, cycleTime);
                potentialStartTime = ceil(nextStart);
                potentialStartTime =
                    max(potentialStartTime,
                        std::make_pair(scheduled[iteration - 1][pred].first + 1,
                                       0.0));
              }
            } else {
              if (!prob.getStartTime(pred) || !prob.getStartTimeInCycle(pred))
                return failure();
              potentialStartTime = getNextStart(
                  prob,
                  getEndTime(prob,
                             std::make_pair(*prob.getStartTime(pred),
                                            *prob.getStartTimeInCycle(pred)),
                             pred),
                  op, cycleTime);
            }
            if (first) {
              first = false;
              startTime = potentialStartTime;
            } else
              startTime = getCompared(startTime, potentialStartTime, isGamma);
          }
          prob.setStartTime(op, startTime.first);
          prob.setStartTimeInCycle(op, startTime.second);
          return success();
        })))
      return failure();
    std::unordered_map<Operation *, StartTime> current;
    for (auto *op : prob.getOperations()) {
      current.emplace(op, std::make_pair(*prob.getStartTime(op),
                                         *prob.getStartTimeInCycle(op)));
    }
    scheduled.push_back(current);
  }
  if (debug) {
    for (unsigned iteration = 0; iteration < 2 * sumDistance; ++iteration) {
      std::cout << "iteration " << iteration << std::endl;
      for (auto *op : prob.getOperations()) {
        op->dump();
        std::cout << scheduled[iteration][op].first << std::endl;
      }
      std::cout << std::endl;
    }
  }
  unsigned delta[sumDistance];
  for (unsigned iteration = sumDistance; iteration < 2 * sumDistance;
       ++iteration) {
    bool first = true;
    for (auto *op : prob.getOperations()) {
      if (first) {
        first = false;
        delta[iteration - sumDistance] = scheduled[iteration][op].first;
      } else {
        delta[iteration - sumDistance] = std::min(
            scheduled[iteration][op].first, delta[iteration - sumDistance]);
      }
    }
  }
  for (auto *op : prob.getOperations())
    if (op->hasAttr("SpecHLS.gamma")) {
      unsigned minPosition = UINT_MAX;
      for (unsigned iteration = sumDistance; iteration < 2 * sumDistance;
           ++iteration)
        minPosition = std::min(minPosition, scheduled[iteration][op].first -
                                                delta[iteration - sumDistance]);
      prob.setMinPosition(op, minPosition);
      op->dump();
      if (debug) {
        op->dump();
        std::cerr << "min position: " << minPosition << std::endl;
      }
    }
  return success();
}