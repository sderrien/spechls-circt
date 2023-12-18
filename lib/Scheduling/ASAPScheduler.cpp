#include "Scheduling/Algorithms.h"
#include "circt/Dialect/SSP/Utilities.h"
#include "circt/Scheduling/Utilities.h"
#include <iostream>

using namespace mlir;
using StartTime = std::pair<unsigned, float>;

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
  if ((iteration < maxDep) || (iteration < 3))
    return false;
  for (auto *op : prob.getOperations()) {
    auto t1 = scheduled[iteration - 3][op];
    auto t2 = scheduled[iteration - 2][op];
    auto t3 = scheduled[iteration - 1][op];
    if (t2.second != t3.second)
      return false;
    if ((t3.first - t2.first) != (t2.first - t1.first))
      return false;
  }
  return true;
}

LogicalResult scheduleASAP(GammaMobilityProblem &prob, float cycleTime) {
  std::vector<std::unordered_map<Operation *, StartTime>> scheduled;
  unsigned maxDep = 0;
  for (auto *op : prob.getOperations())
    for (auto &dep : prob.getDependences(op)) {
      if (dep.isAuxiliary())
        maxDep = std::max(maxDep, *prob.getDistance(dep));
    }
  unsigned iteration = 0;
  do {
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
                potentialStartTime = getNextStart(
                    prob, getEndTime(prob, predStartTime, pred), op, cycleTime);
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
    ++iteration;
  } while (!fixpointReached(prob, scheduled, iteration, maxDep));
  auto &last = scheduled.back();
  int minCycle = -1;
  for (auto *op : prob.getOperations()) {
    if (minCycle < 0)
      minCycle = last[op].first;
    else
      minCycle = std::min((unsigned)minCycle, last[op].first);
  }
  for (auto *op : prob.getOperations()) {
    prob.setStartTime(op, last[op].first - minCycle);
    prob.setStartTimeInCycle(op, last[op].second);
  }
  auto *op = prob.getOperations().front();
  prob.setInitiationInterval(scheduled[iteration - 1][op].first -
                             scheduled[iteration - 2][op].first);
  return success();
}