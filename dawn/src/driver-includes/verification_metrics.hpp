#pragma once

struct VerificationMetrics {
  int iteration;
  bool isValid;
  double maxRelErr;
  double minRelErr;
  double maxAbsErr;
  double minAbsErr;
};