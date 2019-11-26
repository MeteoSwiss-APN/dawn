//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//
#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>

#include "timer.hpp"

namespace gridtools {
namespace dawn {

/**
 * @class timer_cuda
 * CUDA implementation of the Timer interface
 */
class timer_cuda : public timer<timer_cuda> // CRTP
{
  struct event_deleter {
    void operator()(cudaEvent_t event) const { cudaEventDestroy(event); }
  };
  using event_holder = std::unique_ptr<CUevent_st, event_deleter>;

  static event_holder create_event() {
    cudaEvent_t event;
    cudaEventCreate(&event);
    return event_holder{event};
  }

  event_holder m_start = create_event();
  event_holder m_stop = create_event();

public:
  timer_cuda(std::string name) : timer<timer_cuda>(name) {}

  /**
   * Reset counters
   */
  void set_impl(double) {}

  /**
   * Start the stop watch
   */
  void start_impl() {
    // insert a start event
    cudaEventRecord(m_start.get(), 0);
  }

  /**
   * Pause the stop watch
   */
  double pause_impl() {
    // insert stop event and wait for it
    cudaEventRecord(m_stop.get(), 0);
    cudaEventSynchronize(m_stop.get());

    // compute the timing
    float result;
    cudaEventElapsedTime(&result, m_start.get(), m_stop.get());
    return result * 0.001; // convert ms to s
  }
};
} // namespace dawn
} // namespace gridtools
