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

#include <sstream>
#include <string>
#include <utility>

namespace gridtools {
namespace dawn {

/**
 * @class Timer
 * Measures total elapsed time between all start and stop calls
 */
template <typename TimerImpl>
class timer {
protected:
  timer(std::string name) : m_name(std::move(name)) {}

public:
  /**
   * Reset counters
   */
  void reset() {
    m_total_time = 0;
    m_counter = 0;
  }

  /**
   * Start the stop watch
   */
  void start() { impl().start_impl(); }

  /**
   * Pause the stop watch
   */
  void pause() {
    m_total_time += impl().pause_impl();
    m_counter++;
  }

  /**
   * @return total elapsed time [s]
   */
  double total_time() const { return m_total_time; }

  /**
   * @return how often the timer was paused
   */
  size_t count() const { return m_counter; }

  /**
   * @return total elapsed time [s] as string
   */
  std::string to_string() const {
    std::ostringstream out;
    if(m_total_time < 0)
      out << "\t[s]\t" << m_name << "NO_TIMES_AVAILABLE";
    else
      out << m_name << "\t[s]\t" << m_total_time << " (" << m_counter << "x called)";
    return out.str();
  }

private:
  TimerImpl& impl() { return *static_cast<TimerImpl*>(this); }

  std::string m_name;
  double m_total_time = 0;
  size_t m_counter = 0;
};
} // namespace dawn
} // namespace gridtools
