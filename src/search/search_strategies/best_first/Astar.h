/**
 * \class Astar
 * \brief Implements the A* Search strategy to explore the search space.
 *
 * A* search augments the heuristic estimate with the cost to reach the current
 * state (in this case, the state's depth). This encourages exploration of
 * states that are both promising and closer to the initial state.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date July 10, 2025
 */

#pragma once
#include "BestFirst.h"

/**
 * \brief A* search strategy for use with SpaceSearcher.
 *
 * \tparam StateRepr The state representation type (must satisfy
 * StateRepresentation).
 */
template <StateRepresentation StateRepr>
class Astar final : public BestFirst<StateRepr> {
public:
  using Base = BestFirst<StateRepr>; ///< Alias for base class
  using Base::Base;                  ///< Inherit base constructor

  /**
   * \brief Push a state into the search container.
   *
   * This method computes the f-value as the sum of the heuristic value and
   * the state depth. States with negative combined values are skipped.
   *
   * \param s The state to be pushed into the priority queue.
   */
  void push(State<StateRepr> &s) override {
    const auto heuristics_value =
        this->m_heuristics_manager.get_heuristic_value(s);
    const auto plan_length = s.get_plan_length();

    // This is to exclude the initial state that might cause problems
    if (heuristics_value < 0 && plan_length != 0) {
      return; // Skip states with negative heuristic values.
    }
    s.set_heuristic_value(heuristics_value +
                          plan_length); // Overwrite heuristic with f = g + h
    this->search_space.push(s);
  }

  /**
   * \brief Get the name of the search strategy.
   *
   * \return A string containing the strategy name and the heuristic used.
   */
  [[nodiscard]] std::string get_name() const override {
    return "A* Search (" + this->m_heuristics_manager.get_used_h_name() + ")";
  }
};
