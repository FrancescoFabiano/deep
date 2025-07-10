/**
 * \class HeuristicFirst
 * \brief Implements the Best First Search strategy to explore the search space.
 *
 * This class extends BestFirstBase by filtering out states with negative
 * heuristic values, thereby ensuring that only promising states are added to
 * the search queue.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 29, 2025
 */

#pragma once
#include "BestFirst.h"

/**
 * \brief BestFirst search strategy for use with SpaceSearcher.
 *
 * \tparam StateRepr The state representation type (must satisfy
 * StateRepresentation).
 */
template <StateRepresentation StateRepr>
class HeuristicFirst final : public BestFirst<StateRepr> {
public:
  using Base = BestFirst<StateRepr>; ///< Alias for base class
  using Base::Base;                  ///< Inherit base constructor

  /**
   * \brief Push a state into the search container.
   *
   * This implementation filters out states with negative heuristic values,
   * which are considered invalid or unpromising.
   *
   * \param s The state to be pushed into the priority queue.
   */
  void push(State<StateRepr> &s) override {
    const auto heuristics_value =
        this->m_heuristics_manager.get_heuristic_value(s);
    // This is to exclude the initial state that might cause problems
    if (heuristics_value < 0 && s.get_plan_length() != 0) {
        return; // Skip states with negative heuristic values.
    }
    s.set_heuristic_value(heuristics_value); // Set the heuristic value
    this->search_space.push(s);
  }

  /**
   * \brief Get the name of the search strategy.
   *
   * \return A string containing the strategy name and the heuristic used.
   */
  [[nodiscard]] std::string get_name() const override {
    return "Heuristics First Search (" +
           this->m_heuristics_manager.get_used_h_name() + ")";
  }
};
