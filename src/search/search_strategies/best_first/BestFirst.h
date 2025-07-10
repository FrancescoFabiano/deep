/**
 * \class BestFirst
 * \brief Abstract base class for Best First Search strategies.
 *
 * This class provides the core search queue functionality for heuristic-driven
 * search. Derived classes must implement the \ref push and \ref get_name methods.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date July 10, 2025
 */

#pragma once
#include "states/State.h"
#include <queue>
#include <string>

#include "argparse/Configuration.h"
#include "heuristics/HeuristicsManager.h"

/**
 * \brief Compares two states based on their heuristic value.
 *
 * \tparam StateRepr The state representation type.
 * \param state1 The first state to compare.
 * \param state2 The second state to compare.
 * \return true if state1 has a higher heuristic value than state2, false otherwise.
 *
 * \note Lower scores are better. States with higher heuristic values have lower priority.
 */
template <StateRepresentation StateRepr>
struct StateComparator {
  bool operator()(const State<StateRepr> &state1,
                  const State<StateRepr> &state2) const {
    return state1.get_heuristic_value() > state2.get_heuristic_value();
  }
};

/**
 * \brief Abstract base class for Best First Search strategies.
 *
 * \tparam StateRepr The state representation type (must satisfy StateRepresentation).
 */
template <StateRepresentation StateRepr>
class BestFirst {
public:
  /**
   * \brief Constructor.
   *
   * \param initial_state The initial state used to initialize the heuristics manager.
   */
  explicit BestFirst(const State<StateRepr> &initial_state)
      : m_heuristics_manager(initial_state) {}

  /**
   * \brief Virtual destructor.
   */
  virtual ~BestFirst() = default;

  /**
   * \brief Pure virtual function to push a state into the search container.
   *
   * Must be implemented by derived classes to define filtering or priority behavior.
   *
   * \param s The state to push.
   */
  virtual void push(State<StateRepr> &s) = 0;

  /**
   * \brief Pop the state with the highest priority (lowest heuristic value).
   */
  void pop() { search_space.pop(); }

  /**
   * \brief Peek at the next state in the search container without removing it.
   *
   * \return The next state in the priority queue.
   */
  State<StateRepr> peek() const { return search_space.top(); }

  /**
   * \brief Pure virtual function to return the name of the search strategy.
   *
   * \return A descriptive name of the strategy and heuristic used.
   */
  [[nodiscard]] virtual std::string get_name() const = 0;

  /**
   * \brief Clear and reset the search container.
   */
  void reset() { search_space = StatePriorityQueue(); }

  /**
   * \brief Check whether the search container is empty.
   *
   * \return true if the container is empty, false otherwise.
   */
  [[nodiscard]] bool empty() const { return search_space.empty(); }

protected:
  /**
   * \brief Priority queue for managing the search space.
   *
   * States are ordered based on their heuristic value, with lower values having higher priority.
   */
  using StatePriorityQueue =
      std::priority_queue<State<StateRepr>, std::vector<State<StateRepr>>,
                          StateComparator<StateRepr>>;

  StatePriorityQueue search_space; ///< The search space represented as a priority queue of states.

  HeuristicsManager<StateRepr>
      m_heuristics_manager; ///< Heuristics manager to compute heuristic values for states.
};
