/**
 * \class DepthFirst
 * \brief Implements the Depth First Search strategy to explore the search
 * space.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 29, 2025
 */

#pragma once
#include "states/State.h"
#include <stack>
#include <string>

/**
 * \brief DepthFirst search strategy for use with SpaceSearcher.
 * \tparam StateRepr The state representation type (must satisfy
 * StateRepresentation).
 */
template <StateRepresentation StateRepr> class DepthFirst {
public:
  /**
   * \brief Default constructor.
   */
  explicit DepthFirst([[maybe_unused]] const State<StateRepr> &initial_state) {}

  /**
   * \brief Push a state into the search container.
   */
  void push(const State<StateRepr> &s) { search_space.push(s); }

  /**
   * \brief Pop a state from the search container.
   */
  void pop() { search_space.pop(); }

  /**
   * \brief Peek at the next state in the search container.
   */
  State<StateRepr> peek() const { return search_space.top(); }

  /**
   * \brief Get the name of the search strategy.
   */
  [[nodiscard]] std::string get_name() const { return m_name; }

  /**
   * \brief Reset the search container.
   */
  void reset() { search_space = std::stack<State<StateRepr>>(); }

  /**
   * \brief Check if the search container is empty.
   */
  [[nodiscard]] bool empty() const { return search_space.empty(); }

private:
  std::stack<State<StateRepr>> search_space;
  std::string m_name = "Depth First Search"; ///< Name of the search strategy.
};