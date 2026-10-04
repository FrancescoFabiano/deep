/**
 * \class BreadthFirst
 * \brief Implements the Breadth First Search strategy to explore the search
 * space.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 29, 2025
 */

#pragma once
#include "states/State.h"
#include <queue>
#include <string>

/**
 * \brief BreadthFirst search strategy for use with SpaceSearcher.
 * \tparam StateRepr The state representation type (must satisfy
 * StateRepresentation).
 */
template <StateRepresentation StateRepr> class BreadthFirst {
public:
  /**
   * \brief Default constructor.
   */
  explicit BreadthFirst(
      [[maybe_unused]] const State<StateRepr> &initial_state) {}
  /**
   * \brief Push a state into the search container.
   */
  void push(State<StateRepr> &s) {
    search_space.push(std::move(s)); // takes the successor
  }

  /**
   * \brief Push the initial state into the search container.
   */
  void push_initial(const State<StateRepr> &s) { search_space.push(s); }

  /**
   * \brief Push a list of states into the search container. Not implemented for
   * searches that are not RL-based
   */
  void push_vector([[maybe_unused]] std::vector<State<StateRepr>> &s) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::SearchMethodNotImplemented,
        "Error: push of a vector of states is not implemented for BFS. It is "
        "solely added for RL reasoning");
  }

  /**
   * \brief Pop a state from the search container.
   */
  void pop() { search_space.pop(); }

  /**
   * \brief Peek at the next state in the search container.
   */
  State<StateRepr> peek() const { return search_space.front(); }

  /**
   * \brief Remove the next state from the container and return it (moved
   * out, not copied).
   */
  State<StateRepr> take() {
    State<StateRepr> next = std::move(search_space.front());
    search_space.pop();
    return next;
  }

  /**
   * \brief Get the name of the search strategy.
   */
  [[nodiscard]] std::string get_name() const { return m_name; }

  /**
   * \brief Reset the search container.
   */
  void reset() { search_space = std::queue<State<StateRepr>>(); }

  /**
   * \brief Check if the search container is empty.
   */
  [[nodiscard]] bool empty() const { return search_space.empty(); }

private:
  std::queue<State<StateRepr>> search_space;
  std::string m_name = "Breadth First Search";
};
