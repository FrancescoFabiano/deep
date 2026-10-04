/**
 * \class IterativeDepthFirst
 * \brief Implements the Iterative Depth First Search strategy to explore the
 * search space.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 29, 2025
 */

#pragma once
#include "states/State.h"
#include <stack>
#include <string>
#include <utility>

/**
 * \brief IterativeDepthFirst search strategy for use with SpaceSearcher.
 * \tparam StateRepr The state representation type (must satisfy
 * StateRepresentation).
 */
template <StateRepresentation StateRepr> class IterativeDepthFirst {
public:
  /**
   * \brief Default constructor.
   */
  explicit IterativeDepthFirst(const State<StateRepr> &initial_state) {
    m_initial_state = initial_state;
  }

  /**
   * \brief Push a state into the search container.
   */
  void push(State<StateRepr> &s) {
    if (s.get_plan_length() <= max_depth) {
      search_space.push(std::move(s)); // takes the successor
    } else {
      m_reached_max_depth = true;
    }
  }

  /**
   * \brief Push the initial state into the search container.
   */
  void push_initial(const State<StateRepr> &s) {
    m_initial_state = s; // the searcher's (contracted) initial state
    State<StateRepr> initial = s;
    push(initial);
  }

  /**
   * \brief True once after each restart from the initial state, so the
   * searcher can clear its visited set for the new depth bound.
   */
  [[nodiscard]] bool consume_restart() {
    return std::exchange(m_restarted, false);
  }

  /**
   * \brief Whether \p s lies within the current depth bound. A state beyond
   * it is neither goal-tested nor pushed, and the next iteration deepens.
   */
  [[nodiscard]] bool within_bound(const State<StateRepr> &s) {
    if (s.get_plan_length() <= max_depth) {
      return true;
    }
    m_reached_max_depth = true;
    return false;
  }

  /**
   * \brief Push a list of states into the search container. Not implemented for
   * searches that are not RL-based
   */
  void push_vector([[maybe_unused]] std::vector<State<StateRepr>> &s) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::SearchMethodNotImplemented,
        "Error: push of a vector of states is not implemented for IDFS. It is "
        "solely added for RL reasoning");
  }

  /**
   * \brief Pop a state from the search container.
   */
  void pop() { search_space.pop(); }

  State<StateRepr> peek() const { return search_space.top(); }

  /**
   * \brief Remove the next state from the container and return it (moved
   * out, not copied).
   */
  State<StateRepr> take() {
    // The previous iteration is exhausted and cut some state: restart from
    // the initial state with a deeper bound, so each iteration expands the
    // initial state first.
    if (search_space.empty() && m_reached_max_depth) {
      search_space.push(m_initial_state);
      m_reached_max_depth = false;
      max_depth += iterative_step;
      m_restarted = true;
    }
    State<StateRepr> next = std::move(search_space.top());
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
  void reset() {
    search_space = std::stack<State<StateRepr>>();
    max_depth = 1;
    m_reached_max_depth = false;
    m_restarted = false;
  }

  /**
   * \brief Whether \p s may be expanded: its successors would lie beyond the
   * current bound otherwise (the bound is then marked as reached, so a
   * deeper iteration follows).
   */
  [[nodiscard]] bool expandable(const State<StateRepr> &s) {
    if (s.get_plan_length() < max_depth) {
      return true;
    }
    m_reached_max_depth = true;
    return false;
  }

  /**
   * \brief Check if the search container is empty.
   */
  /** \brief Empty only when no deeper iteration is pending either. */
  [[nodiscard]] bool empty() const {
    return search_space.empty() && !m_reached_max_depth;
  }

private:
  std::stack<State<StateRepr>> search_space;
  std::string m_name =
      "Iterative Depth First Search"; ///< Name of the search strategy.
  State<StateRepr> m_initial_state;   ///< Initial state of the search used when
                                      ///< we reset the search space.
  short iterative_step = 1; ///< Iterative step for the search strategy, used to
                            ///< control the increase in depth of the search.
  short max_depth =
      1; ///< Maximum depth of the search, used to control the maximum depth of
         ///< the search. This will be increased at the beginning.
  bool m_restarted = false; ///< Set by a restart, see \ref consume_restart.
  bool m_reached_max_depth =
      false; ///< True when a state beyond the bound was cut in this iteration
             ///< (only then does a deeper iteration follow).
};