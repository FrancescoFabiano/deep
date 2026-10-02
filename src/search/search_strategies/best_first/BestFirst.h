/**
 * \class BestFirst
 * \brief Abstract base class for Best First Search strategies.
 *
 * This class provides the core search queue functionality for heuristic-driven
 * search. Derived classes must implement the \ref push and \ref get_name
 * methods.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date July 10, 2025
 */

#pragma once
#include "states/State.h"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <queue>
#include <string>
#include <vector>

#include "argparse/Configuration.h"
#include "heuristics/HeuristicsManager.h"
#ifdef USE_NEURALNETS
#include "neuralnets/FringeEvalRL.h"
#endif

/**
 * \brief Compares two states based on their heuristic value.
 *
 * \tparam StateRepr The state representation type.
 * \param state1 The first state to compare.
 * \param state2 The second state to compare.
 * \return true if state1 has a higher heuristic value than state2, false
 * otherwise.
 *
 * \note Lower scores are better. States with higher heuristic values have lower
 * priority.
 */
template <StateRepresentation StateRepr> struct StateComparator {
  bool operator()(const State<StateRepr> &state1,
                  const State<StateRepr> &state2) const {
    return state1.get_heuristic_value() > state2.get_heuristic_value();
  }
};

/**
 * \brief Abstract base class for Best First Search strategies.
 *
 * \tparam StateRepr The state representation type (must satisfy
 * StateRepresentation).
 */
template <StateRepresentation StateRepr> class BestFirst {
public:
  /**
   * \brief Constructor.
   *
   * \param initial_state The initial state used to initialize the heuristics
   * manager.
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
   * Must be implemented by derived classes to define filtering or priority
   * behavior.
   *
   * \param s The state to push.
   */
  virtual void push(State<StateRepr> &s) = 0;

  /**
   * \brief Push the initial state into the search container.
   */
  virtual void push_initial(const State<StateRepr> &s) { search_space.push(s); }

  /**
   * \brief Push a list of states into the search container. Not implemented for
   * searches that are not RL-based
   */
  virtual void push_vector([[maybe_unused]] std::vector<State<StateRepr>> &s) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::SearchMethodNotImplemented,
        "Error: push of a vector of states is not implemented for non-RL "
        "searches. "
        "It is solely added for RL reasoning");
  }

  /**
   * \brief Pop the state with the highest priority (lowest heuristic value).
   */
  virtual void pop() { search_space.pop(); }

  /**
   * \brief Peek at the next state in the search container without removing it.
   *
   * Removed const because in RL_BestFirst we swap element among the reservoir
   * and the search space, so we need to be able to modify the state in the peek
   * function
   *
   * \return The next state in the priority queue.
   */
  [[nodiscard]] virtual State<StateRepr> peek() {
    flush_pending();
    return search_space.top();
  }

  /**
   * \brief Remove the best state and return it, moved out instead of copied.
   *
   * \details The queue's elements are not const objects, so moving out of
   * top() is legal; pop() then only compares heuristic values, which the move
   * leaves intact, before destroying the moved-from element.
   */
  [[nodiscard]] virtual State<StateRepr> take() {
    flush_pending();
    State<StateRepr> next =
        std::move(const_cast<State<StateRepr> &>(search_space.top()));
    search_space.pop();
    return next;
  }

  /**
   * \brief Pure virtual function to return the name of the search strategy.
   *
   * \return A descriptive name of the strategy and heuristic used.
   */
  [[nodiscard]] virtual std::string get_name() const = 0;

  /**
   * \brief Clear and reset the search container.
   */
  virtual void reset() {
    search_space = StatePriorityQueue();
    m_pending.clear();
  }

  /**
   * \brief Check whether the search container is empty.
   *
   * \return true if the container is empty, false otherwise.
   */
  [[nodiscard]] virtual bool empty() const {
    return search_space.empty() && m_pending.empty();
  }

protected:
  /**
   * \brief Whether the heuristic is the batched GNN (--GNN_batch > 0).
   *
   * Successors are then buffered by \ref push and scored together, up to
   * --GNN_batch per model call, the next time the search picks a state.
   */
  [[nodiscard]] bool batched() const noexcept { return m_batch_size > 0; }

  /** \brief Buffer a state for the next batched evaluation. */
  void push_pending(State<StateRepr> &s) { m_pending.push_back(std::move(s)); }

  /**
   * \brief Whether the batched priority adds the state depth (A*) to the
   * model's distance (HFS uses the distance alone).
   */
  [[nodiscard]] virtual bool batched_uses_depth() const { return false; }

  /**
   * \brief Score the buffered states and move them to the search space.
   *
   * The fringe export of the GNN outputs -distance for every slot, so the
   * priority is the absolute distance (rounded, floored at 0), plus the
   * depth when \ref batched_uses_depth. Batches are independent: a state's
   * value does not depend on the other states scored with it.
   */
  void flush_pending() {
    if (m_pending.empty()) {
      return;
    }
#ifdef USE_NEURALNETS
    auto &evaluator = FringeEvalRL<StateRepr>::get_instance();
    std::vector<float> raw_scores;
    for (std::size_t first = 0; first < m_pending.size();
         first += m_batch_size) {
      const std::size_t last = std::min(m_pending.size(), first + m_batch_size);
      std::vector<State<StateRepr>> batch(
          std::make_move_iterator(m_pending.begin() + first),
          std::make_move_iterator(m_pending.begin() + last));
      [[maybe_unused]] const auto ranks =
          evaluator.get_score(batch, &raw_scores);

      for (std::size_t i = 0; i < batch.size(); ++i) {
        constexpr double max_value = std::numeric_limits<short>::max();
        double value = std::isfinite(raw_scores[i])
                           ? std::max(0.0, -static_cast<double>(raw_scores[i]))
                           : max_value;
        if (batched_uses_depth()) {
          value += batch[i].get_plan_length();
        }
        batch[i].set_heuristic_value(
            static_cast<short>(std::lround(std::min(value, max_value))));
        search_space.push(std::move(batch[i]));
      }
    }
#else
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::HeuristicsBadDeclaration,
        "--GNN_batch needs neural network support. Please recompile with the "
        "nn option.");
#endif
    m_pending.clear();
  }

  /** \brief Name suffix of the batched heuristic, empty when per-state. */
  [[nodiscard]] std::string batched_name() const {
    return batched() ? ", batch " + std::to_string(m_batch_size) : "";
  }

  /**
   * \brief Priority queue for managing the search space.
   *
   * States are ordered based on their heuristic value, with lower values having
   * higher priority.
   */
  using StatePriorityQueue =
      std::priority_queue<State<StateRepr>, std::vector<State<StateRepr>>,
                          StateComparator<StateRepr>>;

  StatePriorityQueue search_space; ///< The search space represented as a
                                   ///< priority queue of states.

  HeuristicsManager<StateRepr>
      m_heuristics_manager; ///< Heuristics manager to compute heuristic values
                            ///< for states.

private:
  std::size_t m_batch_size = static_cast<std::size_t>(
      ArgumentParser::get_instance()
          .get_GNN_batch_size()); ///< Batched GNN size (0 = off).
  std::vector<State<StateRepr>>
      m_pending; ///< Successors waiting for the batched GNN evaluation.
};
