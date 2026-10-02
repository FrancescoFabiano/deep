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
#include <random>
#include <string>
#include <vector>

#include "argparse/Configuration.h"
#include "heuristics/HeuristicsManager.h"
#ifdef USE_NEURALNETS
#include "neuralnets/FringeEvalRL.h"
#endif

/** \brief How HFS/A* order states with equal values (--tie_breaking). */
enum class TieBreaking { None, Fifo, Lifo, Random };

inline TieBreaking configured_tie_breaking() {
  const std::string &mode = ArgumentParser::get_instance().get_tie_breaking();
  if (mode == "fifo") {
    return TieBreaking::Fifo;
  }
  if (mode == "lifo") {
    return TieBreaking::Lifo;
  }
  if (mode == "random") {
    return TieBreaking::Random;
  }
  return TieBreaking::None;
}

/**
 * \brief Open-list order: lower primary key first (h for HFS, f for A*). With
 * --tie_breaking none that is all, as before tie-breaking existed; otherwise
 * then lower secondary key (A*: h) and then the insertion key: oldest first
 * (fifo), newest first (lifo) or a seeded random key (random).
 *
 * \tparam StateRepr The state representation type.
 * \return true if state1 has a lower priority than state2 (max-heap
 * convention of std::priority_queue).
 */
template <StateRepresentation StateRepr> struct StateComparator {
  TieBreaking mode = configured_tie_breaking();

  bool operator()(const State<StateRepr> &state1,
                  const State<StateRepr> &state2) const {
    if (state1.get_search_primary() != state2.get_search_primary()) {
      return state1.get_search_primary() > state2.get_search_primary();
    }
    if (mode == TieBreaking::None) {
      return false;
    }
    if (state1.get_search_secondary() != state2.get_search_secondary()) {
      return state1.get_search_secondary() > state2.get_search_secondary();
    }
    // fifo and random: smaller key first (insertion counter / random draw)
    return mode == TieBreaking::Lifo
               ? state1.get_search_order() < state2.get_search_order()
               : state1.get_search_order() > state2.get_search_order();
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
  virtual void push_initial(const State<StateRepr> &s) {
    State<StateRepr> initial = s;
    enqueue(std::move(initial), 0, 0);
  }

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
   * \brief Push \p s with its ordering keys (see \ref StateComparator); the
   * insertion order is assigned here.
   */
  void enqueue(State<StateRepr> &&s, const double primary,
               const double secondary) {
    const std::uint64_t order =
        m_tie_breaking == TieBreaking::Random ? m_tie_rng() : m_next_order++;
    s.set_search_keys(primary, secondary, order);
    search_space.push(std::move(s));
  }

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
        constexpr double max_value = std::numeric_limits<int>::max();
        // the model's distance, rounded unless --GNN_raw_distance
        const double distance =
            std::isfinite(raw_scores[i])
                ? std::min(std::max(0.0, -static_cast<double>(raw_scores[i])),
                           max_value)
                : max_value;
        const double h = m_raw_distance ? distance : std::round(distance);
        const double g = batch[i].get_plan_length();
        const double primary = batched_uses_depth() ? g + h : h;
        batch[i].set_heuristic_value(
            static_cast<int>(std::llround(std::min(primary, max_value))));
        enqueue(std::move(batch[i]), primary, batched_uses_depth() ? h : 0);
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
  std::uint64_t m_next_order = 0; ///< Insertion counter for tie-breaking.
  TieBreaking m_tie_breaking = configured_tie_breaking();
  std::mt19937_64 m_tie_rng{
      ArgumentParser::get_instance().get_tie_breaking_seed()}; ///< random keys
  bool m_raw_distance = ArgumentParser::get_instance()
                            .get_GNN_raw_distance(); ///< Unrounded GNN h.
};
