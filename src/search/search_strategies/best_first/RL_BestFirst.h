#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "BestFirst.h"
#include "neuralnets/FringeEvalRL.h"
#include "strategies/SatisfiedGoals.h"

enum class RefillMode { RANDOM, HEURISTIC };

template <StateRepresentation StateRepr>
class RL_BestFirst final : public BestFirst<StateRepr> {
public:
  using Base = BestFirst<StateRepr>;
  using Base::Base;

  explicit RL_BestFirst(const State<StateRepr> &initial_state)
      : Base(initial_state) {
    m_refill_mode = Configuration::get_instance().get_RL_heuristics() ==
                            RLHeuristicType::RNG
                        ? RefillMode::RANDOM
                        : RefillMode::HEURISTIC;

    m_seed = ArgumentParser::get_instance().get_RL_seed();
    if (m_seed < 0) {
      m_seed = std::random_device{}(); // Use random device if seed is negative
    }
    m_rng.seed(m_seed);

    m_adaptive = ArgumentParser::get_instance().get_RL_adaptive();
    m_exploration_max = std::max<std::size_t>(1, m_max_beam_size / 2);
    m_beam_width = m_max_beam_size;
    if (m_adaptive) {
      m_initial_unsatisfied =
          SatisfiedGoals::get_instance().get_unsatisfied_goals(initial_state);
      m_best_unsatisfied_goals = m_initial_unsatisfied;
    }
  }

  void set_refill_mode(const RefillMode mode) {
    m_refill_mode = mode;
    if (m_refill_mode == RefillMode::HEURISTIC && !m_reservoir.empty()) {
      std::make_heap(m_reservoir.begin(), m_reservoir.end(),
                     ReservoirCompare{});
    }
  }

  void push([[maybe_unused]] State<StateRepr> &s) override {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::SearchMethodNotImplemented,
        "Error: push of a single state is not implemented for RL_BestFirst.");
  }

  void push_vector(std::vector<State<StateRepr>> &states) override {
    if (states.empty() && m_reservoir.empty() && this->search_space.empty()) {
      return;
    }

    // Move the previously unexpanded beam states to the reservoir.
    while (!this->search_space.empty()) {
      auto not_expanded = this->search_space.top();
      this->search_space.pop();
      reservoir_push(std::move(not_expanded), false);
    }

    std::vector<State<StateRepr>> batch;
    batch.reserve(m_max_beam_size);

    // Add the newly generated states first. In adaptive mode the beam is
    // capped at the current dynamic width, minus the exploration budget so
    // random reservoir states always have beam slots to occupy, even when
    // the branching factor exceeds the beam.
    const std::size_t new_state_cap =
        m_adaptive
            ? std::max<std::size_t>(1, beam_limit() - exploration_budget())
            : m_max_beam_size;
    for (auto &s : states) {
      if (batch.size() < new_state_cap) {
        batch.push_back(std::move(s));
      } else {
        reservoir_push(std::move(s), true);
      }
    }

    // Fill remaining slots from the reservoir.
    refill_beam(batch);

    if (batch.empty()) {
      return;
    }

    const std::vector<float> heuristic_values =
        FringeEvalRL<StateRepr>::get_instance().get_score(batch);

    for (std::size_t i = 0; i < batch.size(); ++i) {
      batch[i].set_heuristic_value(heuristic_values[i]);
    }

    if (m_adaptive) {
      update_adaptive_schedule(batch);
    }

    for (auto &state : batch) {
      this->search_space.push(std::move(state));
    }
  }

  void reset() override {
    Base::reset();
    m_reservoir.clear();
    m_rng.seed(m_seed);
    m_exploration_current = 0;
    m_stall_rounds = 0;
    m_best_unsatisfied_goals = m_initial_unsatisfied;
    m_beam_width = m_max_beam_size;
    m_dive_rounds_left = 0;
    m_dive_count = 0;
  }

  [[nodiscard]] std::string get_name() const override {
    std::string name = "RLBeam x BestFirst Search (";
    if (m_refill_mode == RefillMode::RANDOM) {
      name += "random)";
    } else {
      name += "heuristic: " + this->m_heuristics_manager.get_used_h_name();
      if (this->m_heuristics_manager.get_used_h() == Heuristics::RL_H) {
        name += "-" + Configuration::get_instance().get_RL_heuristics_name();
      }
      name += ")";
    }
    if (m_adaptive) {
      name += " [adaptive]";
    }
    return name;
  }

  void pop() override {
    if (this->search_space.empty()) {
      if (!m_reservoir.empty()) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::SearchMethodError,
            "Error: Trying to pop from an empty queue while reservoir is not "
            "empty. Should call peek before pop.");
      } else {
        ExitHandler::exit_with_message(ExitHandler::ExitCode::SearchMethodError,
                                       "Error: Trying to pop from an empty "
                                       "queue while reservoir is also empty.");
      }
    }
    this->search_space.pop();
  }

  State<StateRepr> peek() override {
    if (this->search_space.empty()) {
      if (m_reservoir.empty()) {
        ExitHandler::exit_with_message(ExitHandler::ExitCode::SearchMethodError,
                                       "Error: Trying to peek from an empty "
                                       "queue while reservoir is also empty.");
      } else {
        if (m_refill_mode == RefillMode::RANDOM) {
          this->search_space.push(reservoir_take_random());
        } else {
          this->search_space.push(reservoir_take_best());
        }
      }
    }
    return this->search_space.top();
  }

  [[nodiscard]] bool empty() const override {
    return (this->search_space.empty() && m_reservoir.empty());
  }

private:
  std::size_t m_max_beam_size =
      ArgumentParser::get_instance().get_RL_fringe_size();
  RefillMode m_refill_mode{RefillMode::HEURISTIC};

  std::vector<State<StateRepr>> m_reservoir;
  std::size_t m_exploration_size =
      Configuration::get_instance().get_exploration_nodes();

  // Adaptive schedule: exploration starts at zero (pure exploitation) and
  // escalates when the best number of satisfied subgoals stops improving.
  // When escalation is exhausted without progress the search enters a dive
  // burst: a narrow beam (cycling widths) that mimics a small fringe size,
  // returning to full width if the dive does not bite.
  bool m_adaptive{false};
  std::size_t m_exploration_current{0};
  std::size_t m_exploration_max{1};
  std::size_t m_beam_width{1};
  int m_stall_rounds{0};
  int m_dive_rounds_left{0};
  std::size_t m_dive_count{0};
  unsigned short m_initial_unsatisfied{
      std::numeric_limits<unsigned short>::max()};
  unsigned short m_best_unsatisfied_goals{
      std::numeric_limits<unsigned short>::max()};
  static constexpr int STALL_PATIENCE = 5;
  static constexpr int DIVE_ROUNDS = 60;
  static constexpr std::array<std::size_t, 4> DIVE_WIDTHS{4, 8, 2, 16};

  std::mt19937_64 m_rng;
  int64_t m_seed{-1};

  struct ReservoirCompare {
    bool operator()(const State<StateRepr> &lhs,
                    const State<StateRepr> &rhs) const {
      const int lh = lhs.get_heuristic_value();
      const int rh = rhs.get_heuristic_value();
      if (lh != rh) {
        return lh > rh; // smaller heuristic = higher priority
      }
      return rhs < lhs;
    }
  };

  void refill_beam(std::vector<State<StateRepr>> &batch) {
    if (batch.size() >= beam_limit() || m_reservoir.empty()) {
      return;
    }

    if (m_refill_mode == RefillMode::RANDOM) {
      refill_beam_random(batch);
    } else {
      refill_beam_heuristic(batch);
    }
  }

  void refill_beam_random(std::vector<State<StateRepr>> &batch) {
    while (batch.size() < beam_limit() && !m_reservoir.empty()) {
      batch.push_back(reservoir_take_random());
    }
  }

  [[nodiscard]] std::size_t exploration_budget() const {
    return m_adaptive ? m_exploration_current : m_exploration_size;
  }

  [[nodiscard]] std::size_t beam_limit() const {
    return m_adaptive ? m_beam_width : m_max_beam_size;
  }

  void update_adaptive_schedule(const std::vector<State<StateRepr>> &batch) {
    const auto &satisfied_goals = SatisfiedGoals::get_instance();
    unsigned short best = std::numeric_limits<unsigned short>::max();
    for (const auto &state : batch) {
      best = std::min(best, satisfied_goals.get_unsatisfied_goals(state));
    }

    if (best < m_best_unsatisfied_goals) {
      m_best_unsatisfied_goals = best;
      m_stall_rounds = 0;
      m_exploration_current = 0;
      if (m_dive_rounds_left > 0) {
        m_dive_rounds_left = DIVE_ROUNDS; // The dive is biting: keep diving.
      }
      return;
    }

    if (m_dive_rounds_left > 0) {
      if (--m_dive_rounds_left == 0) {
        m_beam_width = m_max_beam_size; // Dive did not bite: back to full.
        m_stall_rounds = 0;
      }
      return;
    }

    if (++m_stall_rounds >= STALL_PATIENCE) {
      m_stall_rounds = 0;
      if (m_exploration_current < m_exploration_max) {
        m_exploration_current = m_exploration_current == 0
                                    ? 1
                                    : std::min(m_exploration_current * 2,
                                               m_exploration_max);
      } else {
        // Exploration exhausted: burst into a narrow dive, emulating the
        // small fringe sizes that solve what full width cannot.
        m_beam_width = std::min(DIVE_WIDTHS[m_dive_count++ % DIVE_WIDTHS.size()],
                                m_max_beam_size);
        m_exploration_current = 0;
        m_dive_rounds_left = DIVE_ROUNDS;
      }
    }
  }

  void refill_beam_heuristic(std::vector<State<StateRepr>> &batch) {
    if (batch.size() >= beam_limit() || m_reservoir.empty()) {
      return;
    }
    const std::size_t free_slots = beam_limit() - batch.size();

    // Fill most of the missing slots with the best reservoir states.
    // Keep a small amount of random exploration inside the beam budget.
    const std::size_t exploration_slots = std::min<std::size_t>(
        {free_slots, m_reservoir.size(), exploration_budget()});
    const std::size_t exploit_slots = free_slots - exploration_slots;

    for (std::size_t i = 0; i < exploit_slots && !m_reservoir.empty(); ++i) {
      batch.push_back(reservoir_take_best());
    }

    for (std::size_t i = 0; i < exploration_slots && !m_reservoir.empty();
         ++i) {
      batch.push_back(reservoir_take_random());
    }

    if (exploration_slots > 0 && !m_reservoir.empty()) {
      std::make_heap(m_reservoir.begin(), m_reservoir.end(),
                     ReservoirCompare{});
    }
  }

  void reservoir_push(State<StateRepr> &&candidate, const bool new_state) {
    if (m_refill_mode == RefillMode::RANDOM) {
      m_reservoir.push_back(std::move(candidate));
      return;
    }

    if (new_state) {
      candidate.set_heuristic_value(0);
    } else {
      // Change Heuristic value to be a different one maybe (like number of
      // subgoals)
      // Think about how to use RL heuristics (which is avg/min/max of the RL
      // assigned scores -- need to keep track of the various score) he
      // heuristic set to work will then be employed -- the previous is about
      // search
      const auto heuristic_value =
          this->m_heuristics_manager.get_heuristic_value(candidate);
      candidate.set_heuristic_value(heuristic_value);
    }

    m_reservoir.push_back(std::move(candidate));
    std::push_heap(m_reservoir.begin(), m_reservoir.end(), ReservoirCompare{});
  }

  State<StateRepr> reservoir_take_random() {
    std::uniform_int_distribution<std::size_t> distribution(
        0, m_reservoir.size() - 1);
    const std::size_t idx = distribution(m_rng);

    State<StateRepr> candidate = std::move(m_reservoir[idx]);
    m_reservoir[idx] = std::move(m_reservoir.back());
    m_reservoir.pop_back();
    return candidate;
  }

  State<StateRepr> reservoir_take_best() {
    std::pop_heap(m_reservoir.begin(), m_reservoir.end(), ReservoirCompare{});
    State<StateRepr> candidate = std::move(m_reservoir.back());
    m_reservoir.pop_back();
    return candidate;
  }
};
