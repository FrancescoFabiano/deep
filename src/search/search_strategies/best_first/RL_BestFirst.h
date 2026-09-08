#pragma once

#include <algorithm>
#include <cstddef>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "BestFirst.h"
#include "neuralnets/FringeEvalRL.h"

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

    // Add the newly generated states first.
    for (auto &s : states) {
      if (batch.size() < m_max_beam_size) {
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
      this->search_space.push(std::move(batch[i]));
    }
  }

  void reset() override {
    Base::reset();
    m_reservoir.clear();
    m_rng.seed(m_seed);
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
      Configuration::get_exploration_nodes();

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
    if (batch.size() >= m_max_beam_size || m_reservoir.empty()) {
      return;
    }

    if (m_refill_mode == RefillMode::RANDOM) {
      refill_beam_random(batch);
    } else {
      refill_beam_heuristic(batch);
    }
  }

  void refill_beam_random(std::vector<State<StateRepr>> &batch) {
    while (batch.size() < m_max_beam_size && !m_reservoir.empty()) {
      batch.push_back(reservoir_take_random());
    }
  }

  void refill_beam_heuristic(std::vector<State<StateRepr>> &batch) {
    const std::size_t free_slots = m_max_beam_size - batch.size();
    if (free_slots == 0 || m_reservoir.empty()) {
      return;
    }

    // Fill most of the missing slots with the best reservoir states.
    // Keep a small amount of random exploration inside the beam budget.
    const std::size_t exploration_slots = std::min<std::size_t>(
        {free_slots, m_reservoir.size(), m_exploration_size});
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
