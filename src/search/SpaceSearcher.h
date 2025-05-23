#pragma once

#include <vector>
#include <iostream>
#include <chrono>

#include "Action.h"
#include "Fluent.h"
#include "FluentFormula.h"
#include "State.h"

/**
 * @brief Concept for a strategy instance that exposes a StateType alias and required interface.
 */
template<typename Strategy>
concept SearchStrategyConcept = requires(Strategy strat, const std::vector<Action>& plan) {
    typename Strategy::StateType;
    { strat.search() } -> std::same_as<bool>;
    { strat.validate_plan(plan) } -> std::same_as<bool>;
};

/**
 * @class SpaceSearcher
 * @brief Class that wraps a search strategy instance, without exposing the State type explicitly.
 *
 * @tparam Strategy Strategy class type that must satisfy SearchStrategyConcept.
 */
template <SearchStrategyConcept Strategy>
class SpaceSearcher {
public:
    /**
     * @brief Constructor.
     *
     * @param strategy Pre-configured strategy instance that internally manages the initial state.
     */
    explicit SpaceSearcher(Strategy strategy)
        : m_strategy(std::move(strategy)) {}

    /**
     * @brief Launches the search.
     *
     * @return true if a plan was found, false otherwise.
     */
    bool run_search(int num_threads = 1) {
        return m_strategy.search();
    }

    /**
     * @brief Validates a plan.
     *
     * @param plan The sequence of actions to validate.
     * @return true if the plan is valid, false otherwise.
     */
    bool validate_plan(const std::vector<Action>& plan) {
        return m_strategy.validate_plan(plan);
    }

private:
    Strategy m_strategy; ///< The search strategy instance.
};
