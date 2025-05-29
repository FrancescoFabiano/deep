/**
 * \class SpaceSearcher
 * \brief Implements a generic search strategy (BFS/DFS) using a user-supplied container.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 29, 2025
 */

#pragma once
#include "states/State.h"
#include "argparse/ArgumentParser.h"
#include <chrono>
#include <string>
#include <thread>



/**
 * @brief Concept that enforces the required interface for a Search Strategy type `T`.
 *
 * This concept defines the contract that a type `T` must fulfill to be used with the `SpaceSearcher<T>` class.
 * It ensures that `T` provides push, pop, peek, and other functionalities.
 *
 * @tparam T The type to be checked against the required interface.
 */
template<typename T, StateRepresentation State>
concept SearchStrategy = requires(T rep, State s) {
    /// Functor/lambda to push a state into the container.
    { rep.push(s) } -> std::same_as<void>;
    /// Functor/lambda to pop a state from the container.
    { rep.pop() } -> std::same_as<void>;
    /// Functor/lambda to peek at the next state in the container.
    { rep.peek() } -> std::same_as<State>;
    /// Functor/lambda to clean container.
    { rep.reset() } -> std::same_as<void>;
    /// Functor/lambda to check if container is empty.
    { rep.empty() } -> std::same_as<bool>;
    /// Return the name of the used strategy.
    { rep.get_name() } -> std::same_as<std::string>;
};


/**
 * \brief Generic search strategy using a user-supplied container (queue/stack).
 * \tparam State The state representation type (must satisfy StateRepresentation).
 * \tparam Strategy The search strategy type (must satisfy SearchStrategy).
 */
template <StateRepresentation State, SearchStrategy<State> Strategy>
class SpaceSearcher {
public:
    /**
     * \brief Constructor with search name and strategy instance.
     * \param strategy The search strategy instance.
     */
    SpaceSearcher(Strategy strategy);

    /**
     * \brief Default destructor.
     */
    ~SpaceSearcher() = default;

    /**
     * \brief Executes the search algorithm.
     *
     * \param initial The initial state to start the search from.
     * \param num_threads The number of threads to use (if > 1, parallel BFS is used).
     * \param plan The plan to validate, if this is empty then standard search is executed
     * \return true if a goal state is found, false otherwise.
     */
    [[nodiscard]]
    bool search(State& initial, int num_threads, const std::vector<Action>& plan = {});

    /**
     * \brief Get the name of the search strategy.
     * \return The search type/name.
     */
    [[nodiscard]]
    const std::string& get_search_type() const noexcept;

    /// \name Search Methods
    ///@{
    /**
     * \brief Launches the portfolio-method search (multiple configurations).
     *
     * For now, uses a fixed configuration: BFS + ALL Heuristics + DFS (depending on the number of threads).
     * \todo Add a parser from file that sets the priority list as parameters.
     * \return true if a plan was found, false otherwise.
     */
    [[nodiscard]]
    bool run_portfolio_search();

private:

    Strategy m_strategy; ///< Search strategy instance.

    unsigned int m_expanded_nodes = 0; ///< Counter for expanded nodes.
    std::chrono::duration<double> m_elapsed_seconds{}; ///< Time taken by the search.

    /**
     * \brief Executes the search algorithm sequentially using the provided container and operations.
     *
     * \param initial The initial state to start the search from.
     * \param actions The list of actions to use.
     * \param check_visited Whether to check for visited states.
     * \param bisimulation_reduction Whether to apply bisimulation reduction.
     * \return true if a goal state is found, false otherwise.
     */
    [[nodiscard]]
    bool search_sequential(const State& initial, const std::vector<Action>& actions, bool check_visited, bool bisimulation_reduction);

    /**
     * \brief Executes the search algorithm in parallel (queue-based BFS only).
     *
     * \param initial The initial state to start the search from.
     * \param actions The list of actions to use.
     * \param check_visited Whether to check for visited states.
     * \param bisimulation_reduction Whether to apply bisimulation reduction.
     * \param num_threads The number of threads to use.
     * \return true if a goal state is found, false otherwise.
     */
    [[nodiscard]]
    bool search_parallel(const State &initial, const std::vector<Action> &actions, bool check_visited, bool bisimulation_reduction, int num_threads);

    /// \name Plan Validation
    ///@{
    /**
     * \brief Validates a plan.
     * \param initial The initial state to start the search from.
     * \param plan The sequence of actions to validate.
     * \param check_visited Whether to check for visited states.
     * \param bisimulation_reduction Whether to apply bisimulation reduction.
     * \return true if the plan is valid, false otherwise.
     */
    [[nodiscard]]
    bool validate_plan(const State& initial, const std::vector<std::string>& plan, bool check_visited, bool bisimulation_reduction);
    ///@}

};


#include "search/SpaceSearcher.tpp"
