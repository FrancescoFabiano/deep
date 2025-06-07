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
#include <chrono>
#include <string>


/**
 * @brief Concept that enforces the required interface for a Search Strategy type `T`.
 *
 * This concept defines the contract that a type `T` must fulfill to be used with the `SpaceSearcher<T>` class.
 * It ensures that `T` provides push, pop, peek, and other functionalities.
 *
 * @tparam T The type to be checked against the required interface.
 */
template<typename T, typename StateRepr>
concept SearchStrategy = requires(T rep, State<StateRepr> s)
{
    /// Functor/lambda to push a state into the container.
    { rep.push(s) } -> std::same_as<void>;
    /// Functor/lambda to pop a state from the container.
    { rep.pop() } -> std::same_as<void>;
    /// Functor/lambda to peek at the next state in the container.
    { rep.peek() } -> std::same_as<State<StateRepr> >;
    /// Functor/lambda to clean container.
    { rep.reset() } -> std::same_as<void>;
    /// Functor/lambda to check if container is empty.
    { std::as_const(rep).empty() } -> std::same_as<bool>;
    /// Return the name of the used strategy.
    { std::as_const(rep).get_name() } -> std::same_as<std::string>;
};


/**
 * \brief Generic search strategy using a user-supplied container (queue/stack).
 * \tparam StateRepr The state representation type (must satisfy StateRepresentation).
 * \tparam Strategy The search strategy type (must satisfy SearchStrategy).
 */
template<StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
class SpaceSearcher {
public:
    /**
     * \brief Constructor with search name and strategy instance.
     * \param strategy The search strategy instance.
     */
    explicit SpaceSearcher(Strategy strategy);

    /**
     * \brief Default destructor.
     */
    ~SpaceSearcher() = default;

    /**
     * \brief Executes the search algorithm.
     *
     * \param initial The initial state to start the search from.
     * \return true if a goal state is found, false otherwise.
     */
    [[nodiscard]]
    bool search(State<StateRepr> &initial);

    /**
     * \brief Get the name of the search strategy.
     * \return The search type/name.
     */
    [[nodiscard]]
    std::string get_search_type() const noexcept;

    /**
     * \brief Get the number of expanded nodes by the search strategy.
     * \return The number of expanded nodes.
     */
    [[nodiscard]]
    unsigned int get_expanded_nodes() const noexcept;

    /**
     * \brief Get the time taken by the search strategy (in seconds).
     * \return The time taken (in seconds).
     */
    [[nodiscard]]
    std::chrono::duration<double> get_elapsed_seconds() const noexcept;

    /**
     * \brief Get the list of action IDs in the plan.
     * \return The list of action IDs.
     */
    [[nodiscard]]
    const ActionIdsList &get_plan_actions_id() const noexcept;

private:
    Strategy m_strategy; ///< Search strategy instance.

    unsigned int m_expanded_nodes = 0; ///< Counter for expanded nodes.
    std::chrono::duration<double> m_elapsed_seconds{}; ///< Time taken by the search.
    ActionIdsList m_plan_actions_id{}; ///< List of actions in the plan.

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
    bool search_sequential(State<StateRepr> &initial, const ActionsSet &actions, bool check_visited,
                           bool bisimulation_reduction);

    /**
     * \brief Executes the search algorithm in parallel (queue-based BFS only).
     *
     * \param initial The initial state to start the search from.
     * \param actions The list of actions to use.
     * \param check_visited Whether to check for visited states.
     * \param bisimulation_reduction Whether to apply bisimulation reduction.
     * \param num_threads The number of threads to use.
     * \return true if a goal state is found, false otherwise.
     *
     * \warning not too thoroughly tested, use with caution.
     */
    [[nodiscard]]
    bool search_parallel(State<StateRepr> &initial, const ActionsSet &actions, bool check_visited,
                         bool bisimulation_reduction, int num_threads);

    /// \name Plan Validation
    ///@{
    /**
     * \brief Validates a plan.
     * \param initial The initial state to start the search from.
     * \param check_visited Whether to check for visited states.
     * \param bisimulation_reduction Whether to apply bisimulation reduction.
     * \return true if the plan is valid, false otherwise.
     */
    [[nodiscard]]
    bool validate_plan(const State<StateRepr> &initial, bool check_visited,
                       bool bisimulation_reduction);

    /**
     * \brief Prints a DOT representation for an action in the execution plan.
     *
     * \param initial Boolean indicating if this the initial state.
     * \param last Boolean indicating if this is the last step.
     * \param action_name The name of the action being executed.
     * \param current The current state after executing the action.
     * \param dot_files_folder The folder where DOT files are stored.
     */
    static void print_dot_for_execute_plan(const bool initial, const bool last, const std::string &action_name,
                                           const State<StateRepr> &current, const std::string &dot_files_folder);

    ///@}
};


#include "search/SpaceSearcher.tpp"
