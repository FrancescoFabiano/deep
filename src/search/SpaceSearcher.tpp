/**
 * Implementation of \ref SpaceSearcher.h
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 29, 2025
 */

#include "search/SpaceSearcher.h"
#include "argparse/Configuration.h"
#include "utilities/ExitHandler.h"
#include <set>
#include <chrono>
#include <string>
#include <unordered_set>
#include <thread>
#include <atomic>
#include <algorithm>

template <StateRepresentation State, SearchStrategy<State> Strategy>
SpaceSearcher<State, Strategy>::SpaceSearcher(Strategy strategy)
{
    m_strategy = std::move(strategy);
}

template <StateRepresentation State, SearchStrategy<State> Strategy>
const std::string& SpaceSearcher<State, Strategy>::get_search_type() const noexcept { return m_strategy.get_name(); }


template <StateRepresentation State, SearchStrategy<State> Strategy>
bool SpaceSearcher<State, Strategy>::search(State& initial, int num_threads, const std::vector<Action>& plan)
{
    m_expanded_nodes = 0;

    const bool check_visited = Configuration::get_instance().get_check_visited();
    const bool bisimulation_reduction = Configuration::get_instance().get_bisimulation();

    // Defensive: Check if Domain singleton is available and actions are not empty
    const auto& domain_instance = Domain::get_instance();
    const auto& actions = domain_instance.get_actions();
    if (actions.empty())
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::SearchNoActions,
            "No actions available in the domain."
        );
    }

    if (bisimulation_reduction)
    {
        initial.calc_min_bisimilar();
    }

    const auto start_timing = std::chrono::system_clock::now();

    if (initial.is_goal())
    {
        m_elapsed_seconds = std::chrono::system_clock::now() - start_timing;
        return true;
    }

    // Dispatch
    if (plan.size() > 0)
    {
        // If a plan is provided, validate it
        return validate_plan(initial, actions, check_visited, bisimulation_reduction);
    }
    const bool result = (num_threads <= 1)
                            ? search_sequential(initial, actions, check_visited, bisimulation_reduction)
                            : search_parallel(initial, actions, check_visited, bisimulation_reduction, num_threads);

    if (result)
    {
        m_elapsed_seconds = std::chrono::system_clock::now() - start_timing;
    }

    return result;
}

template <StateRepresentation State, SearchStrategy<State> Strategy>
bool SpaceSearcher<State, Strategy>::search_sequential(const State& initial, const std::vector<Action>& actions,
                                                       const bool check_visited, const bool bisimulation_reduction)
{
    m_strategy.fresh();

    std::unordered_set<State> visited_states;
    m_expanded_nodes = 0;

    m_strategy.push(initial);
    if (check_visited)
    {
        visited_states.insert(initial);
    }

    while (!m_strategy.empty())
    {
        State current = m_strategy.peek();
        m_strategy.pop();
        ++m_expanded_nodes;

        for (const auto& action : actions)
        {
            if (current.is_executable(action))
            {
                State successor = current.compute_succ(action);
                if (bisimulation_reduction)
                {
                    successor.calc_min_bisimilar();
                }

                if (successor.is_goal())
                {
                    return true;
                }

                if (!check_visited || visited_states.insert(successor).second)
                {
                    m_strategy.push(successor);
                }
            }
        }
    }
    return false;
}

template <StateRepresentation State, SearchStrategy<State> Strategy>
bool SpaceSearcher<State, Strategy>::search_parallel(const State& initial, const std::vector<Action>& actions,
                                                     const bool check_visited, const bool bisimulation_reduction,
                                                     const int num_threads)
{
    std::unordered_set<State> visited_states;
    Strategy current_frontier;
    current_frontier.push(initial);

    if (check_visited)
    {
        visited_states.insert(initial);
    }

    std::atomic<size_t> total_expanded_nodes{0};

    while (!current_frontier.empty())
    {
        std::vector<std::thread> threads;
        Strategy next_frontier;
        std::vector<State> level_states = {};

        while (!current_frontier.empty())
        {
            level_states.push_back(current_frontier.peek());
            current_frontier.pop();
        }

        size_t chunk_size = (level_states.size() + num_threads - 1) / num_threads;
        std::atomic<bool> found_goal{false};
        std::vector<std::unordered_set<State>> local_visited(num_threads);
        std::vector<Strategy> local_frontiers(num_threads);

        for (int t = 0; t < num_threads; ++t)
        {
            threads.emplace_back([&, t]()
            {
                const size_t start = t * chunk_size;
                const size_t end = std::min(start + chunk_size, level_states.size());

                for (size_t i = start; i < end && !found_goal; ++i)
                {
                    State& popped_state = level_states[i];

                    for (const auto& tmp_action : actions)
                    {
                        if (popped_state.is_executable(tmp_action))
                        {
                            State tmp_state = popped_state.compute_succ(tmp_action);

                            if (bisimulation_reduction)
                            {
                                tmp_state.calc_min_bisimilar();
                            }

                            if (tmp_state.is_goal())
                            {
                                found_goal = true;
                                return;
                            }

                            if (!check_visited || local_visited[t].insert(tmp_state).second)
                            {
                                local_frontiers.push(tmp_state);
                            }
                        }
                    }
                }
            });
        }

        for (auto& th : threads)
        {
            if (th.joinable()) th.join();
        }

        if (found_goal)
        {
            m_expanded_nodes += total_expanded_nodes;
            return true;
        }

        // Merge local visited/frontier into global structures
        for (int t = 0; t < num_threads; ++t)
        {
            while (!local_frontiers[t].empty())
            {
                State s = local_frontiers[t].peek();
                local_frontiers[t].pop();

                if (!check_visited || visited_states.insert(s).second)
                {
                    push_fn(next_frontier, s);
                }
            }

            total_expanded_nodes += local_visited[t].size();
        }

        if (next_frontier.empty())
        {
            m_expanded_nodes += total_expanded_nodes;
            return false;
        }

        std::swap(current_frontier, next_frontier);
    }

    m_expanded_nodes += total_expanded_nodes;
    return false;
}


template <StateRepresentation State, SearchStrategy<State> Strategy>
bool SpaceSearcher<State, Strategy>::validate_plan(const State& initial, const std::vector<std::string>& plan, const bool check_visited, const bool bisimulation_reduction)
{
    std::unordered_set<State> visited_states;
    if (check_visited)
    {
        visited_states.insert(initial);
    }

    State current = initial;

    for (const auto& action_name : plan)
    {
        bool found_action = false;
        for (const auto& action : Domain::get_instance().get_actions())
        {
            if (action.get_name() == action_name)
            {
                found_action = true;
                if (current.is_executable(action))
                {
                    current = current.compute_succ(action);
                    if (bisimulation_reduction)
                    {
                        current.calc_min_bisimilar();
                    }
                    if (current.is_goal())
                    {
                        return true;
                    }
                    if (check_visited)
                    {
                        visited_states.insert(current);
                    }
                }
                else
                {
                    ExitHandler::exit_with_message(
                        ExitHandler::ExitCode::StateActionNotExecutableError,
                        std::string("The action \"") + action.get_name() + "\" was not executable while validating the plan."
                    );
                    return false; // Unreachable, but keeps compiler happy
                }
                break;
            }
        }
        if (!found_action)
        {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::ActionTypeConflict,
                std::string("Action \"") + action_name + "\" not found in domain actions while validating the plan."
            );
            return false; // Unreachable, but keeps compiler happy
        }
    }
    return current.is_goal();
}
