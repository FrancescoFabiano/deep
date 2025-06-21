/**
 * Implementation of \ref SpaceSearcher.h
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 29, 2025
 */

#include "search/SpaceSearcher.h"
#include "argparse/ArgumentParser.h"
#include "utilities/ExitHandler.h"
#include "states/State.h"
#include <set>
#include <chrono>
#include <string>
#include <thread>
#include <atomic>
#include <algorithm>
#include <mutex> // <-- Add this
#include <ranges>

#include "Configuration.h"
#include "HelperPrint.h"

template<StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
SpaceSearcher<StateRepr, Strategy>::SpaceSearcher(Strategy strategy, std::atomic<bool> &cancel_flag)
    : m_strategy(std::move(strategy)), m_cancel_flag(cancel_flag) {
}

template<StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
std::string SpaceSearcher<StateRepr, Strategy>::get_search_type() const noexcept {
    return m_strategy.get_name();
}

template<StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
unsigned int SpaceSearcher<StateRepr, Strategy>::get_expanded_nodes() const noexcept {
    return m_expanded_nodes;
}

template<StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
std::chrono::duration<double> SpaceSearcher<StateRepr, Strategy>::get_elapsed_seconds() const noexcept {
    return m_elapsed_seconds;
}

template<StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
const ActionIdsList &SpaceSearcher<StateRepr, Strategy>::get_plan_actions_id() const noexcept {
    return m_plan_actions_id;
}


template<StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
bool SpaceSearcher<StateRepr, Strategy>::search(const State<StateRepr> &passed_initial) {
    m_expanded_nodes = 0;

    const bool check_visited = Configuration::get_instance().get_check_visited();

    // Defensive: Check if Domain singleton is available and actions are not empty
    const auto &domain_instance = Domain::get_instance();
    const auto &actions = domain_instance.get_actions();
    if (actions.empty()) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::SearchNoActions,
            "No actions available in the domain."
        );
    }

    const auto start_timing = std::chrono::system_clock::now();

    auto thread_safe_initial = passed_initial;
    // Make a copy to avoid modifying the original state (avoid side effects for multi-threading)

    if (Configuration::get_instance().get_bisimulation()) {
        thread_safe_initial.contract_with_bisimulation();
    }


    if (thread_safe_initial.is_goal()) {
        m_elapsed_seconds = std::chrono::system_clock::now() - start_timing;
        return true;
    }

    bool result;
    // Dispatch
    if (ArgumentParser::get_instance().get_execute_plan()) {
        // If a plan is provided, validate it
        result = validate_plan(thread_safe_initial, check_visited);
    } else {
        const int num_threads = ArgumentParser::get_instance().get_threads_per_search();
        result = (num_threads <= 1)
                     ? search_sequential(thread_safe_initial, actions, check_visited)
                     : search_parallel(thread_safe_initial, actions, check_visited,
                                       num_threads);
    }

    m_elapsed_seconds = std::chrono::system_clock::now() - start_timing;

    return result;
}

template<StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
bool SpaceSearcher<StateRepr, Strategy>::search_sequential(State<StateRepr> &initial,
                                                           const ActionsSet &actions,
                                                           const bool check_visited) {
    m_strategy.reset();

    std::set<State<StateRepr> > visited_states;
    /// \warning cannot use unordered_set because I am missing a clear way of hashing the state
    m_expanded_nodes = 0;

    m_strategy.push(initial);
    if (check_visited) {
        visited_states.insert(initial);
    }

    while (!m_strategy.empty()) {
        if (m_cancel_flag.load()) {
            return false;
            // Exit early if cancellation requested (it happens when another threads find the solution first)
        }
        State current = m_strategy.peek();
        m_strategy.pop();
        ++m_expanded_nodes;

        for (const auto &action: actions) {
            if (current.is_executable(action)) {
                State successor = current.compute_successor(action);

                /// DEBUG \todo remove this, only for bisimulation testing
                /// check_bisimulation_equivalence(successor);

                if (Configuration::get_instance().get_bisimulation()) {
                    successor.contract_with_bisimulation();
                }


                if (successor.is_goal()) {
                    m_plan_actions_id = successor.get_executed_actions();
                    return true;
                }

                if (!check_visited || visited_states.insert(successor).second) {
                    m_strategy.push(successor);
                }
            }
        }
    }
    return false;
}

template<StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
bool SpaceSearcher<StateRepr, Strategy>::search_parallel(State<StateRepr> &initial,
                                                         const ActionsSet &actions,
                                                         const bool check_visited,
                                                         const int num_threads) {
    /*std::set<State<StateRepr>> visited_states;
    /// \warning cannot use unordered_set because I am missing a clear way of hashing the state
    Strategy current_frontier(initial);
    current_frontier.push(initial);

    if (check_visited)
    {
        visited_states.insert(initial);
    }

    std::atomic<size_t> total_expanded_nodes{0};
    std::mutex plan_mutex; // <-- Add this mutex

    while (!current_frontier.empty())
    {
        std::vector<std::thread> threads;
        Strategy next_frontier(initial);
        std::vector<State<StateRepr>> level_states = {};

        while (!current_frontier.empty())
        {
            level_states.push_back(current_frontier.peek());
            current_frontier.pop();
        }

        size_t chunk_size = (level_states.size() + num_threads - 1) / num_threads;
        std::atomic<bool> found_goal{false};
        std::vector<std::set<State<StateRepr>>> local_visited(num_threads);
        /// \warning cannot use unordered_set because I am missing a clear way of hashing the state

        std::vector<Strategy> local_frontiers;
        local_frontiers.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t)
        {
            local_frontiers.emplace_back(initial);
        }
        for (int t = 0; t < num_threads; ++t)
        {
            threads.emplace_back([&, t]()
            {
                const size_t start = t * chunk_size;
                const size_t end = std::min(start + chunk_size, level_states.size());

                for (size_t i = start; i < end && !found_goal; ++i)
                {
                    State<StateRepr>& popped_state = level_states[i];

                    for (const auto& tmp_action : actions)
                    {
                        if (popped_state.is_executable(tmp_action))
                        {
                            State<StateRepr> tmp_state = popped_state.compute_successor(tmp_action);

                            if (bisimulation_reduction)
                            {
                                tmp_state.contract_with_bisimulation();
                            }

                            if (tmp_state.is_goal())
                            {
                                found_goal = true;
                                {
                                    std::lock_guard<std::mutex> lock(plan_mutex); // <-- Protect this access
                                    m_plan_actions_id = tmp_state.get_executed_actions();
                                }
                                return;
                            }

                            if (!check_visited || local_visited[t].insert(tmp_state).second)
                            {
                                local_frontiers[t].push(tmp_state);
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
                    next_frontier.push(s);
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
    return false;*/
    (void) initial;
    (void) actions;
    (void) check_visited;
    (void) num_threads;
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::SearchParallelNotImplemented,
        "Parallel search is not implemented yet. Please use sequential search."
    );

    // Unreachable code, but keeps compiler happy
    std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
}


template<StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
bool SpaceSearcher<StateRepr, Strategy>::validate_plan(const State<StateRepr> &initial, const bool check_visited) {
    std::set<State<StateRepr> > visited_states;
    /// \warning cannot use unordered_set because I am missing a clear way of hashing the state
    if (check_visited) {
        visited_states.insert(initial);
    }

    const std::string dot_files_folder = std::string(OutputPaths::EXEC_PLAN_FOLDER) + "/" + Domain::get_instance().
                                         get_name()
                                         + "/";
    std::filesystem::create_directories(dot_files_folder);


    State<StateRepr> current = initial;
    if (Configuration::get_instance().get_bisimulation()) {
        current.contract_with_bisimulation();
    }
    print_dot_for_execute_plan(true, false, "initial", current, dot_files_folder);

    const auto &plan = ArgumentParser::get_instance().get_execution_actions();

    for (auto it = plan.begin(); it != plan.end(); ++it) {
        const auto &action_name = *it;
        bool is_last = (std::next(it) == plan.end());

        bool found_action = false;
        for (const auto &action: Domain::get_instance().get_actions()) {
            if (action.get_name() == action_name) {
                found_action = true;
                if (current.is_executable(action)) {
                    ++m_expanded_nodes;
                    current = current.compute_successor(action);
                    if (Configuration::get_instance().get_bisimulation()) {
                        current.contract_with_bisimulation();
                    }
                    print_dot_for_execute_plan(false, is_last, action_name, current,
                                               dot_files_folder);
                    if (current.is_goal()) {
                        m_plan_actions_id = current.get_executed_actions();
                        if (!is_last) {
                            auto &os = ArgumentParser::get_instance().get_output_stream();
                            os << "\n[WARNING] Plan found before the entire plan was used.";
                            os << std::endl;
                        }
                        return true;
                    }
                    if (check_visited && !visited_states.insert(current).second) {
                        auto &os = ArgumentParser::get_instance().get_output_stream();
                        os <<
                                "\n[WARNING] While executing the plan, found an already visited state after the execution of the actions:\n";
                        HelperPrint::get_instance().print_list(current.get_executed_actions());
                        os << "\nThis means that the plan is not optimal." << std::endl;
                    }
                    if (is_last) {
                        auto &os = ArgumentParser::get_instance().get_output_stream();
                        os << "\n[WARNING] No plan found after the execution of:\n";
                        HelperPrint::get_instance().print_list(current.get_executed_actions());
                        os << std::endl;
                    }
                } else {
                    ExitHandler::exit_with_message(
                        ExitHandler::ExitCode::StateActionNotExecutableError,
                        std::string("The action \"") + action.get_name() +
                        "\" was not executable while validating the plan."
                    );
                    return false; // Unreachable, but keeps compiler happy
                }
                break;
            }
        }
        if (!found_action) {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::ActionTypeConflict,
                std::string("Action \"") + action_name + "\" not found in domain actions while validating the plan."
            );
            return false; // Unreachable, but keeps compiler happy
        }
    }
    return current.is_goal();
}

template<StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
void SpaceSearcher<StateRepr, Strategy>::print_dot_for_execute_plan(const bool initial, const bool last,
                                                                    const std::string &action_name,
                                                                    const State<StateRepr> &current,
                                                                    const std::string &dot_files_folder) {
    if (!ArgumentParser::get_instance().get_verbose())
        return;

    const std::string dot_extension = ".dot";
    std::size_t dot_count = std::count_if(
        std::filesystem::directory_iterator(dot_files_folder),
        std::filesystem::directory_iterator{},
        [dot_extension](const auto &entry) {
            return entry.path().extension() == dot_extension;
        }
    );


    std::string print_name = dot_files_folder;
    std::ostringstream oss;
    oss << std::setw(5) << std::setfill('0') << static_cast<int>(dot_count);
    print_name += oss.str() + "-";
    if (!initial) {
        print_name += action_name;
    } else {
        print_name += "initial";
    }

    const std::string adj_dot_extension = Configuration::get_instance().get_bisimulation()
                                              ? ("-bis" + dot_extension)
                                              : dot_extension;

    std::string ofstream_name = print_name + std::string(adj_dot_extension);
    if (std::ofstream ofs(ofstream_name); ofs.is_open()) {
        current.print_dot_format(ofs);
    }

    //DEBUG
    /*
    State<StateRepr> temp = current;
    temp.contract_with_bisimulation();
    std::string bis_ofstream_name = print_name + "_bis" + std::string(dot_extension);
    if (std::ofstream bis_ofs(bis_ofstream_name); bis_ofs.is_open())
    {
        temp.print_dot_format(bis_ofs);
    }
    */

    //DEBUG
    /// \warning This warning is a false positive. Bisimulation does affect the state. (suppressed)
    // NOLINTNEXTLINE
    /*if (temp != current)
    {
        // ReSharper disable once CppDFAUnreachableCode
        auto& os = ArgumentParser::get_instance().get_output_stream();
        os << "\nThe state and its bisimulation differ after the actions:";
        HelperPrint::get_instance().print_list(current.get_executed_actions());
        os << std::endl;
    }*/

    if (last) {
        std::string script_cmd = "./scripts/dot_to_png.sh " + dot_files_folder;
        std::system(script_cmd.c_str());
    }
}

template<StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
void SpaceSearcher<StateRepr, Strategy>::check_bisimulation_equivalence(const State<StateRepr> &state) const {
    if (!ArgumentParser::get_instance().get_verbose())
        return;


    bool are_bisimilar = true;
    State<StateRepr> temp = state;
    temp.contract_with_bisimulation();
    auto &os = ArgumentParser::get_instance().get_output_stream();

    // ReSharper disable once CppDFAConstantConditions
    if (temp == state) {
        // If the state is already bisimilar, no need to check further
        return;
    }
    // ReSharper disable once CppDFAUnreachableCode
    os << "[DEBUG] Checking bisimulation equivalence for possibly different states.";


    std::string fail_case;

    auto &domain_instance = Domain::get_instance();
    auto to_check1 = domain_instance.get_initial_description().get_initial_conditions();
    if (state.entails(to_check1) != temp.entails(to_check1)) {
        are_bisimilar = false;
        fail_case = "initial_conditions";
    }

    auto to_check2 = domain_instance.get_initial_description().get_ff_forS5();
    // ReSharper disable once CppDFAUnreachableCode
    // ReSharper disable once CppDFAUnreachableCode
    if (!to_check2.empty() && (state.entails(to_check2) != temp.entails(to_check2))) {
        are_bisimilar = false;
        fail_case = "ff_forS5";
    }

    auto to_check3 = domain_instance.get_goal_description();
    if (state.entails(to_check3) != temp.entails(to_check3)) {
        are_bisimilar = false;
        fail_case = "goal_description";
    }

    for (const auto &tmp_action: domain_instance.get_actions()) {
        for (auto condition: tmp_action.get_effects() | std::views::values) {
            if (state.entails(condition) != temp.entails(condition)) {
                are_bisimilar = false;
                fail_case = "action_effects of action " + tmp_action.get_name();
            }
        }
        auto to_check5 = tmp_action.get_executability();
        if (state.entails(to_check5) != temp.entails(to_check5)) {
            are_bisimilar = false;
            fail_case = "action_executability of action  " + tmp_action.get_name();
        }
        for (auto condition: tmp_action.get_fully_observants() | std::views::values) {
            if (state.entails(condition) != temp.entails(condition)) {
                are_bisimilar = false;
                fail_case = "Full Observability of action " + tmp_action.get_name();
            }
        }
        for (auto condition: tmp_action.get_partially_observants() | std::views::values) {
            if (state.entails(condition) != temp.entails(condition)) {
                are_bisimilar = false;
                fail_case = "Full Observability of action " + tmp_action.get_name();
            }
        }
    }

    if (!are_bisimilar) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::SearchBisimulationError,
            "Bisimulation reduction failed: there is some discrepancy in " + fail_case +
            ". Use debugger to investigate."
        );
    }
    os << " All good:)" << std::endl;
}
