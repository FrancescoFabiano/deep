/**
 * Implementation of \ref SpaceSearcher.h
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 29, 2025
 */

#include "argparse/ArgumentParser.h"
#include "search/SpaceSearcher.h"
#include "states/State.h"
#include "utilities/ExitHandler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <queue>
#include <ranges>
#include <set>
#include <string>
#include <thread>

#include "Configuration.h"
#include "FormulaHelper.h"
#include "HelperPrint.h"
#include "KripkeEqualityHelper.h"


template <StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
SpaceSearcher<StateRepr, Strategy>::SpaceSearcher(
    Strategy strategy,
    std::atomic<bool> &cancel_flag)
    : m_strategy(std::move(strategy)),
      m_cancel_flag(cancel_flag) {}


template <StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
std::string
SpaceSearcher<StateRepr, Strategy>::get_search_type() const noexcept {
  return m_strategy.get_name();
}


template <StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
unsigned int
SpaceSearcher<StateRepr, Strategy>::get_expanded_nodes() const noexcept {
  return m_expanded_nodes;
}


template <StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
std::chrono::duration<double>
SpaceSearcher<StateRepr, Strategy>::get_elapsed_seconds() const noexcept {
  return m_elapsed_seconds;
}


template <StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
const ActionIdsList &
SpaceSearcher<StateRepr, Strategy>::get_plan_actions_id() const noexcept {
  return m_plan_actions_id;
}


template <StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
bool SpaceSearcher<StateRepr, Strategy>::search(
    const State<StateRepr> &passed_initial) {

  m_expanded_nodes = 0;

  const bool check_visited =
      Configuration::get_instance()
          .get_check_visited();

  const auto &domain_instance =
      Domain::get_instance();

  const auto &actions =
      domain_instance.get_actions();

  if (actions.empty()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::SearchNoActions,
        "No actions available in the domain.");
  }

  const auto start_timing =
      std::chrono::system_clock::now();

  /*
   * Make a copy to avoid modifying the original state.
   */
  auto thread_safe_initial =
      passed_initial;

  /*
   * Always canonicalize the initial state when
   * bisimulation is enabled.
   *
   * Periodic contraction applies to successors.
   */
  if (Configuration::get_instance()
          .get_bisimulation()) {

    thread_safe_initial
        .contract_with_bisimulation();
  }

  if (thread_safe_initial.is_goal()) {

    m_elapsed_seconds =
        std::chrono::system_clock::now() -
        start_timing;

    return true;
  }

  bool result;

  if (ArgumentParser::get_instance()
          .get_execute_plan()) {

    result =
        validate_plan(
            thread_safe_initial,
            check_visited);

  } else {

    const int num_threads =
        ArgumentParser::get_instance()
            .get_threads_per_search();

    result =
        (num_threads <= 1)
            ? search_sequential(
                  thread_safe_initial,
                  actions,
                  check_visited)
            : search_parallel(
                  thread_safe_initial,
                  actions,
                  check_visited,
                  num_threads);
  }

  m_elapsed_seconds =
      std::chrono::system_clock::now() -
      start_timing;

  return result;
}


template <StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
bool SpaceSearcher<StateRepr, Strategy>::search_sequential(
    State<StateRepr> &initial,
    const ActionsSet &actions,
    const bool check_visited) {

  m_strategy.reset();

  const auto &configuration =
      Configuration::get_instance();

  const int RL_node_to_add =
      configuration.get_successors_to_analyze();

  const bool is_RL_search =
      configuration.get_search_strategy() ==
      SearchType::RL;

  std::vector<State<StateRepr>>
      fringe_RL;

    std::set<State<StateRepr>> visited_states;

  m_expanded_nodes = 0;

  m_strategy.push_initial(initial);

    if (check_visited) {
        visited_states.insert(initial);
    }

  while (!m_strategy.empty()) {

    if (m_cancel_flag.load()) {
      return false;
    }

    State current =
        m_strategy.peek();

    m_strategy.pop();

    ++m_expanded_nodes;


#ifdef DEBUG

    if (m_expanded_nodes % 250 == 0) {

      auto &os =
          ArgumentParser::get_instance()
              .get_output_stream();

      os << "[DEBUG] Expanded nodes: "
         << m_expanded_nodes
         << std::endl;
    }

#endif

    for (const auto &action : actions) {

      if (!current.is_executable(action)) {
        continue;
      }

      State successor =
          current.compute_successor(action);


      // ======================================================================
      // Periodic bisimulation
      // ======================================================================

      if (configuration.get_bisimulation()) {

        const std::size_t
            bisimulation_interval =
                configuration
                    .get_bisimulation_interval();

        const auto depth =
            successor.get_plan_length();

        const bool should_contract =
            bisimulation_interval > 0 &&
            depth > 0 &&
            depth % bisimulation_interval == 0;

        if (should_contract) {
          successor
              .contract_with_bisimulation();
        }
      }


      // ======================================================================
      // Goal test
      // ======================================================================

      if (successor.is_goal()) {

        m_plan_actions_id =
            successor.get_executed_actions();

        return true;
      }


#ifdef DEBUG

        /*
         * In Debug we keep the insertion result so that verification is
         * performed only when the successor was actually rejected as visited.
         */
        if (check_visited) {

            const auto [visited_it, inserted] =
                visited_states.insert(successor);

            if (inserted) {

                if (!is_RL_search) {
                    m_strategy.push(successor);
                }

                fringe_RL.push_back(successor);

            } else {

                /*
                 * This is a genuine visited-state hit.
                 *
                 * The set considers *visited_it and successor equivalent.
                 * Verify both strong structural equality and semantic
                 * equivalence with 50 epistemic formulae.
                 */
                if (!KripkeEqualityHelper::verify_equivalence(
                        visited_it->get_representation(),
                        successor.get_representation(),
                        true,
                        500,
                        5)) {

                    ExitHandler::exit_with_message(
                        ExitHandler::ExitCode::SearchMethodError,
                        "DEBUG: visited-state equivalence verification failed.");
                        }
            }

        } else {

            if (!is_RL_search) {
                m_strategy.push(successor);
            }

            fringe_RL.push_back(successor);
        }

#else

        /*
         * Release path stays minimal.
         */
        if (!check_visited ||
            visited_states.insert(successor).second) {

            if (!is_RL_search) {
                m_strategy.push(successor);
            }

            fringe_RL.push_back(successor);
            }

#endif

    }


    // ========================================================================
    // RL fringe handling
    // ========================================================================

    if (is_RL_search &&
        (fringe_RL.size() >=
             static_cast<size_t>(
                 RL_node_to_add) ||
         m_strategy.empty())) {

      m_strategy.push_vector(
          fringe_RL);

      fringe_RL.clear();
    }
  }

  return false;
}


template <StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
bool SpaceSearcher<StateRepr, Strategy>::search_parallel(
    State<StateRepr> &initial,
    const ActionsSet &actions,
    const bool check_visited,
    const int num_threads) {

  (void)initial;
  (void)actions;
  (void)check_visited;
  (void)num_threads;

  ExitHandler::exit_with_message(
      ExitHandler::ExitCode::SearchParallelNotImplemented,
      "Parallel search is not implemented yet. "
      "Please use sequential search.");

  std::exit(
      static_cast<int>(
          ExitHandler::ExitCode::ExitForCompiler));
}


template <StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
bool SpaceSearcher<StateRepr, Strategy>::validate_plan(
    const State<StateRepr> &initial,
    const bool check_visited) {

  /*
   * IMPORTANT:
   *
   * This visited set is intentionally NOT bounded.
   *
   * During explicit plan validation it is used to detect
   * repeated states in the supplied plan.
   */
  std::set<State<StateRepr>>
      visited_states;

  if (check_visited) {
    visited_states.insert(initial);
  }

  const std::string dot_files_folder =
      std::string(
          OutputPaths::EXEC_PLAN_FOLDER) +
      "/" +
      Domain::get_instance().get_name() +
      "/";

  std::filesystem::create_directories(
      dot_files_folder);

  State<StateRepr> current =
      initial;

  if (Configuration::get_instance()
          .get_bisimulation()) {

    current.contract_with_bisimulation();
  }

  print_dot_for_execute_plan(
      true,
      false,
      "initial",
      current,
      dot_files_folder);

  const auto &plan =
      ArgumentParser::get_instance()
          .get_execution_actions();

  for (auto it = plan.begin();
       it != plan.end();
       ++it) {

    const auto &action_name =
        *it;

    bool is_last =
        (std::next(it) == plan.end());

    bool found_action = false;

    for (const auto &action :
         Domain::get_instance()
             .get_actions()) {

      if (action.get_name() !=
          action_name) {
        continue;
      }

      found_action = true;

      if (!current.is_executable(action)) {

        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::
                StateActionNotExecutableError,
            std::string("The action \"") +
                action.get_name() +
                "\" was not executable while "
                "validating the plan.");

        return false;
      }

      ++m_expanded_nodes;

      current =
          current.compute_successor(
              action);

      /*
       * Keep plan-validation behavior unchanged.
       */
      if (Configuration::get_instance()
              .get_bisimulation()) {

        current
            .contract_with_bisimulation();
      }

      print_dot_for_execute_plan(
          false,
          is_last,
          action_name,
          current,
          dot_files_folder);

      if (current.is_goal()) {

        m_plan_actions_id =
            current.get_executed_actions();

        if (!is_last) {

          auto &os =
              ArgumentParser::get_instance()
                  .get_output_stream();

          os << "\n[WARNING] Plan found before "
                "the entire plan was used."
             << std::endl;
        }

        return true;
      }

      if (check_visited &&
          !visited_states
               .insert(current)
               .second) {

        auto &os =
            ArgumentParser::get_instance()
                .get_output_stream();

        os << "\n[WARNING] While executing the plan, "
              "found an already visited state after "
              "the execution of the actions:\n";

        HelperPrint::get_instance()
            .print_list(
                current
                    .get_executed_actions());

        os << "\nThis means that the plan is not "
              "optimal."
           << std::endl;
      }

      if (is_last) {

        auto &os =
            ArgumentParser::get_instance()
                .get_output_stream();

        os << "\n[WARNING] No plan found after "
              "the execution of:\n";

        HelperPrint::get_instance()
            .print_list(
                current
                    .get_executed_actions());

        os << std::endl;
      }

      break;
    }

    if (!found_action) {

      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::
              ActionTypeConflict,
          std::string("Action \"") +
              action_name +
              "\" not found in domain actions "
              "while validating the plan.");

      return false;
    }
  }

  return current.is_goal();
}


template <StateRepresentation StateRepr, SearchStrategy<StateRepr> Strategy>
void SpaceSearcher<StateRepr, Strategy>::print_dot_for_execute_plan(
    const bool initial,
    const bool last,
    const std::string &action_name,
    const State<StateRepr> &current,
    const std::string &dot_files_folder) {

  if (!ArgumentParser::get_instance()
           .get_verbose()) {
    return;
  }

  const std::string dot_extension =
      ".dot";

  std::size_t dot_count =
      std::count_if(
          std::filesystem::directory_iterator(
              dot_files_folder),
          std::filesystem::directory_iterator{},
          [dot_extension](
              const auto &entry) {
            return entry.path().extension() ==
                   dot_extension;
          });

  std::string print_name =
      dot_files_folder;

  std::ostringstream oss;

  oss << std::setw(5)
      << std::setfill('0')
      << static_cast<int>(
             dot_count);

  print_name +=
      oss.str() + "-";

  if (!initial) {
    print_name += action_name;
  } else {
    print_name += "initial";
  }

  const std::string adj_dot_extension =
      Configuration::get_instance()
              .get_bisimulation()
          ? ("-bis" + dot_extension)
          : dot_extension;

  std::string ofstream_name =
      print_name +
      adj_dot_extension;

  if (std::ofstream ofs(ofstream_name);
      ofs.is_open()) {

    current.print_dot_format(ofs);
  }

  if (last) {

    std::string script_cmd =
        "./scripts/dot_to_png.sh " +
        dot_files_folder;

    if (std::system(
            script_cmd.c_str()) != 0) {

      auto &os =
          ArgumentParser::get_instance()
              .get_output_stream();

      os << "[WARNING] dot to png conversion "
            "failed for folder: "
         << dot_files_folder
         << std::endl;
    }
  }
}