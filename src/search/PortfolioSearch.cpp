#include "PortfolioSearch.h"
#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <thread>

#include "HelperPrint.h"
#include "argparse/Configuration.h"
#include "neuralnets/TrainingDataset.h"
#include "search/SpaceSearcher.h"
#include "search_strategies/best_first/HeuristicFirst.h"
#include "search_strategies/BreadthFirst.h"
#include "search_strategies/DepthFirst.h"
#include "search_strategies/IterativeDepthFirst.h"
#include "search_strategies/best_first/Astar.h"
#include "states/State.h"
#include "states/representations/kripke/KripkeState.h"
#include "utilities/ExitHandler.h"

// I am adding this to be seen by the linker because it is a static templated
// singleton
template <>
TrainingDataset<KripkeState> *TrainingDataset<KripkeState>::instance = nullptr;

PortfolioSearch::PortfolioSearch() {
  if (const auto config_file = ArgumentParser::get_instance().get_config_file();
      config_file.empty()) {
    set_default_configurations();
  } else {
    parse_configurations_from_file(config_file);
  }
}

bool PortfolioSearch::run_portfolio_search() const {
  const auto portfolio_threads =
      ArgumentParser::get_instance().get_portfolio_threads();
  using Clock = std::chrono::steady_clock;
  std::atomic<bool> found_goal{false};
  std::atomic<int> winner{-1};
  std::vector<std::thread> threads;
  std::vector<std::string> search_types;
  std::vector<std::chrono::duration<double>> times;
  std::vector<unsigned int> expanded_nodes;
  std::vector<std::string> config_snapshots;
  std::mutex result_mutex;

  const int configs_to_run = std::min(
      portfolio_threads, static_cast<int>(m_search_configurations.size()));
  if (configs_to_run < portfolio_threads) {
    ArgumentParser::get_instance().get_output_stream()
        << "[WARNING] Portfolio threads (" << portfolio_threads
        << ") exceed available configurations ("
        << m_search_configurations.size() << "). " << "Running only "
        << configs_to_run << " configurations." << std::endl;
  }
  times.resize(configs_to_run);
  expanded_nodes.resize(configs_to_run);
  search_types.resize(configs_to_run);
  config_snapshots.resize(configs_to_run);

  auto &os = ArgumentParser::get_instance().get_output_stream();

  // --- Measure initial state build time ---
  if (ArgumentParser::get_instance().get_verbose()) {
    os << "\nBuilding initial state ...\n";
  }
  const auto initial_build_start = Clock::now();
  State<KripkeState> initial_state;
  initial_state.build_initial();
  const auto initial_build_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          Clock::now() - initial_build_start);
  if (ArgumentParser::get_instance().get_verbose()) {
    os << "Initial state built (in " << initial_build_duration.count()
       << " ms).\n";
  }
  // --- End measure ---

  std::vector<ActionIdsList> plan_actions_id(configs_to_run);

  auto run_search = [&](int idx,
                        const std::map<std::string, std::string> &config_map,
                        const bool is_user_config) {
    if (found_goal)
      return; // Early exit if another thread found the goal

    // Each thread gets its own Configuration instance
    if (!is_user_config) {
      Configuration::create_instance();
      auto &config = Configuration::get_instance();
      for (const auto &[key, value] : config_map) {
        config.set_field_by_name(key, value);
      }
    }
    const auto &config = Configuration::get_instance();
    const SearchType search_type = config.get_search_strategy();

    std::string search_type_name;
    std::chrono::duration<double> elapsed{};
    unsigned int expanded = 0;
    bool result = false;
    ActionIdsList actions_id;

    switch (search_type) {
    case SearchType::BFS: {
      SpaceSearcher<KripkeState, BreadthFirst<KripkeState>> searcherBFS{
          BreadthFirst<KripkeState>(initial_state), found_goal};
      result = searcherBFS.search(initial_state);
      actions_id = searcherBFS.get_plan_actions_id();
      search_type_name = searcherBFS.get_search_type();
      elapsed = searcherBFS.get_elapsed_seconds();
      expanded = searcherBFS.get_expanded_nodes();
      break;
    }
    case SearchType::DFS: {
      SpaceSearcher<KripkeState, DepthFirst<KripkeState>> searcherDFS{
          DepthFirst<KripkeState>(initial_state), found_goal};
      result = searcherDFS.search(initial_state);
      actions_id = searcherDFS.get_plan_actions_id();
      search_type_name = searcherDFS.get_search_type();
      elapsed = searcherDFS.get_elapsed_seconds();
      expanded = searcherDFS.get_expanded_nodes();
      break;
    }
    case SearchType::IDFS: {
      SpaceSearcher<KripkeState, IterativeDepthFirst<KripkeState>> searcherIDFS{
          IterativeDepthFirst<KripkeState>(initial_state), found_goal};
      result = searcherIDFS.search(initial_state);
      actions_id = searcherIDFS.get_plan_actions_id();
      search_type_name = searcherIDFS.get_search_type();
      elapsed = searcherIDFS.get_elapsed_seconds();
      expanded = searcherIDFS.get_expanded_nodes();
      break;
    }
    case SearchType::HFS: {
      SpaceSearcher<KripkeState, HeuristicFirst<KripkeState>> searcherHFS{
          HeuristicFirst<KripkeState>(initial_state), found_goal};
      result = searcherHFS.search(initial_state);
      actions_id = searcherHFS.get_plan_actions_id();
      search_type_name = searcherHFS.get_search_type();
      elapsed = searcherHFS.get_elapsed_seconds();
      expanded = searcherHFS.get_expanded_nodes();
      break;
    }
        case SearchType::Astar: {
      SpaceSearcher<KripkeState, Astar<KripkeState>> searcherHFS{
        Astar<KripkeState>(initial_state), found_goal};
      result = searcherHFS.search(initial_state);
      actions_id = searcherHFS.get_plan_actions_id();
      search_type_name = searcherHFS.get_search_type();
      elapsed = searcherHFS.get_elapsed_seconds();
      expanded = searcherHFS.get_expanded_nodes();
      break;
      }
    default:
      if (ArgumentParser::get_instance().get_verbose()) {
        static std::mutex cout_mutex;
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "[Thread " << idx << "] Unknown search type!" << std::endl;
      }
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::PortfolioConfigError, "Unknown search type");
      break;
    }

    {
      std::lock_guard<std::mutex> lock(result_mutex);
      times[idx] = elapsed;
      expanded_nodes[idx] = expanded;
      search_types[idx] = search_type_name;
      plan_actions_id[idx] = actions_id;
      std::ostringstream oss;
      config.print(oss);
      config_snapshots[idx] = oss.str();

      if (result && !found_goal.exchange(true)) {
        winner = idx;
      }
    }
  };

  // Launch threads
  for (int i = 0; i < configs_to_run; ++i) {
    bool is_user_config = (portfolio_threads == 1);
    const auto &config_map = is_user_config
                                 ? std::map<std::string, std::string>()
                                 : m_search_configurations[i];
    threads.emplace_back(run_search, i, config_map, is_user_config);
  }

  // Wait for threads to finish or a goal to be found
  for (auto &th : threads) {
    if (th.joinable())
      th.join();
  }

  if (found_goal) {
    const auto total_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - initial_build_start);
    int idx = winner;
    os << "\nGoal found :)";
    os << "\n  Problem filename: " << Domain::get_instance().get_name();
    os << "\n  Action executed: ";
    HelperPrint::get_instance().print_list(plan_actions_id[idx]);
    os << "\n  Plan length: " << plan_actions_id[idx].size()
       << "\n  Search used: " << search_types[idx]
       << "\n  Nodes expanded: " << expanded_nodes[idx];
    HelperPrint::print_time("Total execution time", total_duration);
    HelperPrint::print_time(
        "  Initial state construction (including parsing and domain setup)",
        initial_build_duration);
    HelperPrint::print_time("  Search time", times[idx]);
    HelperPrint::print_time("  Thread management overhead",
                            total_duration - initial_build_duration -
                                times[idx]);

    if (ArgumentParser::get_instance().get_results_info()) {
      os << "\n" << config_snapshots[idx];
    }
    os << std::endl << std::endl;
    return true;
  } else {
    os << "\nNo goal found :(" << std::endl << std::endl;
    return false;
  }
}

void PortfolioSearch::parse_configurations_from_file(
    const std::string &file_path) {
  m_search_configurations.clear();
  std::ifstream infile(file_path);
  if (!infile) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::PortfolioConfigFileError,
        "[PortfolioSearch] Could not open configuration file: " + file_path);
    // No return needed, exit_with_message will terminate.
  }
  std::string line;
  while (std::getline(infile, line)) {
    std::map<std::string, std::string> config;
    std::istringstream iss(line);
    std::string token;
    while (std::getline(iss, token, ',')) {
      if (auto pos = token.find('='); pos != std::string::npos) {
        std::string key = token.substr(0, pos);
        std::string value = token.substr(pos + 1);
        config[key] = value;
      }
    }
    if (!config.empty()) {
      m_search_configurations.push_back(config);
    }
  }
}

void PortfolioSearch::set_default_configurations() {
  m_search_configurations.clear();
  // Whatever is not set here will is kept from the user input.
  m_search_configurations.push_back({{"search", "BFS"}});
  m_search_configurations.push_back(
      {{"search", "HFS"}, {"heuristics", "SUBGOALS"}});
  m_search_configurations.push_back(
      {{"search", "HFS"}, {"heuristics", "L_PG"}});
  m_search_configurations.push_back(
      {{"search", "HFS"}, {"heuristics", "S_PG"}});
  m_search_configurations.push_back(
      {{"search", "HFS"}, {"heuristics", "C_PG"}});
  m_search_configurations.push_back({{"search", "Astar"}, {"heuristics", "GNN"}});
  m_search_configurations.push_back({{"search", "IDFS"}});
}
