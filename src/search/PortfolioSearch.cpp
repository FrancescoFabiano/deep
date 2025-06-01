#include "PortfolioSearch.h"
#include <fstream>
#include <sstream>
#include <thread>
#include <atomic>
#include <functional>
#include <chrono>
#include "utilities/ExitHandler.h"
#include "search/SpaceSearcher.h"
#include "search_strategies/BestFirst.h"
#include "search_strategies/BreadthFirst.h"
#include "search_strategies/DepthFirst.h"
#include "states/representations/kripke/KripkeState.h"
#include "states/State.h"
#include "argparse/Configuration.h"

bool PortfolioSearch::run_portfolio_search( std::ostream & os, int user_threads, int threads_per_search) {
    using Clock = std::chrono::steady_clock;
    std::atomic<bool> found_goal{false};
    std::atomic<int> winner{-1};
    std::vector<std::thread> threads;
    std::vector<std::string> search_types;
    std::vector<std::chrono::duration<double>> times;
    std::vector<unsigned int> expanded_nodes;
    std::vector<std::string> config_snapshots;

    const int configs_to_run = std::min(user_threads, static_cast<int>(m_search_configurations.size()));
    times.resize(configs_to_run);
    expanded_nodes.resize(configs_to_run);
    search_types.resize(configs_to_run);
    config_snapshots.resize(configs_to_run);

    // Prepare the user configuration for the first thread
    const auto& user_config = Configuration::get_instance();

    // --- Measure initial state build time ---
    os << "Building initial state ...\n";
    auto initial_build_start = Clock::now();
    State<KripkeState> initial_state;
    initial_state.build_initial();
    const auto initial_build_end = Clock::now();
    auto initial_build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(initial_build_end - initial_build_start);
    os << "Initial state built in " << initial_build_duration.count() << " ms.\n";
    // --- End measure ---

    auto run_search = [&](int idx, const std::map<std::string, std::string>& config_map, bool is_user_config) {
        // Each thread gets its own Configuration instance
        if (!is_user_config) {
            Configuration::create_instance();
            auto& config = Configuration::get_instance();
            for (const auto& [key, value] : config_map) {
                config.set_field_by_name(key, value);
            }
        }
        const auto& config = Configuration::get_instance();
        const std::string search_type = config.get_search_strategy();

        // Select searcher
        std::string used_search_type;
        std::chrono::duration<double> elapsed{};
        unsigned int expanded = 0;
        bool result = false;

        if (search_type == "BFS") {
            SpaceSearcher<KripkeState, BreadthFirst<State<KripkeState>>> searcherBFS{BreadthFirst<State<KripkeState>>()};
            result = searcherBFS.search(initial_state, threads_per_search);
            used_search_type = searcherBFS.get_search_type();
            elapsed = searcherBFS.get_elapsed_seconds();
            expanded = searcherBFS.get_expanded_nodes();
        } else if (search_type == "HFS") {
            SpaceSearcher<KripkeState, BestFirst<State<KripkeState>>> searcherHFS{BestFirst<State<KripkeState>>()};
            result = searcherHFS.search(initial_state, threads_per_search);
            used_search_type = searcherHFS.get_search_type();
            elapsed = searcherHFS.get_elapsed_seconds();
            expanded = searcherHFS.get_expanded_nodes();
        } else if (search_type == "DFS") {
            SpaceSearcher<KripkeState, DepthFirst<State<KripkeState>>> searcherDFS{DepthFirst<State<KripkeState>>()};
            result = searcherDFS.search(initial_state, threads_per_search);
            used_search_type = searcherDFS.get_search_type();
            elapsed = searcherDFS.get_elapsed_seconds();
            expanded = searcherDFS.get_expanded_nodes();
        } else {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::PortfolioConfigError,
                "Unknown search type: " + search_type
            );
        }
        auto end = Clock::now();

        times[idx] = elapsed;
        expanded_nodes[idx] = expanded;
        search_types[idx] = used_search_type;
        std::ostringstream oss;
        config.print(oss);
        config_snapshots[idx] = oss.str();

        if (result && !found_goal.exchange(true)) {
            winner = idx;
        }
    };

    // Launch threads
    for (int i = 0; i < configs_to_run; ++i) {
        bool is_user_config = (i == 0);
        const auto& config_map = is_user_config ? std::map<std::string, std::string>() : m_search_configurations[i];
        threads.emplace_back(run_search, i, config_map, is_user_config);
    }

    // Wait for threads to finish or a goal to be found
    for (auto& th : threads) {
        if (th.joinable()) th.join();
    }

    if (found_goal) {
        int idx = winner;
        os << "\nGoal found :) using " << search_types[idx]
           << " in " << times[idx].count() << "s"
           << " expanding " << expanded_nodes[idx] << " nodes.\n";
        os << "Configuration used:\n" << config_snapshots[idx] << std::endl;
        return true;
    } else {
        os << "\nNo goal found :(" << std::endl;
        return false;
    }
}

void PortfolioSearch::parse_configurations_from_file(const std::string& file_path) {
    m_search_configurations.clear();
    std::ifstream infile(file_path);
    if (!infile) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::PortfolioConfigFileError,
            "[PortfolioSearch] Could not open configuration file: " + file_path
        );
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
    m_search_configurations.push_back({{"search", "HFS"}, {"heuristic", "SUBGOALS"}});
    m_search_configurations.push_back({{"search", "HFS"}, {"heuristic", "L_PG"}});
    m_search_configurations.push_back({{"search", "HFS"}, {"heuristic", "S_PG"}});
    m_search_configurations.push_back({{"search", "HFS"}, {"heuristic", "C_PG"}});
    m_search_configurations.push_back({{"search", "HFS"}, {"heuristic", "GNN"}});
    m_search_configurations.push_back({{"strategy", "DFS"}});
}
