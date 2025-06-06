#include "PortfolioSearch.h"
#include <fstream>
#include <sstream>
#include <thread>
#include <atomic>
#include <functional>
#include <chrono>
#include <mutex>

#include "HelperPrint.h"
#include "utilities/ExitHandler.h"
#include "search/SpaceSearcher.h"
#include "search_strategies/BestFirst.h"
#include "search_strategies/BreadthFirst.h"
#include "search_strategies/DepthFirst.h"
#include "states/representations/kripke/KripkeState.h"
#include "states/State.h"
#include "argparse/Configuration.h"


// I am adding this to be seen by the linker because it is a static templated singleton
template <>
TrainingDataset<KripkeState>* TrainingDataset<KripkeState>::instance = nullptr;

PortfolioSearch::PortfolioSearch() { set_default_configurations(); }


bool PortfolioSearch::run_portfolio_search() const
{
    const auto portfolio_threads = ArgumentParser::get_instance().get_portfolio_threads();
    using Clock = std::chrono::steady_clock;
    std::atomic<bool> found_goal{false};
    std::atomic<int> winner{-1};
    std::vector<std::thread> threads;
    std::vector<std::string> search_types;
    std::vector<std::chrono::duration<double>> times;
    std::vector<unsigned int> expanded_nodes;
    std::vector<std::string> config_snapshots;
    std::mutex result_mutex;

    const int configs_to_run = std::min(portfolio_threads, static_cast<int>(m_search_configurations.size()));
    times.resize(configs_to_run);
    expanded_nodes.resize(configs_to_run);
    search_types.resize(configs_to_run);
    config_snapshots.resize(configs_to_run);

    auto& os = ArgumentParser::get_instance().get_output_stream();

    // --- Measure initial state build time ---
    os << "\nBuilding initial state ...\n";
    const auto initial_build_start = Clock::now();
    State<KripkeState> initial_state;
    initial_state.build_initial();
    const auto initial_build_end = Clock::now();
    const auto initial_build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        initial_build_end - initial_build_start);
    os << "Initial state built in " << initial_build_duration.count() << " ms.\n";
    // --- End measure ---

    std::vector<ActionIdsList> plan_actions_id(configs_to_run);

    auto run_search = [&](int idx, const std::map<std::string, std::string>& config_map, const bool is_user_config)
    {
        // --- DEBUG: Print reached at start of thread ---
        if (ArgumentParser::get_instance().get_debug())
        {
            static std::mutex cout_mutex;
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "[Thread " << idx << "] Entered run_search lambda." << std::endl;
        }

        if (found_goal) return; // Early exit if another thread found the goal

        // Each thread gets its own Configuration instance
        if (!is_user_config)
        {
            Configuration::create_instance();
            auto& config = Configuration::get_instance();
            for (const auto& [key, value] : config_map)
            {
                config.set_field_by_name(key, value);
            }
        }
        const auto& config = Configuration::get_instance();
        const SearchType search_type = config.get_search_strategy();

        std::string search_type_name;
        std::chrono::duration<double> elapsed{};
        unsigned int expanded = 0;
        bool result = false;
        ActionIdsList actions_id;

        switch (search_type)
        {
        case SearchType::BFS:
            {
                SpaceSearcher<KripkeState, BreadthFirst<KripkeState>> searcherBFS{
                    BreadthFirst<KripkeState>(initial_state)
                };
                result = searcherBFS.search(initial_state);
                actions_id = searcherBFS.get_plan_actions_id();
                search_type_name = searcherBFS.get_search_type();
                elapsed = searcherBFS.get_elapsed_seconds();
                expanded = searcherBFS.get_expanded_nodes();
                break;
            }
        case SearchType::DFS:
            {
                SpaceSearcher<KripkeState, DepthFirst<KripkeState>> searcherDFS{
                    DepthFirst<KripkeState>(initial_state)
                };
                result = searcherDFS.search(initial_state);
                actions_id = searcherDFS.get_plan_actions_id();
                search_type_name = searcherDFS.get_search_type();
                elapsed = searcherDFS.get_elapsed_seconds();
                expanded = searcherDFS.get_expanded_nodes();
                break;
            }
        case SearchType::HFS:
            {
                SpaceSearcher<KripkeState, BestFirst<KripkeState>> searcherHFS{BestFirst<KripkeState>(initial_state)};
                result = searcherHFS.search(initial_state);
                actions_id = searcherHFS.get_plan_actions_id();
                search_type_name = searcherHFS.get_search_type();
                elapsed = searcherHFS.get_elapsed_seconds();
                expanded = searcherHFS.get_expanded_nodes();
                break;
            }
        default:
            if (ArgumentParser::get_instance().get_debug())
            {
                static std::mutex cout_mutex;
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "[Thread " << idx << "] Unknown search type!" << std::endl;
            }
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::PortfolioConfigError,
                "Unknown search type"
            );
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

            if (result && !found_goal.exchange(true))
            {
                winner = idx;
            }
        }
    };

    // Launch threads
    for (int i = 0; i < configs_to_run; ++i)
    {
        bool is_user_config = (i == 0);
        const auto& config_map = is_user_config ? std::map<std::string, std::string>() : m_search_configurations[i];
        threads.emplace_back(run_search, i, config_map, is_user_config);
    }

    // Wait for threads to finish or a goal to be found
    for (auto& th : threads)
    {
        if (th.joinable()) th.join();
    }

    if (found_goal)
    {
        int idx = winner;
        os << "\nGoal found :)";
        os << "\n  Problem filename: " << Domain::get_instance().get_name();
        os << "\n  Action Executed: ";
        HelperPrint::get_instance().print_list(plan_actions_id[idx]);
        os << "\n  Plan length: " << plan_actions_id[idx].size()
            << "\n  Search used: " << search_types[idx]
            << "\n  Nodes Expanded: " << expanded_nodes[idx]
            << "\n  Time elapsed " << std::chrono::duration_cast<std::chrono::milliseconds>(times[idx]).count() <<
            " ms";
        if (std::chrono::duration_cast<std::chrono::milliseconds>(times[idx]).count() > 1000)
        {
            os << " (" << HelperPrint::pretty_print_duration(times[idx]) << ")";
        }
        os << std::endl<< std::endl;
        return true;
    }
    else
    {
        os << "\nNo goal found :(" << std::endl<< std::endl;
        return false;
    }
}

void PortfolioSearch::parse_configurations_from_file(const std::string& file_path)
{
    m_search_configurations.clear();
    std::ifstream infile(file_path);
    if (!infile)
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::PortfolioConfigFileError,
            "[PortfolioSearch] Could not open configuration file: " + file_path
        );
        // No return needed, exit_with_message will terminate.
    }
    std::string line;
    while (std::getline(infile, line))
    {
        std::map<std::string, std::string> config;
        std::istringstream iss(line);
        std::string token;
        while (std::getline(iss, token, ','))
        {
            if (auto pos = token.find('='); pos != std::string::npos)
            {
                std::string key = token.substr(0, pos);
                std::string value = token.substr(pos + 1);
                config[key] = value;
            }
        }
        if (!config.empty())
        {
            m_search_configurations.push_back(config);
        }
    }
}

void PortfolioSearch::set_default_configurations()
{
    m_search_configurations.clear();
    // Whatever is not set here will is kept from the user input.
    m_search_configurations.push_back({{"search", "BFS"}});
    m_search_configurations.push_back({{"search", "HFS"}, {"heuristic", "SUBGOALS"}});
    m_search_configurations.push_back({{"search", "HFS"}, {"heuristic", "L_PG"}});
    m_search_configurations.push_back({{"search", "HFS"}, {"heuristic", "S_PG"}});
    m_search_configurations.push_back({{"search", "HFS"}, {"heuristic", "C_PG"}});
    m_search_configurations.push_back({{"search", "HFS"}, {"heuristic", "GNN"}});
    m_search_configurations.push_back({{"search", "DFS"}});
}
