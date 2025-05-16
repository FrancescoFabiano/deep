/**
 * \brief Implementation of \ref ArgumentParser.h.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 1, 2019
 */

#include "ArgumentParser.h"
#include <iostream>
#include <stdexcept>

ArgumentParser* ArgumentParser::instance = nullptr;



void ArgumentParser::create_instance(int argc, char** argv) {
    if (!instance) {
        instance = new ArgumentParser();
        instance->parse(argc, argv);
    }
}

ArgumentParser& ArgumentParser::get_instance() {
    if (!instance) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::ArgParseInstanceError,
            "ArgumentParser instance not created. Call create_instance(argc, argv) first."
        );
    }
    return *instance;
}

void ArgumentParser::parse(int argc, char** argv) {

    if (argc < 2) {
        print_usage();
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::ArgParseError,
            "No arguments provided." + std::string(ExitHandler::arg_parse_suggestion)
        );
    }

    try {
        app.parse(argc, argv);
    } catch (const CLI::CallForHelp& e) {
        print_usage();
        std::exit(static_cast<int>(ExitHandler::ExitCode::Success));
    } catch (const CLI::ParseError& e) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::ArgParseError,
            std::string("Oops! There was a problem with your command line arguments. Below more info:\n  ") +
            e.what() + ExitHandler::arg_parse_suggestion.data()
        );
    } catch (const std::exception& e) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::ArgParseError,
            std::string("An unexpected error occurred. Below more info:\n  ") +
            e.what() + ExitHandler::arg_parse_suggestion.data()
        );
    }
}

ArgumentParser::ArgumentParser() : app("deep") {

    app.add_option("input_file", m_input_file, "Specify input domain file")->required();

    app.add_flag("--debug", m_debug, "Makes the solving process verbose");

    app.add_flag("--bis", m_bisimulation,
        "Activate e-states size reduction through bisimulation.");

    app.add_flag("--check_visited", m_check_visited, "Planner will check for visited states");

    app.add_option("--dataset-size", m_dataset_depth, "Maximum depth for dataset generation")
        ->default_val("10");

    app.add_option("--search", m_search_strategy,
        "Search strategy to use")
        ->check(CLI::IsMember({"BFS", "DFS", "I_DFS"}))
        ->default_val("BFS");

    app.add_option("--heuristic", m_heuristic_opt,
        "Heuristic to use")
        ->check(CLI::IsMember({"NONE", "L_PG", "S_PG", "C_PG", "GNN", "SUBGOALS"}))
        ->default_val("NONE");

    app.add_option("--parallel", m_parallel_type,
        "Run all heuristics in parallel. Overrides --heuristic and --search.")
        ->check(CLI::IsMember({"SERIAL", "PTHREAD", "FORK"}))
        ->default_val("PTHREAD");

    app.add_option("--parallel-wait", m_parallel_wait,
        "Decide whether to wait for all parallel threads/processes.")
        ->check(CLI::IsMember({"NONE", "WAIT"}))
        ->default_val("NONE");

    app.add_flag("--results_file", m_output_results_file,
        "Log plan execution time to a file for test comparisons");

    app.add_option("--execute-actions", m_exec_actions,
        "Perform sequence of actions instead of planning")->expected(-1);

    app.add_flag("--execute", m_exec_plan, "Execute the plan in the '--plan_file' file");
    app.add_option("--plan-file", m_plan_file, "Specify file for saving/loading plan")
        ->default_val("plan.txt");
}

// Getters
bool ArgumentParser::get_debug() const noexcept { return m_debug; }
bool ArgumentParser::get_bisimulation() const noexcept { return m_bisimulation; }
bool ArgumentParser::get_dataset_mode() const noexcept { return m_dataset_mode; }
int ArgumentParser::get_dataset_size() const noexcept { return m_dataset_depth; }
const std::string& ArgumentParser::get_heuristic() const noexcept { return m_heuristic_opt; }
const std::string& ArgumentParser::get_search_strategy() const noexcept { return m_search_strategy; }
bool ArgumentParser::get_execute_plan() const noexcept { return m_exec_plan; }
const std::string& ArgumentParser::get_plan_file() const noexcept { return m_plan_file; }
const std::string& ArgumentParser::get_input_file() const noexcept { return m_input_file; }
const std::vector<std::string>& ArgumentParser::get_execution_actions() const noexcept { return m_exec_actions; }
bool ArgumentParser::get_results_file() const noexcept { return m_output_results_file; }
const std::string& ArgumentParser::get_parallel_type() const noexcept { return m_parallel_type; }
const std::string& ArgumentParser::get_parallel_wait() const noexcept { return m_parallel_wait; }
bool ArgumentParser::get_check_visited() const noexcept { return m_check_visited; }

void ArgumentParser::print_usage() const {
    std::cout << app.help() << std::endl;
    std::string prog_name = "deep";
    std::cout << "\nEXAMPLES:\n";
    std::cout << "  " << prog_name << " domain.epddl\n";
    std::cout << "    Find a plan for domain.epddl\n\n";
    std::cout << "  " << prog_name << " domain.epddl --heuristic SUBGOALS\n";
    std::cout << "    Plan using heuristic 'SUBGOALS'\n\n";
    std::cout << "  " << prog_name << " domain.epddl --execute-actions open_a peek_a\n";
    std::cout << "    Execute actions [open_a, peek_a] step by step\n\n";
}