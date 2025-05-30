/**
 * \brief Implementation of \ref ArgumentParser.h.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 12, 2025
 */

#include "ArgumentParser.h"
#include "HelperPrint.h"
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
    if (instance == nullptr) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::ArgParseInstanceError,
            "ArgumentParser instance not created. Call create_instance(argc, argv) first."
        );
        //Jut To please the compiler
        exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
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
        // After parsing, if log is enabled, generate the log file path using HelperPrint
        if (m_log_enabled) {
            m_log_file_path = HelperPrint::generate_log_file_path(m_input_file);
        }
    } catch (const CLI::CallForHelp&) {
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

    app.add_option("--bis_type", m_search_strategy,
    "Sets the algorithm to use for bisimulation contraction (does not activate bisimulation)."
    "  Possible values are: 'FB' (Fast Bisimulation) or 'PT' (Paige and Tarjan).")
    ->check(CLI::IsMember({"FB", "PT"}))
    ->default_val("FB");



    app.add_flag("--check_visited", m_check_visited, "Planner will check for visited states");

    app.add_option("--dataset-size", m_dataset_depth, "Maximum depth for dataset generation")
        ->default_val("10");

    app.add_option("--search", m_search_strategy,
        "Search strategy to use")
        ->check(CLI::IsMember({"BFS", "DFS", "HFS"}))
        ->default_val("BFS");

    app.add_option("--heuristic", m_heuristic_opt,
        "Heuristic to use")
        ->check(CLI::IsMember({"SUBGOALS", "L_PG", "S_PG", "C_PG", "GNN"}))
        ->default_val("SUBGOALS");

    app.add_flag("--results_file", m_output_results_file,
        "Log plan execution time to a file for test comparisons");

    app.add_option("--execute-actions", m_exec_actions,
        "Perform sequence of actions instead of planning")->expected(-1);

    app.add_flag("--execute", m_exec_plan, "Execute the plan in the '--plan_file' file");
    app.add_option("--plan-file", m_plan_file, "Specify file for saving/loading plan")
        ->default_val("plan.txt");

    app.add_flag("--log", m_log_enabled, "Enable logging to a file in the log folder");
}

// Getters
bool ArgumentParser::get_debug() const noexcept { return m_debug; }
bool ArgumentParser::get_bisimulation() const noexcept { return m_bisimulation; }
const std::string& ArgumentParser::get_bisimulation_type() const noexcept { return m_bisimulation_type; }
bool ArgumentParser::get_dataset_mode() const noexcept { return m_dataset_mode; }
int ArgumentParser::get_dataset_size() const noexcept { return m_dataset_depth; }
const std::string& ArgumentParser::get_heuristic() const noexcept { return m_heuristic_opt; }
const std::string& ArgumentParser::get_search_strategy() const noexcept { return m_search_strategy; }
bool ArgumentParser::get_execute_plan() const noexcept { return m_exec_plan; }
const std::string& ArgumentParser::get_plan_file() const noexcept { return m_plan_file; }
const std::string& ArgumentParser::get_input_file() const noexcept { return m_input_file; }
const std::vector<std::string>& ArgumentParser::get_execution_actions() const noexcept { return m_exec_actions; }
bool ArgumentParser::get_results_file() const noexcept { return m_output_results_file; }
bool ArgumentParser::get_check_visited() const noexcept { return m_check_visited; }
bool ArgumentParser::get_log_enabled() const noexcept { return m_log_enabled; }
const std::string& ArgumentParser::get_log_file_path() const noexcept { return m_log_file_path; }

void ArgumentParser::print_usage() const {
    std::cout << app.help() << std::endl;
    const std::string prog_name = "deep";
    std::cout << "\nEXAMPLES:\n";
    std::cout << "  " << prog_name << " domain.epddl\n";
    std::cout << "    Find a plan for domain.epddl\n\n";
    std::cout << "  " << prog_name << " domain.epddl --heuristic SUBGOALS\n";
    std::cout << "    Plan using heuristic 'SUBGOALS'\n\n";
    std::cout << "  " << prog_name << " domain.epddl --execute-actions open_a peek_a\n";
    std::cout << "    Execute actions [open_a, peek_a] step by step\n\n";
}
