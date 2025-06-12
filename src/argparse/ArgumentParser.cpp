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

ArgumentParser *ArgumentParser::instance = nullptr;

void ArgumentParser::create_instance(int argc, char **argv) {
    if (!instance) {
        instance = new ArgumentParser();
        instance->parse(argc, argv);
    }
}

ArgumentParser &ArgumentParser::get_instance() {
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

void ArgumentParser::parse(int argc, char **argv) {
    if (argc < 2) {
        print_usage();
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::ArgParseError,
            "No arguments provided. Please specify at least the input domain file." + std::string(
                ExitHandler::arg_parse_suggestion)
        );
    }

    try {
        app.parse(argc, argv);
        // After parsing, if log is enabled, generate the log file path using HelperPrint
        if (m_log_enabled) {
            m_log_file_path = HelperPrint::generate_log_file_path(m_input_file);
            m_log_ofstream.open(m_log_file_path);
            if (!m_log_ofstream.is_open()) {
                ExitHandler::exit_with_message(
                    ExitHandler::ExitCode::ArgParseError,
                    "Failed to open log file: " + m_log_file_path
                );
            }
            m_output_stream = &m_log_ofstream;
        } else {
            m_output_stream = &std::cout;
        }

        // --- Dataset mode consistency check ---
        if (!m_dataset_mode && (m_dataset_depth != 10 || m_dataset_mapped || m_dataset_both)) {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::ArgParseError,
                "Dataset-related options (--dataset_depth, --dataset_mapped, --dataset_both) "
                "were set but --dataset mode is not enabled. Please use --dataset to activate dataset mode."
            );
        }

        // --- Bisimulation consistency check ---
        if (!m_bisimulation && m_bisimulation_type != "FB") {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::ArgParseError,
                "Bisimulation type (--bis_type) was set but --bis is not enabled. Please use --bis to activate bisimulation."
            );
        }

        // --- Heuristic consistency check ---
        if (m_search_strategy != "HFS" && m_heuristic_opt != "SUBGOALS") {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::ArgParseError,
                "Heuristic option (--heuristic) is only valid with HFS search strategy (--search HFS)."
            );
        }

        // --- Execution plan checks and action loading ---
        if (m_exec_plan) {
            if (m_exec_actions.empty()) {
                m_exec_actions = HelperPrint::read_actions_from_file(m_plan_file);
                if (m_exec_actions.empty()) {
                    ExitHandler::exit_with_message(
                        ExitHandler::ExitCode::ArgParseError,
                        "No actions found in the specified plan file: " + m_plan_file
                    );
                }
            }
        }

        // --- Threads per search and portfolio threads informative message ---
        if (m_threads_per_search > 1 && m_portfolio_threads > 1) {
            get_output_stream() << "[INFO] Both multithreaded search and portfolio search are enabled. "
                    "Total threads used will be: "
                    << (m_threads_per_search * m_portfolio_threads)
                    << " (" << m_threads_per_search << " per search x "
                    << m_portfolio_threads << " portfolio threads)." << std::endl;
        }
    } catch (const CLI::CallForHelp &) {
        print_usage();
        std::exit(static_cast<int>(ExitHandler::ExitCode::SuccessNotPlanningMode));
    } catch (const CLI::ParseError &e) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::ArgParseError,
            std::string("Oops! There was a problem with your command line arguments. Details:\n  ") +
            e.what() + ExitHandler::arg_parse_suggestion.data()
        );
    } catch (const std::exception &e) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::ArgParseError,
            std::string("An unexpected error occurred while parsing arguments. Details:\n  ") +
            e.what() + ExitHandler::arg_parse_suggestion.data()
        );
    }
}

ArgumentParser::ArgumentParser() : app("deep") {
    app.add_option("input_file", m_input_file,
                   "Specify the input domain file (e.g., domain.epddl). This file defines the planning problem.")->
            required();

    app.add_flag("--debug", m_debug, "Enable verbose output for debugging the solving process.");

    app.add_flag("--bis", m_bisimulation,
                 "Activate e-states size reduction through bisimulation. Use this to reduce the state space by merging bisimilar states.");

    app.add_option("--bis_type", m_bisimulation_type,
                   "Specify the algorithm for bisimulation contraction (requires --bis). Options: 'FB' (Fast Bisimulation, default) or 'PT' (Paige and Tarjan).")
            ->check(CLI::IsMember({"FB", "PT"}))
            ->default_val("FB");

    app.add_flag("--check_visited", m_check_visited,
                 "Enable checking for previously visited states during planning to avoid redundant exploration.");

    // --- Multithreading options ---
    app.add_option("--threads_per_search", m_threads_per_search,
                   "Set the number of threads to use for each search strategy (default: 1). "
                   "If set > 1, each search strategy (e.g., BFS/HFS) will use this many threads.")
            ->default_val("1");

    app.add_option("--portfolio_threads", m_portfolio_threads,
                   "Set the number of portfolio threads (default: 1). "
                   "If set > 1, multiple planner configurations will run in parallel.")
            ->default_val("1");

    app.add_option("--search", m_search_strategy,
                   "Select the search strategy: 'BFS' (Best First Search, default), 'DFS' (Depth First Search), or 'HFS' (Heuristic First Search).")
            ->check(CLI::IsMember({"BFS", "DFS", "HFS"}))
            ->default_val("BFS");

    app.add_option("--heuristics", m_heuristic_opt,
                   "Specify the heuristic for HFS search: 'SUBGOALS' (default), 'L_PG', 'S_PG', 'C_PG', or 'GNN'. Only used if --search HFS is selected.")
            ->check(CLI::IsMember({"SUBGOALS", "L_PG", "S_PG", "C_PG", "GNN"}))
            ->default_val("SUBGOALS");

    // --- Dataset options ---
    app.add_flag("--dataset", m_dataset_mode, "Enable dataset generation mode for learning or analysis.");
    app.add_option("--dataset_depth", m_dataset_depth, "Set the maximum depth for dataset generation (default: 10).")
            ->default_val("10");
    app.add_flag("--dataset_mapped", m_dataset_mapped,
                 "Use mapped (compact) node labels in dataset generation. If not set, hashed node labels are used.");
    app.add_flag("--dataset_both", m_dataset_both, "Generate both mapped and hashed node labels in the dataset.");
    app.add_flag("--dataset_merged", m_dataset_merged,
                 "Enable merged dataset generation mode.");
    app.add_flag("--dataset_merged_both", m_dataset_merged_both,
                 "Enable both merged and non-merged dataset generation.");

    app.add_flag("--results_file", m_output_results_file,
                 "Log plan execution time and results to a file for scripting and comparisons.");

    app.add_flag("--execute_plan", m_exec_plan,
                 "Enable execution mode. If set, the planner will verify a plan instead of searching for one. "
                 "Actions to execute can be provided directly with --execute_actions, or will be read from the file specified by --plan_file (default: plan.txt). "
                 "When this option is enabled, all multithreading, search strategy, and heuristic flags are ignored; only plan verification is performed. "
                 "The plan file should contain a list of actions separated by spaces or commas. Minimal parsing is performed, so the file should be well formatted.");

    app.add_option("--execute_actions", m_exec_actions,
                   "Specify a sequence of actions to execute directly, bypassing planning. "
                   "Example: --execute_actions open_a peek_a. "
                   "If this option is set, the actions provided will be executed in order. "
                   "If not set, actions will be loaded from the plan file (see --plan_file).")
            ->expected(-1);

    app.add_option("--plan_file", m_plan_file,
                   "Specify the file from which to load the plan for execution (default: plan.txt). "
                   "Used only if --execute_plan is set and --execute_actions is not provided.")
            ->default_val("plan.txt");

    app.add_flag("--log", m_log_enabled,
                 "Enable logging to a file in the '" + std::string(OutputPaths::LOGS_FOLDER) +
                 "' folder. The log file will be named automatically. If this is not activated, std::cout will be used.");
}

ArgumentParser::~ArgumentParser() {
    if (m_log_ofstream.is_open()) {
        m_log_ofstream.close();
    }
}

// Getters
const std::string &ArgumentParser::get_input_file() const noexcept { return m_input_file; }

bool ArgumentParser::get_debug() const noexcept { return m_debug; }

bool ArgumentParser::get_check_visited() const noexcept { return m_check_visited; }

bool ArgumentParser::get_bisimulation() const noexcept { return m_bisimulation; }
const std::string &ArgumentParser::get_bisimulation_type() const noexcept { return m_bisimulation_type; }

bool ArgumentParser::get_dataset_mode() const noexcept { return m_dataset_mode; }
int ArgumentParser::get_dataset_depth() const noexcept { return m_dataset_depth; }
bool ArgumentParser::get_dataset_mapped() const noexcept { return m_dataset_mapped; }
bool ArgumentParser::get_dataset_both() const noexcept { return m_dataset_both; }
bool ArgumentParser::get_dataset_merged() const noexcept { return m_dataset_merged; }
bool ArgumentParser::get_dataset_merged_both() const noexcept { return m_dataset_merged_both; }

const std::string &ArgumentParser::get_heuristic() const noexcept { return m_heuristic_opt; }
const std::string &ArgumentParser::get_search_strategy() const noexcept { return m_search_strategy; }

bool ArgumentParser::get_execute_plan() const noexcept { return m_exec_plan; }
const std::string &ArgumentParser::get_plan_file() const noexcept { return m_plan_file; }

const std::vector<std::string> &ArgumentParser::get_execution_actions() noexcept {
    for (auto &action: m_exec_actions) {
        std::erase(action, ',');
    }
    return m_exec_actions;
}

bool ArgumentParser::get_results_file() const noexcept { return m_output_results_file; }
bool ArgumentParser::get_log_enabled() const noexcept { return m_log_enabled; }

std::ostream &ArgumentParser::get_output_stream() const { return *m_output_stream; }

int ArgumentParser::get_threads_per_search() const noexcept { return m_threads_per_search; }
int ArgumentParser::get_portfolio_threads() const noexcept { return m_portfolio_threads; }

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
    std::cout << "  " << prog_name << " domain.epddl --threads_per_search 4\n";
    std::cout << "    Run search with 4 threads per search strategy\n\n";
    std::cout << "  " << prog_name << " domain.epddl --portfolio_threads 3\n";
    std::cout << "    Run 3 planner configurations in parallel (portfolio search)\n\n";
    std::cout << "  " << prog_name << " domain.epddl --threads_per_search 2 --portfolio_threads 2\n";
    std::cout << "    Run 2 planner configurations in parallel, each using 2 threads (total 4 threads)\n\n";
}
