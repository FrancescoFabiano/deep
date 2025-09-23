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
    ExitHandler::exit_with_message(ExitHandler::ExitCode::ArgParseInstanceError,
                                   "ArgumentParser instance not created. Call "
                                   "create_instance(argc, argv) first.");
    // Jut To please the compiler
    exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
  }
  return *instance;
}

void ArgumentParser::parse(int argc, char **argv) {
  if (argc < 2) {
    print_usage();
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::ArgParseError,
        "No arguments provided. Please specify at least the input domain "
        "file." +
            std::string(ExitHandler::arg_parse_suggestion));
  }

  try {
    app.parse(argc, argv);
    // After parsing, if log is enabled, generate the log file path using
    // HelperPrint
    if (m_log_enabled) {
      m_log_file_path = HelperPrint::generate_log_file_path(m_input_file);
      m_log_ofstream.open(m_log_file_path);
      if (!m_log_ofstream.is_open()) {
        ExitHandler::exit_with_message(ExitHandler::ExitCode::ArgParseError,
                                       "Failed to open log file: " +
                                           m_log_file_path);
      }
      m_output_stream = &m_log_ofstream;
    } else {
      m_output_stream = &std::cout;
    }

    // --- Dataset mode consistency check ---
    if (!m_dataset_mode &&
        (app.count("--dataset_depth") ||
         app.count("--dataset_discard_factor") ||
         app.count("--dataset_seed"))) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::ArgParseError,
          "Dataset-related options (--dataset_depth, --dataset_discard_factor, "
          "--dataset_seed) "
          "were set but --dataset mode is not enabled. Please use --dataset to "
          "activate dataset mode.");
    }

    if (app.count("--dataset_type")) {
      set_dataset_type();
    }

    // --- Bisimulation consistency check ---
    if (!m_bisimulation && app.count("--bisimulation_type")) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::ArgParseError,
          "Bisimulation type (--bisimulation_type) was set but --bisimulation "
          "is not enabled. Please use --bis to activate bisimulation.");
    }

    // --- Heuristic consistency check ---
    if (m_search_strategy != "HFS" && m_search_strategy != "Astar" &&
        app.count("--heuristics")) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::ArgParseError,
          "--heuristics can only be used with --search HFS or --search Astar.");
    }
    /*if ((m_search_strategy == "HFS" || m_search_strategy == "Astar") &&
        m_heuristic_opt != "GNN" && app.count("--GNN_model")) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::ArgParseError,
          "--GNN_model can only be used with "
          "--search HFS or --search Astar and --heuristics GNN.");
    }

    if (app.count("--GNN_constant_file") && m_heuristic_opt != "GNN") {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::ArgParseError,
          "--GNN_constant_file can only be used with --heuristics GNN.");
    }*/

    // --- Execution plan checks and action loading ---
    if (m_exec_plan) {
      if (m_exec_actions.empty()) {
        m_exec_actions = HelperPrint::read_actions_from_file(m_plan_file);
        if (m_exec_actions.empty()) {
          ExitHandler::exit_with_message(
              ExitHandler::ExitCode::ArgParseError,
              "No actions found in the specified plan file: " + m_plan_file);
        }
      }
    }

    // --- Threads per search and portfolio threads informative message ---
    if (m_threads_per_search > 1 && m_portfolio_threads > 1) {
      get_output_stream() << "[INFO] Both multithreaded search and portfolio "
                             "search are enabled. "
                             "Total threads used will be: "
                          << (m_threads_per_search * m_portfolio_threads)
                          << " (" << m_threads_per_search << " per search x "
                          << m_portfolio_threads << " portfolio threads)."
                          << std::endl;
    }
  } catch (const CLI::CallForHelp &) {
    print_usage();
    std::exit(static_cast<int>(ExitHandler::ExitCode::SuccessNotPlanningMode));
  } catch (const CLI::ParseError &e) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::ArgParseError,
        std::string("Oops! There was a problem with your command line "
                    "arguments. Details:\n  ") +
            e.what() + ExitHandler::arg_parse_suggestion.data());
  } catch (const std::exception &e) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::ArgParseError,
        std::string("An unexpected error occurred while parsing arguments. "
                    "Details:\n  ") +
            e.what() + ExitHandler::arg_parse_suggestion.data());
  }
}

ArgumentParser::ArgumentParser() : app("deep") {
  app.add_option("input_file", m_input_file,
                 "Specify the input problem file (e.g., problem.txt). This "
                 "file defines the planning problem.")
      ->required();

  // Debug/logging group
  auto *debug_group = app.add_option_group("Debug/Logging");
  debug_group->add_flag("-v,--verbose", m_verbose,
                        "Enable verbose solving process.");
  debug_group->add_flag(
      "-l,--log", m_log_enabled,
      "Enable logging to a file in the '" +
          std::string(OutputPaths::LOGS_FOLDER) +
          "' folder. The log file will be named automatically. If this is not "
          "activated, std::cout will be used.");
  debug_group->add_flag(
      "-r,--results_info", m_output_results_info,
      "Prints extra plan information for scripting and comparisons.");

  // Bisimulation group
  auto *bis_group = app.add_option_group("Bisimulation");
  bis_group->add_flag(
      "-b,--bisimulation", m_bisimulation,
      "Activate e-states size reduction through bisimulation. Use this to "
      "reduce the state space by merging bisimilar states.");
  bis_group
      ->add_option("--bisimulation_type", m_bisimulation_type,
                   "Specify the algorithm for bisimulation contraction "
                   "(requires --bis). Options: 'FB' (Fast Bisimulation, "
                   "default) or 'PT' (Paige and Tarjan).")
      ->check(CLI::IsMember({"FB", "PT"}))
      ->default_val("FB");

  // Dataset group
  auto *dataset_group = app.add_option_group("Dataset");
  dataset_group->add_flag(
      "-d,--dataset", m_dataset_mode,
      "Enable dataset generation mode for learning or analysis.");
  dataset_group
      ->add_option(
          "--dataset_depth", m_dataset_depth,
          "Set the maximum depth for dataset generation (default: 10).")
      ->default_val("10");
  dataset_group
      ->add_option(
          "--dataset_type", m_dataset_type_string,
          "Specifies how node labels are represented in dataset generation. "
          "Options: MAPPED (compact integer mapping), HASHED (standard "
          "hashing), "
          "or BITMASK (bitmask representation of fluents and goals).")
      ->check(CLI::IsMember({"MAPPED", "HASHED", "BITMASK"}))
      ->default_val("HASHED");
  dataset_group->add_flag("--dataset_separated", m_dataset_separated,
                          "Enable non-merged dataset generation mode.");
  dataset_group
      ->add_option("--dataset_discard_factor", m_dataset_discard_factor,
                   "Set the maximum value for discard factor during dataset "
                   "generation (Must be within 0 and 1, not included).")
      ->default_val("0.4");
  dataset_group
      ->add_option("--dataset_seed", m_dataset_seed,
                   "Set the seed used for value generation. "
                   "If no seed is provided, the default (42) is used. "
                   "If a negative value is given, a random seed will be "
                   "generated instead, "
                   "as negative seeds are not accepted.")
      ->default_val("42");

  // Search group
  auto *search_group = app.add_option_group("Search");
  search_group
      ->add_option("-s,--search", m_search_strategy,
                   "Select the search strategy: 'BFS' (Best First Search, "
                   "default), 'DFS' (Depth First Search), 'IDFS' (Iterative "
                   "Depth First Search), 'HFS' (Heuristic First Search), or "
                   "'Astar' (A* Search, uses heuristics with A* method).")
      ->check(CLI::IsMember({"BFS", "DFS", "IDFS", "HFS", "Astar"}))
      ->default_val("BFS");
  search_group->add_flag("-c,--check_visited", m_check_visited,
                         "Enable checking for previously visited states during "
                         "planning to avoid redundant exploration.");
  search_group
      ->add_option(
          "-u,--heuristics", m_heuristic_opt,
          "Specify the heuristic for HFS or Astar search: 'SUBGOALS' "
          "(default), 'L_PG', "
          "'S_PG', 'C_PG', or 'GNN'. Only used if HFS or Astar are selected as "
          "search method."
          "If GNN is enabled, ensure you are using a model compiled with the "
          "'ENABLE_NEURALNETS' option; otherwise, torch will not be installed "
          "or linked for efficiency purposes.")
      ->check(CLI::IsMember({"SUBGOALS", "L_PG", "S_PG", "C_PG", "GNN"}))
      ->default_val("SUBGOALS");
  search_group
      ->add_option("--GNN_model", m_GNN_model_path,
                   "Specify the path of the model used by the heuristics "
                   "'GNN'. The default model is the one located in "
                   "'lib/gnn_handler/models/distance_estimator.onnx'. "
                   "Only used if "
                   "HFS/Astar with GNN heuristics is selected.")
      ->default_val("lib/gnn_handler/models/distance_estimator.onnx");
  search_group
      ->add_option("--GNN_constant_file", m_GNN_constant_path,
                   "Specify the path to the normalization constant file for "
                   "the GNN model. "
                   "Only used if --heuristics GNN is selected.")
      ->default_val("lib/gnn_handler/models/"
                    "distance_estimator_C.txt");

  /*search_group->add_option("--search_threads", m_threads_per_search,
                            "Set the number of threads to use for each search
     strategy (default: 1). If set > 1, each search strategy (e.g., BFS/DFS/HFS)
     will use this many threads.")
               ->default_val("1");*/

  // Portfolio group
  auto *portfolio_group = app.add_option_group("Portfolio Related");
  portfolio_group
      ->add_option(
          "-p,--portfolio_threads", m_portfolio_threads,
          "Set the number of portfolio threads (default: 1). If set > 1, "
          "multiple planner configurations will run in parallel. "
          "The configurations will override the specified search and heuristic "
          "options but will keep other options such as --bisimulation, "
          "--check_visited, etc. "
          "Currently, the portfolio supports up to 7 default configurations.")
      ->default_val("1");

  portfolio_group
      ->add_option(
          "--config_file", m_config_file,
          "Enable reading portfolio configuration from a file. If set, the "
          "planner will read the configuration from the specified file. "
          "An example can be found in `utils/configs/config.ut`. "
          "Please check the command line arguments for the possible field "
          "names (search-related options without the - or -- prefix). "
          "Whatever is set in the file will be used; otherwise, the given "
          "values will be used as defaults. The number of configurations to "
          "run in parallel is set by --portfolio_threads. "
          "Minimal parsing is performed, so the file should be well formatted.")
      ->default_val("");

  // Execution group
  auto *exec_group = app.add_option_group("Test Plan Execution");
  exec_group->add_flag(
      "-e,--execute_plan", m_exec_plan,
      "Enable execution mode. If set, the planner will verify a plan instead "
      "of searching for one. "
      "Actions to execute can be provided directly with --execute_actions, or "
      "will be read from the file specified by --plan_file (default: "
      "utils/plans/plan.ut). "
      "When this option is enabled, all multithreading, search strategy, and "
      "heuristic flags are ignored; only plan verification is performed. "
      "The plan file should contain a list of actions separated by spaces or "
      "commas. Minimal parsing is performed, so the file should be well "
      "formatted.");
  exec_group
      ->add_option("-a,--execute_actions", m_exec_actions,
                   "Specify a sequence of actions to execute directly, "
                   "bypassing planning. "
                   "Example: --execute_actions open_a peek_a. "
                   "If this option is set, the actions provided will be "
                   "executed in order. "
                   "If not set, actions will be loaded from the plan file (see "
                   "--plan_file).")
      ->expected(-1);
  exec_group
      ->add_option("--plan_file", m_plan_file,
                   "Specify the file from which to load the plan for execution "
                   "(default: utils/plans/plan.ut)."
                   "The syntax of the actions in the file should be "
                   "space-separated or comma-separated."
                   "Used only if --execute_plan is set and --execute_actions "
                   "is not provided.")
      ->default_val("utils/plans/plan.ut");
}

ArgumentParser::~ArgumentParser() {
  if (m_log_ofstream.is_open()) {
    m_log_ofstream.close();
  }
}

// Getters
const std::string &ArgumentParser::get_input_file() const noexcept {
  return m_input_file;
}

bool ArgumentParser::get_verbose() const noexcept { return m_verbose; }

bool ArgumentParser::get_check_visited() const noexcept {
  return m_check_visited;
}

bool ArgumentParser::get_bisimulation() const noexcept {
  return m_bisimulation;
}

const std::string &ArgumentParser::get_bisimulation_type() const noexcept {
  return m_bisimulation_type;
}

bool ArgumentParser::get_dataset_mode() const noexcept {
  return m_dataset_mode;
}

int ArgumentParser::get_dataset_depth() const noexcept {
  return m_dataset_depth;
}

double ArgumentParser::get_dataset_discard_factor() const noexcept {
  return m_dataset_discard_factor;
}

void ArgumentParser::set_dataset_type() noexcept {
  // convert to uppercase for case-insensitive comparison
  std::string value = m_dataset_type_string;

  std::ranges::transform(value, value.begin(),
                         [](const unsigned char c) { return std::toupper(c); });

  if (value == "MAPPED")
    m_dataset_type = DatasetType::MAPPED;
  else if (value == "HASHED")
    m_dataset_type = DatasetType::HASHED;
  else if (value == "BITMASK")
    m_dataset_type = DatasetType::BITMASK;
  else {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::ArgParseError,
        "Invalid dataset type: " + m_dataset_type_string +
            ". Expected one of: MAPPED, HASHED, BITMASK." +
            std::string(ExitHandler::arg_parse_suggestion));
  }
}

bool ArgumentParser::get_dataset_separated() const noexcept {
  return m_dataset_separated;
}

int64_t ArgumentParser::get_dataset_seed() const noexcept {
  return m_dataset_seed;
}

const std::string &ArgumentParser::get_heuristic() const noexcept {
  return m_heuristic_opt;
}

const std::string &ArgumentParser::get_GNN_model_path() const noexcept {
  return m_GNN_model_path;
}

const std::string &ArgumentParser::get_GNN_constant_path() const noexcept {
  return m_GNN_constant_path;
}

const std::string &ArgumentParser::get_search_strategy() const noexcept {
  return m_search_strategy;
}

bool ArgumentParser::get_execute_plan() const noexcept { return m_exec_plan; }

const std::string &ArgumentParser::get_plan_file() const noexcept {
  return m_plan_file;
}

const std::vector<std::string> &
ArgumentParser::get_execution_actions() noexcept {
  for (auto &action : m_exec_actions) {
    std::erase(action, ',');
  }
  return m_exec_actions;
}

bool ArgumentParser::get_results_info() const noexcept {
  return m_output_results_info;
}

bool ArgumentParser::get_log_enabled() const noexcept { return m_log_enabled; }

DatasetType ArgumentParser::get_dataset_type() const noexcept {
  return m_dataset_type;
}

std::ostream &ArgumentParser::get_output_stream() const {
  return *m_output_stream;
}

int ArgumentParser::get_threads_per_search() const noexcept {
  return m_threads_per_search;
}

int ArgumentParser::get_portfolio_threads() const noexcept {
  return m_portfolio_threads;
}

const std::string &ArgumentParser::get_config_file() const noexcept {
  return m_config_file;
}

void ArgumentParser::print_usage() const {
  std::cout << app.help() << std::endl;
  const std::string prog_name = "deep";
  std::cout << "\nEXAMPLES:\n";
  std::cout << "  " << prog_name << " domain.txt\n";
  std::cout << "    Find a plan for domain.txt\n\n";
  std::cout << "  " << prog_name
            << " domain.txt -s Astar --heuristic SUBGOALS\n";
  std::cout << "    Plan using heuristic 'SUBGOALS' and 'Astar' search\n\n";
  std::cout << "  " << prog_name
            << " domain.txt -e --execute-actions open_a peek_a\n";
  std::cout << "    Execute actions [open_a, peek_a] step by step\n\n";
  // std::cout << "  " << prog_name << " domain.txt --threads_per_search 4\n";
  // std::cout << "    Run search with 4 threads per search strategy\n\n";
  std::cout << "  " << prog_name << " domain.txt --portfolio_threads 3\n";
  std::cout
      << "    Run 3 planner configurations in parallel (portfolio search)\n\n";
  // std::cout << "  " << prog_name
  //           << " domain.txt --threads_per_search 2 --portfolio_threads 2\n";
  // std::cout << "    Run 2 planner configurations in parallel, each using 2 "
  //              "threads (total 4 threads)\n\n";
}
