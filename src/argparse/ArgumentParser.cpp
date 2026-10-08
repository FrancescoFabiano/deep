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
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>

#include "KripkeEqualityHelper.h"

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
#include <filesystem>

    if (m_log_enabled) {
      const std::string domain_name =
          std::filesystem::path(m_domain_file).stem().string();

      const std::string problem_name =
          std::filesystem::path(m_problem_file).stem().string();

      const std::string log_input = domain_name + "_" + problem_name;

      m_log_file_path = HelperPrint::generate_log_file_path(log_input);

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
         app.count("--dataset_discard_factor") || app.count("--dataset_seed") ||
         app.count("--dataset_max_generation") ||
         app.count("--dataset_min_creation") ||
         app.count("--dataset_max_creation"))) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::ArgParseError,
          "Dataset-related options (--dataset_depth, --dataset_discard_factor, "
          "--dataset_seed, etc.) "
          "were set but --dataset mode is not enabled. Please use --dataset to "
          "activate dataset mode.");
    }

    // Normalize and convert dataset-related string options to their enums.
    set_dataset_type();
    set_dataset_generation_type();

    // --- Bisimulation consistency check ---
    if (!m_bisimulation && app.count("--bisimulation_type")) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::ArgParseError,
          "Bisimulation type (--bisimulation_type) was set but --bisimulation "
          "is not enabled. Please use --bisimulation to activate it.");
    }

    // --- Heuristic consistency check ---
    const bool dataset_hfs = m_dataset_mode && m_dataset_generation_type ==
                                                   DatasetGenerationType::HFS;

    if (m_search_strategy != "HFS" && m_search_strategy != "Astar" &&
        m_search_strategy != "RL" && !dataset_hfs &&
        app.count("--heuristics")) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::ArgParseError,
          "--heuristics can only be used with --search HFS, --search Astar, "
          "--search RL, or --dataset --dataset_generation HFS.");
    }

    // RL heuristic can only be used with RL search
    if (m_heuristic_opt == "RL_H" && m_search_strategy != "RL") {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::ArgParseError,
          "Heuristic RL_H can only be used with RL search (--search RL).");
    }

    if (m_GNN_batch_size > 0 &&
        (m_heuristic_opt != "GNN" ||
         (m_search_strategy != "HFS" && m_search_strategy != "Astar"))) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::ArgParseError,
          "--GNN_batch can only be used with --heuristics GNN and --search "
          "HFS or --search Astar.");
    }

    if (m_RL_exploration_percentage + m_RL_exploitation_percentage >= 100) {
      ExitHandler::exit_with_message(ExitHandler::ExitCode::ArgParseError,
                                     "The sum of --RL_exploration and "
                                     "--RL_exploitation must be below 100.");
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
    if (!m_exec_plan && !m_exec_dump_successors.empty()) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::ArgParseError,
          "--execute_dump_successors can only be used with --execute_plan.");
    }
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

    // --- Learned sibling ranker: --ranker_model / --ranker_model_gpu ---
    // A folder holds per-domain/<domain name>.onnx (the name after "domain" in
    // the domain file) and general.onnx, used when the domain has none.
    namespace fs = std::filesystem;
    const auto resolve_ranker = [&](std::string &model,
                                    const std::string &flag) {
      if (fs::is_directory(model)) {
        std::ifstream in(m_domain_file);
        std::stringstream text;
        text << in.rdbuf();
        const std::string s = text.str();
        std::smatch m;
        std::string name;
        if (std::regex_search(
                s, m,
                std::regex(R"(\(\s*domain\s+([^\s()]+))", std::regex::icase))) {
          name = m[1];
        }
        const fs::path own = fs::path(model) / "per-domain" / (name + ".onnx"),
                       general = fs::path(model) / "general.onnx";
        if (!name.empty() && fs::exists(own)) {
          model = own.string();
        } else if (fs::exists(general)) {
          model = general.string();
        } else {
          ExitHandler::exit_with_message(
              ExitHandler::ExitCode::ArgParseError,
              flag + ": no model for domain '" + name +
                  "' and no general.onnx in " + model);
        }
      } else if (!fs::exists(model)) {
        ExitHandler::exit_with_message(ExitHandler::ExitCode::ArgParseError,
                                       flag + ": file not found: " + model);
      }
    };
    if (!m_ranker_model_gpu.empty() && m_ranker_model.empty()) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::ArgParseError,
          "--ranker_model_gpu needs --ranker_model (the model for the states "
          "below --ranker_gpu_edges).");
    }
    if (!m_ranker_model.empty()) {
      if (m_heuristic_opt != "GNN" && m_heuristic_opt != "RL_H") {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::ArgParseError,
            "--ranker_model needs --heuristics GNN (best-first search) or RL_H "
            "(RL beam search).");
      }
      resolve_ranker(m_ranker_model, "--ranker_model");
      get_output_stream() << "[INFO] Ranker model: " << m_ranker_model
                          << std::endl;
      if (!m_ranker_model_gpu.empty()) {
        resolve_ranker(m_ranker_model_gpu, "--ranker_model_gpu");
        get_output_stream()
            << "[INFO] Ranker model for states with at least "
            << m_ranker_gpu_edges << " edges (GPU): " << m_ranker_model_gpu
            << std::endl;
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
  app.add_option("domain_file", m_domain_file, "Specify the EPDDL domain file.")
      ->required();

  app.add_option("problem_file", m_problem_file,
                 "Specify the EPDDL problem file.")
      ->required();

  app.add_option(
      "--act_lib", m_library_files,
      "Specify a grounded EPDDL action library path processed through Plank. "
      "Can be provided multiple times.");

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
      "Activate epistemic-state reduction through bisimulation. Use this to "
      "reduce the state space by merging bisimilar states.");
  bis_group
      ->add_option("--bisimulation_type", m_bisimulation_type,
                   "Specify the algorithm for bisimulation contraction "
                   "(requires --bisimulation). Options: 'SIG' (signature "
                   "refinement on the Kripke structure, default), 'FB' (Fast "
                   "Bisimulation) or 'PT' (Paige and Tarjan).")
      ->check(CLI::IsMember({"FB", "PT", "SIG"}))
      ->default_val("SIG");
  bis_group
      ->add_option("--bisimulation-interval", m_bisimulation_interval,
                   "With -b, contract states every N search-depth levels; 0 "
                   "(default) or 1 contracts at every level. Without -b no "
                   "state is contracted.")
      ->default_val(0);

  // Dataset group
  auto *dataset_group = app.add_option_group("Dataset");
  dataset_group->add_flag(
      "-d,--dataset", m_dataset_mode,
      "Enable dataset generation mode for learning or analysis.");
  dataset_group
      ->add_option("--dataset_depth", m_dataset_depth,
                   "Set the maximum depth for dataset generation.")
      ->default_val("10");
  dataset_group
      ->add_option("--dataset_max_generation", m_dataset_generation_threshold,
                   "Set the maximum number of nodes to generate before dataset "
                   "generation stops.")
      ->default_val("100000");
  dataset_group
      ->add_option("--dataset_max_creation", m_dataset_max_creation_threshold,
                   "Set the maximum number of valid nodes to create before "
                   "dataset generation stops.")
      ->default_val("60000");
  dataset_group
      ->add_option("--dataset_min_creation", m_dataset_min_creation_threshold,
                   "Set the minimum number of valid nodes to create to create "
                   "a meaningful dataset.")
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
  dataset_group
      ->add_option(
          "--dataset_generation", m_dataset_generation_type_string,
          "Specify the search strategy used for dataset generation. "
          "Options: BFS (Breadth First Search), DFS (Depth First Search), "
          "S_DFS (Stochastic Depth First Search), or HFS (Heuristic First "
          "Search). When HFS is selected, --heuristics specifies the "
          "heuristic.")
      ->check(CLI::IsMember({"BFS", "DFS", "S_DFS", "HFS"}))
      ->default_val("S_DFS");
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
                   "Select the search strategy: 'BFS' (Breadth First Search, "
                   "default), 'DFS' (Depth First Search), 'IDFS' (Iterative "
                   "Depth First Search), 'HFS' (Heuristic First Search), "
                   "'Astar' (A* Search, uses heuristics with A* method), or "
                   "'RL' (Reinforcement Learning-Like search).")
      ->check(CLI::IsMember({"BFS", "DFS", "IDFS", "HFS", "Astar", "RL"}))
      ->default_val("BFS");
  search_group->add_flag("-c,--check_visited", m_check_visited,
                         "Enable checking for previously visited states during "
                         "planning to avoid redundant exploration.");
  search_group
      ->add_option(
          "-u,--heuristics", m_heuristic_opt,
          "Specify the heuristic for HFS, Astar or RL search: 'SUBGOALS' "
          "(default), 'L_PG', "
          "'S_PG', 'C_PG', 'GNN', or 'RL_H'. Only used if HFS, Astar, or RL "
          "are selected as "
          "search method. "
          "'RL_H' only works in association with RL search method."
          "If GNN or RL_H are enabled, ensure the planner was built with "
          "neural-network support so the required ONNX Runtime integration is "
          "available.")
      ->check(
          CLI::IsMember({"SUBGOALS", "L_PG", "S_PG", "C_PG", "GNN", "RL_H"}))
      ->default_val("SUBGOALS");
  search_group
      ->add_option("--GNN_model", m_GNN_model_path,
                   "Specify the path of the model used by the heuristics "
                   "'GNN'. The default model is the one located in "
                   "'lib/gnn_handler/models/distance_estimator.onnx'. "
                   "Only used if "
                   "HFS/Astar/RL with GNN heuristics is selected.")
      ->default_val("lib/gnn_handler/models/distance_estimator.onnx");
  search_group
      ->add_option(
          "--GNN_batch", m_GNN_batch_size,
          "Evaluate the GNN heuristic on batches of up to N successors with a "
          "fringe export of the model (--GNN_model gnn_F<N>.onnx, N its "
          "fringe size) instead of one state at a time. The heuristic is the "
          "model's absolute distance to the goal: --search HFS orders states "
          "by it, --search Astar by depth + distance. 0 (default) keeps the "
          "per-state model and its --GNN_constant_file.")
      ->check(CLI::NonNegativeNumber)
      ->default_val("0");
  search_group
      ->add_option("--GNN_constant_file", m_GNN_constant_path,
                   "Specify the path to the normalization constant file for "
                   "the GNN model. "
                   "Only used if --heuristics GNN is selected.")
      ->default_val("lib/gnn_handler/models/"
                    "distance_estimator_C.txt");

  search_group
      ->add_option(
          "--RL_model", m_RL_model_path,
          "Specify the path of the model used by the search and heuristics "
          "'RL'. The default model is the one located in "
          "'lib/rl_handler/models/rl_estimator.onnx'. "
          "Only used if RL (with or without 'RL-H') is selected.")
      ->default_val("lib/rl_handler/models/rl_estimator.onnx");

  search_group
      ->add_option("--RL_fringe_size", m_RL_fringe_size,
                   "The size of the fringe on which RL is applied.")
      ->default_val("32");

  search_group
      ->add_option("--RL_exploration", m_RL_exploration_percentage,
                   "The maximum percentage of the fringe to be filled with "
                   "random states "
                   "to allow exploration. The new fringe is the exploration "
                   "(states expanded from the best of the previous fringe) "
                   "plus the ones "
                   "from the reservoir plus these.")
      ->default_val("10");

  search_group
      ->add_option(
          "--RL_exploitation", m_RL_exploitation_percentage,
          "The minimum percentage of the fringe to be filled by new states if "
          "possible "
          "(expanded from the best of the previous fringe). The new fringe is "
          "these plus the ones from the reservoir plus the exploration.")
      ->default_val("70");

  search_group
      ->add_option("--tie_breaking", m_tie_breaking,
                   "Order of HFS/A* states with equal values: 'none' (default) "
                   "compares only h (HFS) or f (A*) and leaves ties to the "
                   "heap; 'fifo', 'lifo' and 'random' first compare h within "
                   "equal f (A*) and then expand the oldest, the newest or a "
                   "random one (seeded by --tie_breaking_seed).")
      ->check(CLI::IsMember({"none", "fifo", "lifo", "random"}))
      ->default_val("none");
  search_group
      ->add_option("--tie_breaking_seed", m_tie_breaking_seed,
                   "Seed of --tie_breaking random (the same seed gives the "
                   "same search).")
      ->check(CLI::NonNegativeNumber)
      ->default_val("42");
  search_group->add_flag(
      "--GNN_raw_distance", m_GNN_raw_distance,
      "With --GNN_batch, order states by the GNN's distance as predicted "
      "(fractional) instead of rounded to an integer.");

  search_group
      ->add_option("--RL_heuristics", m_RL_heur_selection,
                   "Specify the heuristic mode for RL.")
      ->check(CLI::IsMember({"MIN", "MAX", "AVG", "RNG"}))
      ->default_val("MIN");

  search_group
      ->add_option("--onnx_device", m_onnx_device,
                   "Where the neural networks run: 'cpu', 'cuda' (fails if "
                   "CUDA is not available: needs a build with use_gpu and a "
                   "GPU ONNX Runtime), or 'auto' (default: CUDA when the "
                   "build has it, otherwise the CPU).")
      ->check(CLI::IsMember({"auto", "cpu", "cuda"}))
      ->default_val("auto");
  search_group
      ->add_option("--onnx_device_id", m_onnx_device_id,
                   "CUDA device used by --onnx_device cuda/auto.")
      ->check(CLI::NonNegativeNumber)
      ->default_val("0");
  search_group
      ->add_option("--onnx_gpu_mem_limit", m_onnx_gpu_mem_limit_mib,
                   "Cap (MiB) on the GPU memory ONNX Runtime may reserve for "
                   "the networks; a run that needs more stops with an error. "
                   "0 (default) means no cap.")
      ->check(CLI::NonNegativeNumber)
      ->default_val("0");
  search_group->add_flag(
      "--onnx_placement", m_onnx_placement,
      "Print the execution provider (CPU or CUDA) every model node was "
      "placed on when the model is loaded, and ORT's warnings about nodes "
      "left on the CPU or copies between devices.");
  search_group->add_option(
      "--ranker_model", m_ranker_model,
      "Learned sibling ranker as the heuristic: with --heuristics GNN it "
      "guides best-first search, with --search RL --heuristics RL_H it ranks "
      "the beam in place of the RL network. "
      "The value is an ONNX model with its .vocab sidecar, or a folder with "
      "per-domain/<domain name>.onnx and general.onnx, such as "
      "exp/gnn_epddl/models/cpu. Runs on the CPU when --ranker_model_gpu is "
      "given, else on the device of --onnx_device.");
  search_group->add_option(
      "--ranker_model_gpu", m_ranker_model_gpu,
      "The same ranker exported for the GPU (file or folder, as "
      "--ranker_model, such as exp/gnn_epddl/models/gpu): states with at "
      "least --ranker_gpu_edges edges are scored with it on CUDA, smaller "
      "ones with --ranker_model on the CPU. Needs a CUDA build (build.sh nn "
      "use_gpu).");
  search_group
      ->add_option("--ranker_gpu_edges", m_ranker_gpu_edges,
                   "Edge count (both directions) from which a state is scored "
                   "on the GPU with --ranker_model_gpu.")
      ->default_val(10000)
      ->check(CLI::NonNegativeNumber);
  dataset_group->add_flag(
      "--ranker_encoding", m_ranker_encoding,
      "Write the learned ranker's state encoding in the dataset DOT files: an "
      "edge (label 4) from every world to each positive fluent true in it, the "
      "fluent names in fluent_names.csv and the goal operators in "
      "goal_ops.csv (training data).");
  search_group
      ->add_option("--onnx_threads", m_onnx_threads,
                   "Number of threads ONNX Runtime may use for one model "
                   "evaluation. 0 (default) keeps the ONNX Runtime default "
                   "(one per core); 1 keeps neural-network planning "
                   "single-threaded, e.g. for CPU-time measured runs.")
      ->check(CLI::NonNegativeNumber)
      ->default_val("0");

  dataset_group
      ->add_option("--RL_seed", m_RL_seed,
                   "Set the seed used for RL exploration and RNG heuristics. "
                   "If no seed is provided, the default (94) is used. "
                   "If a negative value is given, a random seed will be "
                   "generated instead, "
                   "as negative seeds are not accepted.")
      ->default_val("94");

  search_group->add_flag(
      "--fast-world-comparison", m_fast_world_comparison,
      "Use KripkeWorldPointer IDs only for world comparison, skipping the "
      "structural collision fallback (might not be complete).");

  search_group->add_flag(
      "--fast-state-comparison", m_fast_state_comparison,
      "Use state hashes only for state comparison, skipping the "
      "structural collision fallback (might not be complete).");

  search_group->add_flag_function(
      "--fast-comparison",
      [this](std::int64_t) {
        m_fast_world_comparison = true;
        m_fast_state_comparison = true;
      },
      "Enable both --fast-world-comparison and --fast-state-comparison.");

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
          "Set the number of portfolio threads. If set > 1, "
          "multiple planner configurations will run in parallel. "
          "The configurations will override the specified search and heuristic "
          "options. The built-in default portfolio enables bisimulation, "
          "visited-state checking, and both fast comparison modes. "
          "Currently, the portfolio supports up to 9 default configurations.")
      ->default_val("1");

  portfolio_group
      ->add_option(
          "--config_file", m_config_file,
          "Enable reading portfolio configuration from a file. If set, the "
          "planner will read the configuration from the specified file. "
          "Examples can be found in `utils/configs/config-P5.ut` and "
          "`utils/configs/config-ALL.ut`. "
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
                   "Example: --execute_actions open_A peek_A. "
                   "If this option is set, the actions provided will be "
                   "executed in order. "
                   "If not set, actions will be loaded from the plan file (see "
                   "--plan_file).")
      ->expected(-1);
  exec_group
      ->add_option("--plan_file", m_plan_file,
                   "Specify the file from which to load the plan for execution "
                   "(default: utils/plans/plan.ut)."
                   "The action names in the file should be "
                   "space-separated or comma-separated."
                   "Used only if --execute_plan is set and --execute_actions "
                   "is not provided.")
      ->default_val("utils/plans/plan.ut");
  exec_group
      ->add_option(
          "--execute_dump_successors", m_exec_dump_successors,
          "With --execute_plan, write every successor of every plan step "
          "(dataset DOT format, contracted as in search) into this folder, "
          "indexed by successors.csv (step, action, is_plan_action, is_goal, "
          "revisits_plan, file). Used to probe learned heuristics along a "
          "known plan.")
      ->default_val("");
  exec_group
      ->add_option(
          "--expand_server", m_expand_server,
          "Instead of searching, serve expansions to an external search (the "
          "ranker's training code): the initial state is id 0 (written as "
          "<folder>/0.dot); each stdin line 'expand <id>' generates that "
          "state's successors (contracted and goal-tested as in search, "
          "duplicates detected with the visited set when -c is set), writes "
          "each new one as <folder>/<id>.dot (dataset format) and prints '@@ "
          "<id> <action> <is_goal> <is_new>' per successor, then '@@end'. "
          "'quit' stops.")
      ->default_val("");
}

ArgumentParser::~ArgumentParser() {
  if (m_log_ofstream.is_open()) {
    m_log_ofstream.close();
  }
}

const std::string &ArgumentParser::get_domain_file() const noexcept {
  return m_domain_file;
}

const std::string &ArgumentParser::get_problem_file() const noexcept {
  return m_problem_file;
}

const std::vector<std::string> &
ArgumentParser::get_library_files() const noexcept {
  return m_library_files;
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

std::size_t ArgumentParser::get_bisimulation_interval() const noexcept {
  return m_bisimulation_interval;
}

bool ArgumentParser::get_dataset_mode() const noexcept {
  return m_dataset_mode;
}

int ArgumentParser::get_dataset_depth() const noexcept {
  return m_dataset_depth;
}

int ArgumentParser::get_generation_threshold() const noexcept {
  return m_dataset_generation_threshold;
}

int ArgumentParser::get_max_creation_threshold() const noexcept {
  return m_dataset_max_creation_threshold;
}

int ArgumentParser::get_min_creation_threshold() const noexcept {
  return m_dataset_min_creation_threshold;
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

void ArgumentParser::set_dataset_generation_type() noexcept {
  std::string value = m_dataset_generation_type_string;

  std::ranges::transform(value, value.begin(),
                         [](const unsigned char c) { return std::toupper(c); });

  if (value == "BFS") {
    m_dataset_generation_type = DatasetGenerationType::BFS;
  } else if (value == "DFS") {
    m_dataset_generation_type = DatasetGenerationType::DFS;
  } else if (value == "S_DFS") {
    m_dataset_generation_type = DatasetGenerationType::S_DFS;
  } else if (value == "HFS") {
    m_dataset_generation_type = DatasetGenerationType::HFS;
  } else {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::ArgParseError,
        "Invalid dataset generation type: " + m_dataset_generation_type_string +
            ". Expected one of: BFS, DFS, S_DFS, HFS." +
            std::string(ExitHandler::arg_parse_suggestion));
  }

  m_dataset_generation_type_string = value;
}

bool ArgumentParser::get_dataset_separated() const noexcept {
  return m_dataset_separated;
}

int64_t ArgumentParser::get_dataset_seed() const noexcept {
  return m_dataset_seed;
}

int64_t ArgumentParser::get_RL_seed() const noexcept { return m_RL_seed; }

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

const std::string &ArgumentParser::get_RL_model_path() const noexcept {
  return m_RL_model_path;
}

int ArgumentParser::get_RL_fringe_size() const noexcept {
  return m_RL_fringe_size;
}

int ArgumentParser::get_RL_exploration_percentage() const noexcept {
  return m_RL_exploration_percentage;
}

int ArgumentParser::get_RL_exploitation_percentage() const noexcept {
  return m_RL_exploitation_percentage;
}

std::string ArgumentParser::get_RL_heur_selection() const noexcept {
  return m_RL_heur_selection;
}

int ArgumentParser::get_onnx_threads() const noexcept { return m_onnx_threads; }

const std::string &ArgumentParser::get_onnx_device() const noexcept {
  return m_onnx_device;
}

int ArgumentParser::get_onnx_device_id() const noexcept {
  return m_onnx_device_id;
}

int ArgumentParser::get_onnx_gpu_mem_limit_mib() const noexcept {
  return m_onnx_gpu_mem_limit_mib;
}

bool ArgumentParser::get_onnx_placement() const noexcept {
  return m_onnx_placement;
}

bool ArgumentParser::get_GNN_raw_distance() const noexcept {
  return m_GNN_raw_distance;
}

const std::string &ArgumentParser::get_tie_breaking() const noexcept {
  return m_tie_breaking;
}

std::uint64_t ArgumentParser::get_tie_breaking_seed() const noexcept {
  return m_tie_breaking_seed;
}

int ArgumentParser::get_GNN_batch_size() const noexcept {
  return m_GNN_batch_size;
}
bool ArgumentParser::get_execute_plan() const noexcept { return m_exec_plan; }

const std::string &ArgumentParser::get_plan_file() const noexcept {
  return m_plan_file;
}

const std::string &
ArgumentParser::get_execute_dump_successors() const noexcept {
  return m_exec_dump_successors;
}

const std::string &ArgumentParser::get_ranker_model() const noexcept {
  return m_ranker_model;
}

const std::string &ArgumentParser::get_ranker_model_gpu() const noexcept {
  return m_ranker_model_gpu;
}

int ArgumentParser::get_ranker_gpu_edges() const noexcept {
  return m_ranker_gpu_edges;
}

bool ArgumentParser::get_ranker_encoding() const noexcept {
  return m_ranker_encoding;
}

const std::string &ArgumentParser::get_expand_server() const noexcept {
  return m_expand_server;
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

DatasetGenerationType
ArgumentParser::get_dataset_generation_type() const noexcept {
  return m_dataset_generation_type;
}

std::string
ArgumentParser::get_dataset_generation_type_string() const noexcept {
  if (m_dataset_generation_type == DatasetGenerationType::HFS) {
    return m_dataset_generation_type_string + " (" + m_heuristic_opt + ")";
  }

  return m_dataset_generation_type_string;
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
  std::string prog_name = "deep";
  std::cout << "\nEXAMPLES:\n";
  std::cout << "  " << prog_name << " domain.epddl problem.epddl\n";
  std::cout << "    Find a plan for the given domain/problem pair\n\n";
  std::cout << "  " << prog_name
            << " domain.epddl problem.epddl -s Astar --heuristics SUBGOALS\n";
  std::cout << "    Plan using heuristic 'SUBGOALS' and 'Astar' search\n\n";
  std::cout << "  " << prog_name
            << " domain.epddl problem.epddl --act_lib library.epddl -e -a "
               "open_A peek_A\n";
  std::cout << "    Execute actions [open_A, peek_A] step by step\n\n";
  // std::cout << "  " << prog_name
  //           << " domain.epddl problem.epddl --threads_per_search 4\n";
  // std::cout << "    Run search with 4 threads per search strategy\n\n";
  std::cout << "  " << prog_name
            << " domain.epddl problem.epddl --portfolio_threads 3\n";
  std::cout
      << "    Run 3 planner configurations in parallel (portfolio search)\n\n";
  // std::cout << "  " << prog_name
  //           << " domain.epddl problem.epddl --threads_per_search 2"
  //              " --portfolio_threads 2\n";
  // std::cout << "    Run 2 planner configurations in parallel, each using 2 "
  //              "threads (total 4 threads)\n\n";
}
