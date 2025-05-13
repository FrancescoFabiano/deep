#include "ArgumentParser.h"

ArgumentParser& ArgumentParser::getInstance(int argc, char** argv) {
    static ArgumentParser instance(argc, argv);
    return instance;
}

ArgumentParser::ArgumentParser(int argc, char** argv) : app("Argument Parser") {
    app.add_option("input_file", input_file, "Specify input domain file")->required();

    app.add_flag("--debug", debug, "Makes the solving process verbose");

    app.add_option("--ir", restrict_initial,
        "Set the restriction that the Initial state must encode.")
        ->check(CLI::IsMember({"S5", "K45", "NONE"}))
        ->default_val("S5");

    app.add_option("--gr", restrict_goal,
        "Set the restriction for the Goal description.")
        ->check(CLI::IsMember({"NONE", "NONEG"}))
        ->default_val("NONE");

    app.add_option("--bis", bisim,
        "Set the bisimulation type used for kstate reduction.")
        ->check(CLI::IsMember({"NONE", "PT", "FB"}))
        ->default_val("NONE");

    app.add_option("--act_obsv", observability,
        "Set action observability type.")
        ->check(CLI::IsMember({"GLOBAL", "RELATIVE"}))
        ->default_val("GLOBAL");

    app.add_option("--act_check", executability,
        "Set the type of executability check for actions.")
        ->check(CLI::IsMember({"PW", "PP", "WW"}))
        ->default_val("PW");

    app.add_flag("--check_visited", check_visited,
        "Planner will check for visited states (default: false)");

    app.add_flag("--generate_asp", generate_asp_mode,
        "Generate a version of the input file for ASP solver");

    app.add_option("--generate_dataset", dataset_type,
        "Planner will generate dataset for ML heuristic.")
        ->check(CLI::IsMember({"DFS", "BFS"}))
        ->default_val("DFS");

    app.add_option("--dataset-size", dataset_depth, "Maximum depth for dataset generation")
        ->default_val("10");

    app.add_option("--search", search_strategy,
        "Search strategy to use: BFS (default), DFS, I_DFS")
        ->check(CLI::IsMember({"BFS", "DFS", "I_DFS"}))
        ->default_val("BFS");

    app.add_option("--heuristic", heuristic_opt,
        "Heuristic to use: NONE (default), L_PG, S_PG, C_PG, GNN, SUBGOALS")
        ->check(CLI::IsMember({"NONE", "L_PG", "S_PG", "C_PG", "GNN", "SUBGOALS"}))
        ->default_val("NONE");

    app.add_option("--parallel", parallel_type,
        "Run all heuristics in parallel. Overrides --heuristic and --search.")
        ->check(CLI::IsMember({"SERIAL", "PTHREAD", "FORK"}))
        ->default_val("PTHREAD");

    app.add_option("--parallel-wait", parallel_wait,
        "Decide whether to wait for all parallel threads/processes.")
        ->check(CLI::IsMember({"NONE", "WAIT"}))
        ->default_val("NONE");

    app.add_flag("--results_file", output_results_file,
        "Log plan execution time to a file for test comparisons");

    app.add_option("--execute-actions", exec_actions,
        "Perform sequence of actions instead of planning")->expected(-1);

    app.add_flag("--execute", exec_plan, "Execute the generated plan");
    app.add_option("--plan-file", plan_file, "Specify file for saving/loading plan")
        ->default_val("plan.txt");

    app.parse(argc, argv);
}

// Getters
bool ArgumentParser::debugMode() const { return debug; }
bool ArgumentParser::verbose() const { return verbose; }
bool ArgumentParser::restrictInitial() const { return restrict_initial != "NONE"; }
bool ArgumentParser::restrictGoal() const { return restrict_goal != "NONE"; }
bool ArgumentParser::allowUnreachable() const { return !check_visited; }
std::string ArgumentParser::bisimulationType() const { return bisim; }
bool ArgumentParser::useBisimulation() const { return bisim != "NONE"; }
bool ArgumentParser::isObservable() const { return observability == "GLOBAL"; }
bool ArgumentParser::isExecutable() const { return !executability.empty(); }
bool ArgumentParser::isDatasetMode() const { return !dataset_type.empty(); }
int ArgumentParser::datasetSize() const { return dataset_depth; }
std::string ArgumentParser::datasetName() const { return dataset_name; }
std::string ArgumentParser::datasetType() const { return dataset_type; }
std::string ArgumentParser::heuristic() const { return heuristic_opt; }
std::string ArgumentParser::searchStrategy() const { return search_strategy; }
bool ArgumentParser::executePlan() const { return exec_plan; }
std::string ArgumentParser::planFile() const { return plan_file; }
std::string ArgumentParser::inputFile() const { return input_file; }
std::vector<std::string> ArgumentParser::executionActions() const { return exec_actions; }
bool ArgumentParser::generateAsp() const { return generate_asp_mode; }
bool ArgumentParser::resultsFile() const { return output_results_file; }
std::string ArgumentParser::parallelType() const { return parallel_type; }
std::string ArgumentParser::parallelWait() const { return parallel_wait; }
bool ArgumentParser::checkVisited() const { return check_visited; }

void ArgumentParser::printUsage() const {
    std::cout << app.help() << std::endl;
    std::string prog_name = "planner";
    std::cout << "\nEXAMPLES:\n";
    std::cout << "  " << prog_name << " domain.al\n";
    std::cout << "    Find a plan for domain.al\n\n";
    std::cout << "  " << prog_name << " domain.al --heuristic SUBGOALS\n";
    std::cout << "    Plan using heuristic 'SUBGOALS'\n\n";
    std::cout << "  " << prog_name << " domain.al --execute-actions open_a peek_a\n";
    std::cout << "    Execute actions [open_a, peek_a] step by step\n\n";
}
