//
// Created by fraano on 13/05/2025.
//

#ifndef ARGUMENT_PARSER_H
#define ARGUMENT_PARSER_H

#include <string>
#include <vector>
#include <memory>
#include <CLI/CLI.hpp>

/**
 * @class ArgumentParser
 * @brief Singleton class to parse and access command-line arguments using CLI11.
 *
 * This class handles parsing of command-line arguments and provides access to
 * parsed values. It is designed as a singleton to ensure only one instance
 * manages the arguments for the application. It also offers convenient getter
 * methods to retrieve the values of the parsed arguments.
 */
class ArgumentParser {
public:
    /**
     * @brief Returns the singleton instance of ArgumentParser.
     *
     * This method creates and returns a single instance of ArgumentParser,
     * ensuring the application only has one parser instance.
     *
     * @param argc Argument count.
     * @param argv Argument values.
     * @return Reference to the singleton instance.
     */
    static ArgumentParser& getInstance(int argc = 0, char** argv = nullptr);

    // Accessors

    /**
     * @brief Checks if debug mode is enabled.
     * @return true if debug mode is enabled, false otherwise.
     */
    bool debugMode() const;

    /**
     * @brief Checks if verbose mode is enabled.
     * @return true if verbose mode is enabled, false otherwise.
     */
    bool verbose() const;

    /**
     * @brief Checks if initial state restriction is enabled.
     * @return true if initial state restriction is enabled, false otherwise.
     */
    bool restrictInitial() const;

    /**
     * @brief Checks if goal restriction is enabled.
     * @return true if goal restriction is enabled, false otherwise.
     */
    bool restrictGoal() const;

    /**
     * @brief Checks if unreachable states are allowed.
     * @return true if unreachable states are allowed, false otherwise.
     */
    bool allowUnreachable() const;

    /**
     * @brief Retrieves the bisimulation type.
     * @return The bisimulation type as a string.
     */
    std::string bisimulationType() const;

    /**
     * @brief Checks if bisimulation is used.
     * @return true if bisimulation is used, false otherwise.
     */
    bool useBisimulation() const;

    /**
     * @brief Checks if action observability is enabled.
     * @return true if observability is enabled, false otherwise.
     */
    bool isObservable() const;

    /**
     * @brief Checks if action executability is enabled.
     * @return true if executability is enabled, false otherwise.
     */
    bool isExecutable() const;

    /**
     * @brief Checks if dataset generation mode is enabled.
     * @return true if dataset mode is enabled, false otherwise.
     */
    bool isDatasetMode() const;

    /**
     * @brief Retrieves the dataset size.
     * @return The dataset size.
     */
    int datasetSize() const;

    /**
     * @brief Retrieves the dataset name.
     * @return The dataset name.
     */
    std::string datasetName() const;

    /**
     * @brief Retrieves the dataset generation type (DFS or BFS).
     * @return The dataset generation type.
     */
    std::string datasetType() const;

    /**
     * @brief Retrieves the heuristic to be used.
     * @return The heuristic type as a string.
     */
    std::string heuristic() const;

    /**
     * @brief Retrieves the search strategy to be used.
     * @return The search strategy as a string.
     */
    std::string searchStrategy() const;

    /**
     * @brief Checks if the generated plan should be executed.
     * @return true if the plan should be executed, false otherwise.
     */
    bool executePlan() const;

    /**
     * @brief Retrieves the file path for saving/loading the plan.
     * @return The plan file path.
     */
    std::string planFile() const;

    /**
     * @brief Retrieves the input file path.
     * @return The input file path.
     */
    std::string inputFile() const;

    /**
     * @brief Retrieves the sequence of actions to execute.
     * @return A vector containing the actions to be executed.
     */
    std::vector<std::string> executionActions() const;

    /**
     * @brief Checks if the ASP solver version of the input file should be generated.
     * @return true if ASP generation is enabled, false otherwise.
     */
    bool generateAsp() const;

    /**
     * @brief Checks if the results should be logged to a file.
     * @return true if results file logging is enabled, false otherwise.
     */
    bool resultsFile() const;

    /**
     * @brief Retrieves the parallel type to be used for running heuristics.
     * @return The parallel type (e.g., "SERIAL", "PTHREAD", "FORK").
     */
    std::string parallelType() const;

    /**
     * @brief Retrieves the parallel wait strategy.
     * @return The parallel wait strategy (e.g., "NONE", "WAIT").
     */
    std::string parallelWait() const;

    /**
     * @brief Checks if visited states should be checked during planning.
     * @return true if visited state checking is enabled, false otherwise.
     */
    bool checkVisited() const;

    /**
     * @brief Prints the usage of the application (command-line arguments).
     */
    void printUsage() const;

private:
    /**
     * @brief Constructor for ArgumentParser, parsing the command-line arguments.
     * @param argc Argument count.
     * @param argv Argument values.
     */
    ArgumentParser(int argc, char** argv);

    CLI::App app;  ///< CLI11 app object for argument parsing.

    // Singleton instance tracking
    static std::unique_ptr<ArgumentParser> instance;

    // Option storage
    std::string input_file;  ///< Input domain file path.
    bool debug = false;      ///< Debug mode flag.
    bool verbose = false;    ///< Verbose mode flag.

    std::string restrict_initial = "S5";  ///< Initial state restriction (S5 by default).
    std::string restrict_goal = "NONE";   ///< Goal state restriction (NONE by default).

    std::string bisim = "NONE";  ///< Bisimulation type (NONE by default).

    std::string observability = "GLOBAL";  ///< Action observability (GLOBAL by default).
    std::string executability = "PW";      ///< Action executability (PW by default).

    bool check_visited = false;  ///< Flag to check for visited states.

    bool dataset_mode = false;  ///< Flag to indicate dataset mode.
    int dataset_depth = 10;     ///< Maximum depth for dataset generation.
    std::string dataset_name = "default";  ///< Dataset name.
    std::string dataset_type = "DFS";     ///< Dataset generation type (DFS by default).

    std::string heuristic_opt = "NONE";  ///< Heuristic type (NONE by default).
    std::string search_strategy = "BFS"; ///< Search strategy (BFS by default).

    std::string parallel_type = "PTHREAD";  ///< Parallel execution type (PTHREAD by default).
    std::string parallel_wait = "NONE";     ///< Parallel wait strategy (NONE by default).

    bool exec_plan = false;  ///< Flag to indicate if the plan should be executed.
    std::vector<std::string> exec_actions;  ///< Actions to execute instead of planning.

    bool output_results_file = false;  ///< Flag to enable results file logging.
    bool generate_asp_mode = false;   ///< Flag to enable ASP solver generation.

    std::string plan_file = "plan.txt";  ///< Plan file path.
};

#endif // ARGUMENT_PARSER_H
