#pragma once

#include <string>
#include <vector>
#include <CLI/CLI.hpp>
#include "utilities/ExitHandler.h"


/**
 * \class ArgumentParser
 * \brief Singleton class to parse and access command-line arguments using CLI11.
 *
 * The ArgumentParser class handles parsing of command-line arguments and provides access to
 * parsed values. It is designed as a singleton to ensure only one instance manages the arguments
 * for the application. It also offers convenient getter methods to retrieve the values of the parsed arguments.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 12, 2025
 */
class ArgumentParser {
public:
    /**
     * \brief Creates the singleton instance of ArgumentParser.
     *
     * This method explicitly creates the singleton instance of ArgumentParser
     * with the provided command-line arguments. It should be called once at
     * program startup before calling get_instance().
     *
     * \param argc Argument count.
     * \param argv Argument values.
     */
    static void create_instance(int argc, char** argv);

    /**
     * \brief Returns the singleton instance of ArgumentParser.
     *
     * This method returns a single instance of ArgumentParser,
     * ensuring the application only has one parser instance.
     *
     * \return Reference to the singleton instance.
     */
    static ArgumentParser& get_instance();


    /**
       * \brief Checks if debug mode is enabled.
       * \return true if debug mode is enabled, false otherwise.
       */
    [[nodiscard]] bool get_debug() const noexcept;

    /**
     * \brief Checks if dataset generation mode is enabled.
     * \return true if dataset mode is enabled, false otherwise.
     */
    [[nodiscard]] bool get_dataset_mode() const noexcept;

    /**
     * \brief Retrieves the dataset size.
     * \return The dataset size.
     */
    [[nodiscard]] int get_dataset_size() const noexcept;

    /**
     * \brief Checks if the generated plan should be executed.
     * \return true if the plan should be executed, false otherwise.
     */
    [[nodiscard]] bool get_execute_plan() const noexcept;

    /**
     * \brief Retrieves the file path for saving/loading the plan.
     * \return The plan file path.
     */
    [[nodiscard]] const std::string& get_plan_file() const noexcept;

    /**
     * \brief Retrieves the input file path.
     * \return The input file path.
     */
    [[nodiscard]] const std::string& get_input_file() const noexcept;

    /**
     * \brief Retrieves the sequence of actions to execute.
     * \return A vector containing the actions to be executed.
     */
    [[nodiscard]] const std::vector<std::string>& get_execution_actions() const noexcept;

    /**
     * \brief Checks if the results should be logged to a file.
     * \return true if results file logging is enabled, false otherwise.
     */
    [[nodiscard]] bool get_results_file() const noexcept;

    /**
     * \brief Checks if logging to a file is enabled.
     * \return true if logging is enabled, false otherwise.
     */
    [[nodiscard]] bool get_log_enabled() const noexcept;

    /**
     * \brief Prints the usage of the application (command-line arguments).
     */
    void print_usage() const;

    /** \brief Copy constructor removed since this is a Singleton class. */
    ArgumentParser(const ArgumentParser&) = delete;
    /** \brief Copy operator removed since this is a Singleton class. */
    ArgumentParser& operator=(const ArgumentParser&) = delete;

private:
    /**
     * \brief Constructor for ArgumentParser, parsing the command-line arguments.
     */
    ArgumentParser();

    /**
     * \brief Parses the command-line arguments.
     * \param argc Argument count.
     * \param argv Argument values.
     */
    void parse(int argc, char** argv);

    CLI::App app;  ///< CLI11 app object for argument parsing.


    static ArgumentParser* instance;  ///< Singleton instance of the class.

    // Option storage
    std::string m_input_file;  ///< Input domain file path.
    bool m_debug = false;      ///< Debug mode flag.
    bool m_bisimulation = false;  ///< Bisimulation type (NONE by default).
    std::string m_bisimulation_type = "FB";  ///< Bisimulation type (PT by default).
    bool m_check_visited = false;  ///< Flag to check for visited states.
    bool m_dataset_mode = false;  ///< Flag to indicate dataset mode.
    int m_dataset_depth = 10;     ///< Maximum depth for dataset generation.
    std::string m_search_strategy = "BFS"; ///< Search strategy (BFS by default).
    std::string m_heuristic_opt = "SUBGOALS";  ///< Heuristic type (SUBGOALS by default).
    bool m_exec_plan = false;  ///< Flag to indicate if the plan should be executed.
    std::vector<std::string> m_exec_actions;  ///< Actions to execute instead of planning.
    bool m_output_results_file = false;  ///< Flag to enable results file logging.
    std::string m_plan_file = "plan.txt";  ///< Plan file path.
    bool m_log_enabled = false; ///< True if --log is enabled.
    std::string m_log_file_path; ///< The log file path if logging is enabled.


    // Accessors private because they can be accessed only by friend class \ref Configuration

    /**
     * \brief Checks if bisimulation is used.
     * \return true if bisimulation is used, false otherwise.
     */
    [[nodiscard]] bool get_bisimulation() const noexcept;

    /**
    * \brief Return the type of Bisimulation Adopted.
    * \return the string that specifies the type of bisimulation used.
    */
    [[nodiscard]] const std::string &get_bisimulation_type() const noexcept;

    /**
     * \brief Retrieves the heuristic to be used.
     * \return The heuristic type as a string.
     */
    [[nodiscard]] const std::string& get_heuristic() const noexcept;

    /**
     * \brief Retrieves the search strategy to be used.
     * \return The search strategy as a string.
     */
    [[nodiscard]] const std::string& get_search_strategy() const noexcept;

    /**
     * \brief Checks if visited states should be checked during planning.
     * \return true if visited state checking is enabled, false otherwise.
     */
    [[nodiscard]] bool get_check_visited() const noexcept;


    /**
     * \brief Returns the log file path to use if logging is enabled.
     * \return The log file path (empty if logging is not enabled).
     */
    [[nodiscard]] const std::string& get_log_file_path() const noexcept;

    friend class Configuration;

};
