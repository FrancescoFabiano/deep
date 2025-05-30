/**
* \class Configuration
 * \brief Singleton class to store and access configuration options, inheriting from ArgumentParser.
 *
 * The Configuration class provides a singleton interface for storing and accessing
 * configuration options parsed from the command line. It inherits all fields from ArgumentParser
 * and provides public getters and setters for each configuration option.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 30, 2025
 */

#pragma once

#include <string>
#include <vector>
#include "argparse/ArgumentParser.h"


class Configuration {
public:
    /**
     * \brief Creates the singleton instance of Configuration by copying values from ArgumentParser.
     *
     * This method explicitly creates the singleton instance of Configuration
     * by copying all relevant values from the ArgumentParser singleton.
     * It should be called once after ArgumentParser is initialized.
     */
    static void create_instance();

    /**
     * \brief Returns the singleton instance of Configuration.
     *
     * This method returns a single instance of Configuration,
     * ensuring the application only has one configuration instance.
     *
     * \return Reference to the singleton instance.
     */
    static Configuration& get_instance();

    /** \brief Copy constructor removed since this is a Singleton class. */
    Configuration(const Configuration&) = delete;
    /** \brief Copy operator removed since this is a Singleton class. */
    Configuration& operator=(const Configuration&) = delete;

    // Getters and setters for all configuration fields

    /**
     * \brief Checks if debug mode is enabled.
     * \return true if debug mode is enabled, false otherwise.
     */
    [[nodiscard]] bool get_debug() const noexcept;
    /**
     * \brief Sets debug mode.
     * \param val true to enable debug mode, false to disable.
     */
    void set_debug(const std::string& val);
    void set_debug(bool val);

    /**
     * \brief Checks if bisimulation is used.
     * \return true if bisimulation is used, false otherwise.
     */
    [[nodiscard]] bool get_bisimulation() const noexcept;
    /**
     * \brief Sets bisimulation usage.
     * \param val true to enable bisimulation, false to disable.
     */
    void set_bisimulation(const std::string& val);
    void set_bisimulation(bool val);

    /**
     * \brief Gets the bisimulation type.
     * \return The bisimulation type string.
     */
    [[nodiscard]] const std::string& get_bisimulation_type() const noexcept;
    /**
     * \brief Sets the bisimulation type.
     * \param val The bisimulation type string.
     */
    void set_bisimulation_type(const std::string& val);

    /**
     * \brief Gets the bisimulation type as a boolean.
     * \return True if PT, false if FB.
     */
    [[nodiscard]] bool get_bisimulation_type_bool() const noexcept;
    /**
     * \brief Sets the bisimulation type as a boolean.
     * \param val True for PT, false for FB.
     */
    void set_bisimulation_type_bool(const std::string& val);
    void set_bisimulation_type_bool(bool val);

    /**
     * \brief Checks if visited state checking is enabled.
     * \return true if enabled, false otherwise.
     */
    [[nodiscard]] bool get_check_visited() const noexcept;
    /**
     * \brief Sets visited state checking.
     * \param val true to enable, false to disable.
     */
    void set_check_visited(const std::string& val);
    void set_check_visited(bool val);

    /**
     * \brief Checks if dataset mode is enabled.
     * \return true if enabled, false otherwise.
     */
    [[nodiscard]] bool get_dataset_mode() const noexcept;
    /**
     * \brief Sets dataset mode.
     * \param val true to enable, false to disable.
     */
    void set_dataset_mode(const std::string& val);
    void set_dataset_mode(bool val);

    /**
     * \brief Gets the dataset depth.
     * \return The dataset depth.
     */
    [[nodiscard]] int get_dataset_depth() const noexcept;
    /**
     * \brief Sets the dataset depth.
     * \param val The dataset depth.
     */
    void set_dataset_depth(const std::string& val);
    void set_dataset_depth(int val);

    /**
     * \brief Gets the search strategy.
     * \return The search strategy string.
     */
    [[nodiscard]] const std::string& get_search_strategy() const noexcept;
    /**
     * \brief Sets the search strategy.
     * \param val The search strategy string.
     */
    void set_search_strategy(const std::string& val);

    /**
     * \brief Gets the heuristic option.
     * \return The heuristic option string.
     */
    [[nodiscard]] const std::string& get_heuristic_opt() const noexcept;
    /**
     * \brief Sets the heuristic option.
     * \param val The heuristic option string.
     */
    void set_heuristic_opt(const std::string& val);

    /**
     * \brief Gets the parallel type.
     * \return The parallel type string.
     */
    [[nodiscard]] const std::string& get_parallel_type() const noexcept;
    /**
     * \brief Sets the parallel type.
     * \param val The parallel type string.
     */
    void set_parallel_type(const std::string& val);

    /**
     * \brief Gets the parallel wait strategy.
     * \return The parallel wait strategy string.
     */
    [[nodiscard]] const std::string& get_parallel_wait() const noexcept;
    /**
     * \brief Sets the parallel wait strategy.
     * \param val The parallel wait strategy string.
     */
    void set_parallel_wait(const std::string& val);

    /**
     * \brief Checks if plan execution is enabled.
     * \return true if enabled, false otherwise.
     */
    [[nodiscard]] bool get_exec_plan() const noexcept;
    /**
     * \brief Sets plan execution.
     * \param val true to enable, false to disable.
     */
    void set_exec_plan(const std::string& val);
    void set_exec_plan(bool val);

    /**
     * \brief Gets the actions to execute.
     * \return Vector of action strings.
     */
    [[nodiscard]] const std::vector<std::string>& get_exec_actions() const noexcept;
    /**
     * \brief Sets the actions to execute.
     * \param val Comma-separated string of actions.
     */
    void set_exec_actions(const std::string& val);
    /**
     * \brief Sets the actions to execute.
     * \param val Vector of action strings.
     */
    void set_exec_actions(const std::vector<std::string>& val);

    /**
     * \brief Checks if output results file is enabled.
     * \return true if enabled, false otherwise.
     */
    [[nodiscard]] bool get_output_results_file() const noexcept;
    /**
     * \brief Sets output results file flag.
     * \param val true to enable, false to disable.
     */
    void set_output_results_file(const std::string& val);
    void set_output_results_file(bool val);

    /**
     * \brief Gets the plan file path.
     * \return The plan file path string.
     */
    [[nodiscard]] const std::string& get_plan_file() const noexcept;
    /**
     * \brief Sets the plan file path.
     * \param val The plan file path string.
     */
    void set_plan_file(const std::string& val);

    /**
     * \brief Checks if logging is enabled.
     * \return true if enabled, false otherwise.
     */
    [[nodiscard]] bool get_log_enabled() const noexcept;
    /**
     * \brief Sets logging enabled flag.
     * \param val true to enable, false to disable.
     */
    void set_log_enabled(const std::string& val);
    void set_log_enabled(bool val);

    /**
     * \brief Gets the log file path.
     * \return The log file path string.
     */
    [[nodiscard]] const std::string& get_log_file_path() const noexcept;
    /**
     * \brief Sets the log file path.
     * \param val The log file path string.
     */
    void set_log_file_path(const std::string& val);

    /**
     * \brief Gets the input file path.
     * \return The input file path string.
     */
    [[nodiscard]] const std::string& get_input_file() const noexcept;
    /**
     * \brief Sets the input file path.
     * \param val The input file path string.
     */
    void set_input_file(const std::string& val);

private:
    /**
     * \brief Private constructor for singleton pattern.
     */
    Configuration();

    // Singleton instance (thread-local)
    static thread_local Configuration* instance;

    // Configuration fields
    bool m_debug = false; ///< Debug mode flag.
    bool m_bisimulation = false; ///< Bisimulation enabled flag.
    std::string m_bisimulation_type = "FB"; ///< Bisimulation type string.
    bool m_bisimulation_type_bool = true; ///< Bisimulation type as boolean.
    bool m_check_visited = false; ///< Visited state checking flag.
    bool m_dataset_mode = false; ///< Dataset mode flag.
    int m_dataset_depth = 10; ///< Dataset depth.
    std::string m_search_strategy = "BFS"; ///< Search strategy string.
    std::string m_heuristic_opt = "NONE"; ///< Heuristic option string.
    std::string m_parallel_type = "PTHREAD"; ///< Parallel type string.
    std::string m_parallel_wait = "NONE"; ///< Parallel wait strategy string.
    bool m_exec_plan = false; ///< Plan execution flag.
    std::vector<std::string> m_exec_actions; ///< Actions to execute.
    bool m_output_results_file = false; ///< Output results file flag.
    std::string m_plan_file = "plan.txt"; ///< Plan file path.
    bool m_log_enabled = false; ///< Logging enabled flag.
    std::string m_log_file_path; ///< Log file path.
    std::string m_input_file; ///< Input file path.
};
