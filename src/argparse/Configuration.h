/**
* \class Configuration
 * \brief Singleton class to store and access configuration options, inheriting from ArgumentParser.
 *
 * The Configuration class provides a singleton interface for storing and accessing
 * configuration options parsed from the command line. It inherits search based fields from ArgumentParser
 * and provides public getters and setters for each configuration option.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 30, 2025
 */

#pragma once

#include <string>
#include <ostream>


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
     * \brief Gets the log_file_path.
     * \return The log_file_path string.
     */
    [[nodiscard]] const std::string& get_log_file_path() const noexcept;
    /**
     * \brief Sets the log_file_path.
     * \param val The log_file_path  string.
     */
    void set_log_file_path(const std::string& val);

    /**
    * \brief Sets the field of the class to value
    * \param field The name of the field to set (based on the parsing from command line)
    * \param value The value to set the field to.
    */
    void set_field_by_name(const std::string& field, const std::string& value);

    /**
     * \brief Prints all configuration values to the given output stream.
     * \param os The output stream to print to.
     */
    void print(std::ostream& os) const;

private:
    /**
     * \brief Private constructor for singleton pattern.
     */
    Configuration();

    // Singleton instance (thread-local)
    static thread_local Configuration* instance;

    // Configuration fields
    bool m_bisimulation = false; ///< Bisimulation enabled flag.
    std::string m_bisimulation_type = "FB"; ///< Bisimulation type string.
    bool m_bisimulation_type_bool = true; ///< Bisimulation type as boolean.
    bool m_check_visited = false; ///< Visited state checking flag.
    std::string m_search_strategy = "BFS"; ///< Search strategy string.
    std::string m_heuristic_opt = "SUBGOALS"; ///< Heuristic option string.
    std::string m_log_file_path; ///< The log file path if logging is enabled. Here to allow multiple files for threads management.

    /**
     * \brief Sets the bisimulation type as a boolean.
     */
    void set_bisimulation_type_bool();
};
