/**
 * \class Configuration
 * \brief Singleton class to store and access configuration options, inheriting
 * from ArgumentParser.
 *
 * The Configuration class provides a singleton interface for storing and
 * accessing configuration options parsed from the command line. It inherits
 * search based fields from ArgumentParser and provides public getters and
 * setters for each configuration option.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 30, 2025
 */

#pragma once

#include <ostream>
#include <string>

#include "Define.h"

class Configuration {
public:
  /**
   * \brief Returns the thread-local singleton instance of Configuration.
   *
   * Each thread gets its own instance.
   */
  static Configuration &get_instance();

  /**
   * \brief Creates the thread-local singleton instance of Configuration by
   * copying values from ArgumentParser.
   *
   * This method explicitly creates the thread-local singleton instance of
   * Configuration by copying all relevant values from the ArgumentParser
   * singleton. It should be called once per thread after ArgumentParser is
   * initialized.
   */
  static void create_instance();

  /** \brief Copy constructor removed since this is a Singleton class. */
  Configuration(const Configuration &) = delete;
  /** \brief Copy operator removed since this is a Singleton class. */
  Configuration &operator=(const Configuration &) = delete;

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
  void set_bisimulation(const std::string &val);
  void set_bisimulation(bool val);

  /**
   * \brief Gets the bisimulation type.
   * \return The bisimulation type string.
   */
  [[nodiscard]] const std::string &get_bisimulation_type() const noexcept;
  /**
   * \brief Sets the bisimulation type.
   * \param val The bisimulation type string.
   */
  void set_bisimulation_type(const std::string &val);

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
  void set_check_visited(const std::string &val);
  void set_check_visited(bool val);

  /**
   * \brief Gets the search strategy.
   * \return The search strategy enum.
   */
  [[nodiscard]] const SearchType &get_search_strategy() const noexcept;
  /**
   * \brief Sets the search strategy.
   * \param val The search strategy enum.
   */
  void set_search_strategy(const std::string &val);

  /**
   * \brief Gets the heuristic option.
   * \return The heuristic option enum.
   */
  [[nodiscard]] const Heuristics &get_heuristic_opt() const noexcept;
  /**
   * \brief Sets the heuristic option.
   * \param val The heuristic option enum.
   */
  void set_heuristic_opt(const std::string &val);

  /**
   * \brief Sets the path to the model used by the GNN heuristics.
   * \param val The path valeu.
   */
  void set_GNN_model_path(const std::string &val);

  /**
   * \brief Retrieves the path to the model used by the GNN heuristics.
   * \return The path to the model used by the GNN heuristics.
   */
  [[nodiscard]] const std::string &get_GNN_model_path() const noexcept;

  /**
   * \brief Sets the field of the class to value
   * \param field The name of the field to set (based on the parsing from
   * command line) \param value The value to set the field to.
   */
  void set_field_by_name(const std::string &field, const std::string &value);
  void set_search_strategy_enum();

  /**
   * \brief Functions that checks how many times bisimulation has failed during
   * this execution and if we are the threshold we deactivate bisimulation
   * altogether.
   */
  void add_bisimulation_failure();

  void set_heuristic_enum();
  /**
   * \brief Prints all configuration values to the given output stream.
   * \param os The output stream to print to.
   */
  void print(std::ostream &os) const;

private:
  /**
   * \brief Private constructor for singleton pattern.
   */
  Configuration();

  // Configuration fields
  bool m_bisimulation = false;            ///< Bisimulation enabled flag.
  std::string m_bisimulation_type = "FB"; ///< Bisimulation type string.
  int bisimulation_failures =
      0; ///< Counter to keep track of hwo many times Bisimulation failed so we
         ///< can deactivate it.
  bool m_bisimulation_type_bool = true;  ///< Bisimulation type as boolean.
  bool m_check_visited = false;          ///< Visited state checking flag.
  std::string m_search_strategy = "BFS"; ///< Search strategy string.
  SearchType m_search_strategy_enum =
      SearchType::BFS;                      ///< Search strategy enum.
  std::string m_heuristic_opt = "SUBGOALS"; ///< Heuristic option string.
  Heuristics m_heuristic_enum =
      Heuristics::SUBGOALS; ///< Heuristic option enum.
  std::string m_GNN_model_path =
      "lib/RL/models/GNN_model_default.pt"; ///< Path to the GNN model for
                                            ///< heuristic evaluation.

  /**
   * \brief Sets the bisimulation type as a boolean.
   */
  void set_bisimulation_type_bool();

  /**
   * \brief Thread-local flag to track initialization per thread.
   */
  static thread_local bool m_initialized;
};