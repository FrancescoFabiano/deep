#pragma once

#include <chrono>

#include "Define.h"
#include "domain/Grounder.h"

class BeliefFormulaParsed;
/**
 * \class HelperPrint
 * \brief Singleton class to facilitate printing of domain structures.
 *
 * \details Prints \ref string_set, \ref string_set_set, and other
 * domain-related sets. Only std::string representations are printed for
 * clarity. Use \ref get_instance() to access the singleton.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 16, 2025
 */
class HelperPrint {
public:
  /// \brief Deleted copy constructor (singleton).
  HelperPrint(const HelperPrint &) = delete;
  /// \brief Deleted copy assignment (singleton).
  HelperPrint &operator=(const HelperPrint &) = delete;
  /// \brief Deleted move constructor (singleton).
  HelperPrint(HelperPrint &&) = delete;
  /// \brief Deleted move assignment (singleton).
  HelperPrint &operator=(HelperPrint &&) = delete;

  /**
   * \brief Get the singleton instance.
   * \return Reference to the singleton HelperPrint.
   */
  [[nodiscard]]
  static HelperPrint &get_instance();

  /**
   * \brief Set the grounder used for de-grounding fluents.
   * \param gr The grounder to set.
   */
  void set_grounder(const Grounder &gr);

  /**
   * \brief Get the grounder used for de-grounding fluents.
   * \return Reference to the grounder.
   */
  const Grounder &get_grounder() const;

  /**
   * \brief Print all std::string in a set (conjunctive set of fluents).
   * \param to_print The set to print.
   */
  static void print_list(const StringsSet &to_print);

  /**
   * \brief Print all std::string sets in a set (DNF formula).
   * \param to_print The set of sets to print.
   */
  static void print_list(const StringSetsSet &to_print);

  /**
   * \brief Print all fluents in a set (conjunctive set).
   * \param to_print The set to print.
   */
  void print_list(const FluentsSet &to_print) const;

  /**
   * \brief Print all fluent sets in a formula (DNF).
   * \param to_print The formula to print.
   */
  void print_list(const FluentFormula &to_print) const;

  /**
   * \brief Print all belief formulas in a list (CNF).
   * \param to_print The list to print.
   */
  static void print_list(const FormulaeList &to_print);

  /**
   * \brief Print all KripkeWorld pointers in a set.
   * \param to_print The set to print.
   */
  static void print_list(const KripkeWorldPointersSet &to_print);

  /**
   * \brief Print all action names in a list.
   * \param to_print The list to print.
   */
  void print_list(const ActionIdsList &to_print) const;

  /**
   * \brief Print all agent names in a set.
   * \param to_print The set to print.
   */
  void print_list_ag(const AgentsSet &to_print) const;

  /**
   * \brief Print the parsed bf.
   * \param to_print The BeliefFormulaParsed to print.
   */
  static void print_belief_formula_parsed(const BeliefFormulaParsed &to_print);

  /**
   * \brief Print a belief formula using grounder.
   * \param to_print The BeliefFormula to print.
   */
  void print_belief_formula(const BeliefFormula &to_print) const;

  /**
   * \brief Generate a log file path based on domain name, date, time, and
   * repetition. \param domain_file The domain file name. \return The generated
   * log file path.
   */
  static std::string generate_log_file_path(const std::string &domain_file);

  /**
   * \brief Print a KripkeState in a human-readable format.
   * \param kstate The KripkeState to print.
   */
  void print_state(const KripkeState &kstate) const;

  /**
   * \brief Print a KripkeState in DOT format for graph visualization.
   * \param kstate The KripkeState to print.
   * \param ofs The output stream to write the DOT format to. This must be a
   * file.
   */
  void print_dot_format(const KripkeState &kstate, std::ofstream &ofs) const;

  /**
   * \brief Print a KripkeState in format for training the GNN.
   * \param kstate The KripkeState to print.
   * \param ofs The output stream to write the DOT format to. This must be a
   * file. \param use_hash Flag to indicate whether to use hash IDs or adjusted
   * IDs (needed because sometime both is set) \param is_merged Flag to indicate
   * if each dataset entry is merged <goal,state> or not.
   */
  static void print_dataset_format(const KripkeState &kstate,
                                   std::ofstream &ofs, const bool use_hash,
                                   const bool is_merged);

  /**
   * \brief Reads a sequence of actions from a file.
   * \details The file should contain actions separated by spaces or commas.
   * Returns a vector of action strings. Throws an error via ExitHandler if the
   * file cannot be opened or contains malformed content. \param filename The
   * file to read actions from. \return Vector of action strings.
   */
  static std::vector<std::string>
  read_actions_from_file(const std::string &filename);

  /**
   * \brief Prints the time for a specific task following the output format.
   * \param task The task name to print before the duration.
   * \param duration The duration of the task.
   */
  static void print_time(const std::string &task,
                         const std::chrono::duration<double> &duration);

  /**
   * \brief Pretty print a duration in a human-readable format.
   * \details Converts the given duration to a string with appropriate time
   * units (e.g., seconds, milliseconds). \param duration The duration to pretty
   * print. \return A string representing the duration in a human-readable
   * format.
   */
  static std::string
  pretty_print_duration(const std::chrono::duration<double> &duration);

private:
  Grounder m_grounder;         ///< Used to de-ground fluents for printing.
  bool m_set_grounder = false; ///< True if \ref m_grounder has been set.

  /// \brief Private constructor for singleton pattern.
  HelperPrint();
};
