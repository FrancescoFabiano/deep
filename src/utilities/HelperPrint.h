#pragma once

#include <iostream>
#include "Define.h"
#include "domain/Grounder.h"


/**
 * \class HelperPrint
 * \brief Singleton class to facilitate printing of domain structures.
 *
 * \details Prints \ref string_set, \ref string_set_set, and other domain-related sets.
 *          Only std::string representations are printed for clarity.
 *          Use \ref get_instance() to access the singleton.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 16, 2025
 */
class HelperPrint
{

public:
    /// \brief Deleted copy constructor (singleton).
    HelperPrint(const HelperPrint&) = delete;
    /// \brief Deleted copy assignment (singleton).
    HelperPrint& operator=(const HelperPrint&) = delete;
    /// \brief Deleted move constructor (singleton).
    HelperPrint(HelperPrint&&) = delete;
    /// \brief Deleted move assignment (singleton).
    HelperPrint& operator=(HelperPrint&&) = delete;

    /**
     * \brief Get the singleton instance.
     * \return Reference to the singleton HelperPrint.
     */
    [[nodiscard]]
    static HelperPrint& get_instance();

    /**
     * \brief Set the grounder used for de-grounding fluents.
     * \param gr The grounder to set.
     */
    void set_grounder(const Grounder& gr);

    /**
     * \brief Print all std::string in a set (conjunctive set of fluents).
     * \param to_print The set to print.
     * \param os The output stream to print to (default: std::cout).
     */
    static void print_list(const StringsSet& to_print, std::ostream& os = std::cout);

    /**
     * \brief Print all std::string sets in a set (DNF formula).
     * \param to_print The set of sets to print.
     * \param os The output stream to print to (default: std::cout).
     */
    static void print_list(const StringSetsSet& to_print, std::ostream& os = std::cout);

    /**
     * \brief Print all fluents in a set (conjunctive set).
     * \param to_print The set to print.
     * \param os The output stream to print to (default: std::cout).
     */
    void print_list(const FluentsSet& to_print, std::ostream& os = std::cout) const;

    /**
     * \brief Print all fluent sets in a formula (DNF).
     * \param to_print The formula to print.
     * \param os The output stream to print to (default: std::cout).
     */
    void print_list(const FluentFormula& to_print, std::ostream& os = std::cout) const;

    /**
     * \brief Print all belief formulas in a list (CNF).
     * \param to_print The list to print.
     * \param os The output stream to print to (default: std::cout).
     */
    static void print_list(const FormulaeList& to_print, std::ostream& os = std::cout);

    /**
     * \brief Print all KripkeWorld pointers in a set.
     * \param to_print The set to print.
     * \param os The output stream to print to (default: std::cout).
     */
    static void print_list(const KripkeWorldPointersSet& to_print, std::ostream& os = std::cout);

    /**
     * \brief Print all action names in a list.
     * \param to_print The list to print.
     * \param os The output stream to print to (default: std::cout).
     */
    void print_list(const ActionIdsList& to_print, std::ostream& os = std::cout) const;

    /**
     * \brief Print all agent names in a set.
     * \param to_print The set to print.
     * \param os The output stream to print to (default: std::cout).
     */
    void print_list_ag(const AgentsSet& to_print, std::ostream& os = std::cout) const;

    /**
     * \brief Print all agent names in a set.
     * \param to_print The BeliefFormula to print.
     * \param os The output stream to print to (default: std::cout).
     */
    void print_belief_formula(const BeliefFormula& to_print, std::ostream& os = std::cout) const;

    /**
     * \brief Generate a log file path based on domain name, date, time, and repetition.
     * \param domain_file The domain file name.
     * \return The generated log file path.
     */
    static std::string generate_log_file_path(const std::string& domain_file);

    /**
     * \brief Print a KripkeState in a human-readable format.
     * \param kstate The KripkeState to print.
     * \param os The output stream to print to (default: std::cout).
     */
    void print_KripkeState(const KripkeState &kstate, std::ostream &os) const;

    /**
     * \brief Print a KripkeState in DOT format for graph visualization.
     * \param kstate The KripkeState to print.
     * \param os The output stream to print to (default: std::cout).
     */
    void print_KripkeStateDot(const KripkeState &kstate, std::ostream &os) const;

private:

    static HelperPrint* instance; ///< Singleton instance
    Grounder m_grounder;           ///< Used to de-ground fluents for printing.
    bool m_set_grounder = false;///< True if \ref m_grounder has been set.

    /// \brief Private constructor for singleton pattern.
    HelperPrint();
};
