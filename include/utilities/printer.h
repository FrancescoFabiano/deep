/**
 * \brief Singleton class used to facilitate the printing.
 *
 * \details   Class used to print \ref string_set and \ref string_set_set.
 *            Only std::string are printed because are more informative than their grounded counterpart,
 *            e.g., if we want to print a set of fluents (that maybe represents a possible world) we print out its std::string
 *            description and not the int one (that is the grounded one).
 * @see grounder.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 7, 2019
 */

#pragma once
#include <fstream>

#include "define.h"

#include "../domain/grounder.h"
#include "../formulae/belief_formula.h"
#include "../states/possibilities/pworld.h"

class printer
{
private:

    /** \brief A \ref grounder to deground the \ref fluent to print.*/
    grounder m_grounder;
    /** \brief A check if \ref m_grounder has been setted.*/
    bool setted_grounder;

    /** \brief Private constructor since it is a Singleton class.*/
    printer();

public:

    /** \brief To get always (the same instance of) *this* and the same instantiated fields.*/
    static printer& get_instance();

    /** \brief Setter of the field \ref m_grounder.
     *
     * @param[in] gr: the \ref grounder to set as \ref m_grounder.
     *
     * \todo   Is the const reference the best type of parameter passing?*/
    void set_grounder(const grounder & gr);

    /** \brief Function used to print all the std::string inside \p to_print.
     *
     * A \ref string_set represents a conjunctive set of \ref fluent format so the printing is executed accordingly.
     *
     * @param[in] to_print: the \ref string_set to be printed.
     *
     * \todo   Is the const reference the best type of parameter passing?*/
    void print_list(const string_set& to_print) const;

    /** \brief Function used to print all the std::string inside \p to_print.
     *
     * A \ref string_set_set represents a formula in DNF format so the printing is executed accordingly.
     *
     * @param[in] to_print: the \ref string_set_set to be printed.
     *
     *  \todo   Is the const reference the best type of parameter passing?*/
    void print_list(const string_set_set& to_print) const;

    /** \brief Function used to print all the \ref fluent inside \p to_print.
     *
     * A \ref fluent_set represents a conjunctive set of \ref fluent format so the printing is executed accordingly.
     *
     * @param[in] to_print: the \ref fluent_set to be printed.
     *
     * \todo   Is the const reference the best type of parameter passing?*/
    void print_list(const fluent_set& to_print) const;

    /** \brief Function used to print all the \ref fluent inside \p to_print.
     *
     * A \ref fluent_formula represents a formula in DNF format so the printing is executed accordingly.
     *
     * @param[in] to_print: the \ref fluent_formula to be printed.*/
    void print_list(const fluent_formula& to_print) const;

    /** \brief Function used to print all the \ref belief_formula inside \p to_print.
     *
     * A \ref formula_list represents a formula in CNF format so the printing is executed accordingly.
     *
     * @param[in] to_print: the \ref formula_list to be printed.*/
    void print_list(const formula_list& to_print) const;

    /** \brief Function used to print all the \ref kworld_ptr inside \p to_print.
     *
     * @param[in] to_print: the \ref kworld_ptr_set to be printed.*/
    //  void print_list(const kworld_ptr_set& to_print);

    /** \brief Function used to print all the \ref kworld_ptr inside \p to_print.
     * 
     * @param[in] to_print: the \ref kworld_ptr_set to be printed.*/
    void print_list(const pworld_ptr_set& to_print) const;


    /** \brief Function used to print all the \ref action name inside \p to_print.
     * 
     * @param[in] to_print: the \ref action_id_list to be printed.*/
    void print_list(const action_id_list& to_print) const;

    /** \brief Function used to print all the \ref agent name inside \p to_print.
     * 
     * @param[in] to_print: the \ref agent_set to be printed.*/
    void print_list_ag(const agent_set& to_print) const;

    /** \brief Function used to print the map of the transition function for attitudes.
     * 
     * @param[in] to_print: the \ref transition_map_att to be printed.*/
    void print_map_att(const transition_map_att& to_print) const;

    /** \brief Copy constructor removed since is Singleton class. */
    printer(printer const&) = delete;
    /** \brief Copy operator removed since Singleton class. */
    void operator=(printer const&) = delete;
};
