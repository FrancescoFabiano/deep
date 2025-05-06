/**
 * \class asp_maker
 * \brief Class used to transform the EFP input files in one suitable for the asp solver.
 *
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date March 24, 2020
 */

#pragma once
#include "../domain/domain.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <regex>

class asp_maker
{
private:

    //    std::set<std::string> m_already_printed_formulae;
    //    std::set<std::string> m_already_printed_init_sf;
    std::set<std::string> m_already_printed_predicate;
    std::set<std::string> m_already_printed_agents_sets;
    grounder m_grounder;
    std::string m_all_agents = "";


    std::string print_ag_set(const agent_set & ags, std::ofstream & to_print);

    std::string print_subformula(const fluent & fl, std::ofstream & to_print, const std::string & predicate);
    std::string print_subformula(const fluent_set & fs, std::ofstream & to_print, const std::string & predicate);
    std::string print_subformula(const fluent_formula & ff, std::ofstream & to_print, const std::string & predicate);
    std::string print_subformula(const belief_formula & bf, std::ofstream & to_print, const std::string & predicate);
    std::string print_subformula(const formula_list & fl, std::ofstream & to_print, const std::string & predicate);

    //std::string print_subformula_com_kno_ff_ini(const belief_formula & bf, std::ofstream & to_print);


    void print_fluents(std::ofstream & to_print);
    void print_actions(std::ofstream & to_print);
    void print_agents(std::ofstream & to_print) const;
    void print_initially(std::ofstream & to_print);
    void print_goals(std::ofstream & to_print);

    void print_agent_set(std::ofstream & to_print) const;
    void print_all_fluent_set(fluent_set& permutation, unsigned int index, int & permutation_number, std::ofstream & to_print) const;
    void print_worlds(std::ofstream & to_print) const;

    //void minus_set(agent_set & to_modify, const agent_set & factor2) const;


public:
    asp_maker();
    void print_all();
};