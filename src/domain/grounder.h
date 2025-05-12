/**
 * \class grounder
 * \brief Class that grounds all the string of the domain in their numerical id.
 * 
 *  - \ref action are associated to an \ref action_id;
 *  - agents (string) are associated to a \ref agent;
 *  - fluents (string) are associated to a \ref fluent;
 *
 * \see define, action.
 * 
 * \todo Singleton class?.
 * 
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 5, 2019
 */
#pragma once

#include "../utilities/define.h"
#include <string>


//@TODO: Double definition watch out
//typedef unsigned short action_id;

class grounder
{
private:
    /** \brief A map that links each \ref fluent name to its grounded value.*/
    fluent_map m_fluent_map;

    /** \brief A map that links each \ref agent name to its grounded value.*/
    agent_map m_agent_map;

    /** \brief A map that links each \ref action name to its grounded value.*/
    action_name_map m_action_name_map;

    /** \brief A map that links each \ref fluent grounded value to its name (printing value)*/
    reverse_fluent_map r_fluent_map;
    /** \brief A map that links each \ref agent grounded value to its name (printing value)*/
    reverse_agent_map r_agent_map;
    /** \brief A map that links each \ref action grounded value to its name (printing value)*/
    reverse_action_name_map r_action_name_map;

    /* \brief Boolean to check if the \ref r_fluent_map has been created.
     *
     * printing function that is why we don't always create it, only in debug mode.
    bool m_reversed_fl = false;
    \brief Boolean to check if the \ref r_agent_map has been created.
     *
     * printing function that is why we don't always create it, only in debug mode.
    bool m_reversed_ag = false;
    \brief Boolean to check if the \ref r_action_name_map has been created.
     *
     * printing function that is why we don't always create it, only in debug mode.
    bool m_reversed_ac = false;*/

    /** \brief Function that reverses \ref m_fluent_map.*/
    void create_reverse_fl();

    /** \brief Function that reverses \ref m_agent_map.*/
    void create_reverse_ag();

    /** \brief Function that reverses \ref m_action_name_map.*/
    void create_reverse_ac();
public:

    /** \brief Constructor without parameters.*/
    grounder();
    /** \brief Constructor with parameters.
     *
     * @param[in] given_fluent_map: the map to copy into \ref m_fluent_map.
     * @param[in] given_agent_map: the map to copy into \ref m_agent_map.
     * @param[in] given_action_name_map: the map to copy into \ref m_action_name_map.
     * 
     * \todo Is parameter passing the best one? Is maybe better to use const reference?
     */
    grounder(const fluent_map & given_fluent_map, const agent_map & given_agent_map, const action_name_map & given_action_name_map);

    /** \brief Setter for the parameter \ref m_fluent_map.
     *
     * @param[in] given_fluent_map: the map to copy into \ref m_fluent_map.
     *
     * \todo Is parameter passing the best one? Is maybe better to use const reference?*/
    void set_fluent_map(const fluent_map & given_fluent_map);
    /** \brief Setter for the parameter \ref m_agent_map.
     *
     * @param[in] given_agent_map: the map to copy into \ref m_agent_map.
     * 
     * \todo Is parameter passing the best one? Is maybe better to use const reference?*/
    void set_agent_map(const agent_map & given_agent_map);
    /** \brief Setter for the parameter \ref m_action_name_map.
     *
     * @param[in] given_action_name_map: the map to copy into \ref m_action_name_map.
     * 
     * \todo Is parameter passing the best one? Is maybe better to use const reference?*/
    void set_action_name_map(const action_name_map & given_action_name_map);

    /** \brief Getter for the parameter \ref m_fluent_map.
     *
     * @return the map \ref m_fluent_map.
     *
     * \todo Is return the best one? Is maybe better to use const reference?*/
    const fluent_map & get_fluent_map() const;
    /** \brief Getter for the parameter \ref m_agent_map.
     *
     * @return the map \ref m_agent_map.
     *
     * \todo Is return the best one? Is maybe better to use const reference?*/
    const agent_map & get_agent_map() const;
    /** \brief Getter for the parameter \ref m_action_name_map.
     *
     * @return the map \ref m_action_name_map.
     *
     * \todo Is return the best one? Is maybe better to use const reference?*/
    const action_name_map & get_action_name_map() const;

    /** \brief Function that given a \ref fluent name returns its grounded value.
     * 
     * The function reads the field \ref m_fluent_map.
     *
     * @param[in] to_ground: the \ref fluent name to ground.
     * @return: the \p to_ground grounded value.*/
    fluent ground_fluent(const std::string& to_ground) const;
    /** \brief Function that given a \ref fluent set returns its grounded value.
     *
     * This function calls recursively \ref ground_fluent(const std::string&) const.
     * 
     * @param[in] to_ground: the set of \ref fluent name to ground.
     * @return: \p to_ground grounded.
     * 
     * \todo Is the return type the best one? Is maybe better to use const reference?*/
    fluent_set ground_fluent(const string_set& to_ground) const;
    /** \brief Function that given a \ref string_set_set returns its grounded value.
     *
     * This function calls recursively \ref ground_fluent(const string_set&) const.
     * 
     * @param[in] to_ground: the set of set of \ref fluent name to ground.
     * @return: \p to_ground grounded.
     * 
     * \todo Is the return type the best one? Is maybe better to use const reference?*/
    fluent_formula ground_fluent(const string_set_set& to_ground) const;

    /** \brief Function that given an \ref agent name returns its grounded value.
     * 
     * The function reads the field \ref m_agent_map.
     *
     * @param[in] to_ground: the \ref agent name to ground.
     * @return: the \p to_ground grounded value.*/
    agent ground_agent(const std::string& to_ground) const;
    /** \brief Function that given an \ref agent set returns its grounded value.
     * 
     * This function calls recursively \ref ground_agent(const std::string&) const.
     *
     * @param[in] to_ground: the set of \ref agent name to ground.
     * @return: the \p to_ground grounded value.*/
    agent_set ground_agent(const string_set& to_ground) const;


    /** \brief Function that given an \ref action name returns its grounded value.
     * 
     * The function reads the field \ref m_action_name_map.
     *
     * @param[in] to_ground: the \ref action name to ground.
     * @return: the \p to_ground grounded value.*/
    action_id ground_action(const std::string& to_ground) const;

    /** \brief Function that given a \ref fluent grounded value returns its name. 
     * 
     * The function reads the field \ref r_fluent_map.
     *
     * @param[in] to_deground: the \ref fluent name to deground.
     * @return: the name of \p to_deground.*/
    std::string deground_fluent(fluent to_deground) const;
    /** \brief Function that given a set of \ref fluent grounded value returns the set with their names. 
     * 
     * The function uses \ref deground_fluent(fluent) const.
     *
     * @param[in] to_deground: the set of \ref fluent names to deground.
     * @return: the set with the name of the element of \p to_deground.*/
    string_set deground_fluent(const fluent_set& to_deground) const;
    /** \brief Function that given a grounded \ref fluent_formula returns the same formula in string format. 
     * 
     * The function uses \ref deground_fluent(const fluent_set &) const.
     *
     * @param[in] to_deground: the \ref fluent_formula to deground.
     * @return: the \ref fluent_formula in string format.*/
    string_set_set deground_fluent(const fluent_formula& to_deground) const;

    /** \brief Function that given an \ref agent grounded value returns its name. 
     * 
     * The function reads the field \ref r_agent_map.
     *
     * @param[in] to_deground: the \ref agent name to deground.
     * @return: the name of \p to_deground.*/
    std::string deground_agent(agent to_deground) const;

    /** \brief Function that given an \ref agent_set grounded value returns the set of names. 
     * 
     * The function reads the field \ref r_agent_map.
     *
     * @param[in] to_deground: the \ref agent_set names to deground.
     * @return: the set with the names of \p to_deground.*/
    string_set deground_agents(const agent_set & to_deground) const;

    /** \brief Function that given an \ref action grounded value returns its name. 
     * 
     * The function reads the field \ref r_action_name_map.
     *
     * @param[in] to_deground: the \ref action name to deground.
     * @return: the name of \p to_deground.*/
    std::string deground_action(action_id to_deground) const;

    /* \brief Function that print a \ref fluent_set in string format. 
     *
     * @param[in] to_print: the \ref fluent_set to print.
     * void print_ff(const fluent_set& to_print) const;
     \brief Function that print a \ref fluent_formula in string format. 
     *
     * @param[in] to_print: the \ref fluent_formula to print.
    void print_ff(const fluent_formula& to_print) const; */

    /**\brief The copy operator =.
     * 
     * @param[in] to_copy: the \ref grounder to assign to *this*.
     * @return true: if the assignment went through.
     * @return false: otherwise.*/
    bool operator=(const grounder& to_copy);

};
