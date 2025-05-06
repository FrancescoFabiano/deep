/**
 * \brief Class used to store an action and all its information.
 * 
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 3, 2019
 */
#pragma once

#include <set>
#include <vector>

#include "proposition.h"

#include "../utilities/define.h"
#include "../formulae/belief_formula.h"
#include "../domain/grounder.h"

class action
{
private:
    /**
     * \brief The name of *this*.
     */
    std::string m_name;
    /**
     * \brief The unique id of *this*.
     * 
     * This is calculated with \ref grounder.
     */
    action_id m_id;
    /** \brief The agent that execute the action.     */
    agent m_executor;
    /**
     * \brief The \ref proposition_type of the of *this*.
     * 
     * The type is initially NOTSET and then it's updated during the reading of the input file.
     */
    proposition_type m_type = NOTSET;

    /**\brief The executability conditions of *this*.*/
    formula_list m_executability;
    /**\brief The fully_observant frame (each obsv is related to a condition that's why  map) of *this*.*/
    observability_map m_fully_observants;
    /**\brief The partially_observant frame (each obsv is related to a condition that's why  map) of *this*.*/
    observability_map m_partially_observants;

    /**\brief The effects of *this* with the respective conditions.
     *
     * If \ref ONTIC then <\ref fluent_set (deterministic not or),  \ref belief_formula>
     * If \ref SENSING then <\ref fluent, \ref belief_formula>
     * If \ref ANNOUNCEMENT then <\ref fluent (literal), \ref belief_formula>.*/
    effects_map m_effects;


    /* \brief Function that adds an executability condition to *this*.
     *
     * @param[in] to_add: The \ref belief_formula that represents the executability condition to add.
     */
    void add_executability(const belief_formula & to_add);

    /* \brief Function that adds an effect (with its conditions) to *this*.
     *
     * @param[in] to_add: The \ref fluent_formula that represents the effect to add.
     * @param[in] condition: The condition of \p to_add.
     */
    void add_effect(const fluent_formula& to_add, const belief_formula& condition);
    /* \brief Function that adds a fully_observant \ref agent (with its conditions) to *this*.
     *
     * @param[in] ag: The \ref agent that its fully observant of this if \p condition.
     * @param[in] condition: The condition for \p ag to be fully_observant of *this*.*/
    void add_fully_observant(agent ag, const belief_formula& condition);
    /* \brief Function that adds a fully_observant \ref agent (with its conditions) to *this*.
     *
     * @param[in] ag: The \ref agent that its fully observant of this if \p condition.
     * @param[in] condition: The condition for \p ag to be fully_observant of *this*.*/
    void add_partially_observant(agent ag, const belief_formula& condition);

public:
    /**\brief Constructor without parameters.*/
    action();
    /**\brief Constructor with a given name and id.
     *
     * @param[in] name: the value to assign to \ref m_name.
     * @param[in] id: the value to assign to \ref m_id.*/
    action(const std::string & name, action_id id);

    /* \brief Getter of the field \ref m_name.*/
    std::string get_name() const;
    /* \brief Setter of the field \ref m_name.
     *
     * @param[in] name: the value to assign to \ref m_name.*/
    void set_name(const std::string & name);

    /* \brief Getter of the field \ref m_executor.*/
    agent get_executor() const;
    /* \brief Setter of the field \ref m_executor.
     *
     * @param[in] executor: the value to assign to \ref m_executor.*/
    void set_executor(agent executor);

    /* \brief Getter of the field \ref m_id.*/

    action_id get_id() const;
    /* \brief Setter of the field \ref m_id.
     *
     * @param[in] id: the value to assign to \ref m_id.*/
    void set_id(action_id id);

    /* \brief Getter of the field \ref m_type.*/
    const proposition_type get_type() const;
    /* \brief Setter of the field \ref m_type.
     *
     * @param[in] type: the value to assign to \ref m_type.*/
    void set_type(proposition_type type);


    /* \brief Getter of the field \ref m_executability.
     *
     * The return type it's fine because actions survive with the domain.*/
    const formula_list& get_executability() const;
    /* \brief Getter of the field \ref m_effects.
     *
     * The return type it's fine because actions survive with the domain.*/
    const effects_map& get_effects() const;
    /* \brief Getter of the field \ref m_fully_observants.
     *
     * The return type it's fine because actions survive with the domain.*/
    const observability_map& get_fully_observants() const;
    /* \brief Getter of the field \ref m_partially_observants.
     *
     * The return type it's fine because actions survive with the domain.*/
    const observability_map& get_partially_observants() const;



    /* \brief Function that parse a \ref proposition and adds its information to *this*.
     * 
     * This function uses \ref add_executability(const belief_formula &), \ref  add_effect(const fluent_formula&, const belief_formula&),
     * \ref add_fully_observant(agent, const fluent_formula&) and add_partially_observant(agent, const fluent_formula&) to add
     * the appropriate behavior to the *this*.
     *
     * @param[in] to_add: The \ref proposition to add to *this*.*/
    void add_proposition(proposition & to_add);

    /* \brief Function that  prints *this*.*/
    void print() const;

    /* \brief Operator < implmented to use \ref action in std::set.*/
    bool operator<(const action&) const;
    /* \brief Operator =.*/
    bool operator=(const action&);
};

typedef std::set<action> action_set; /**< \brief A representation a set of \ref action.*/

typedef std::vector<action> action_list; /**< \brief A representation of a sequential executution of \ref action*/
