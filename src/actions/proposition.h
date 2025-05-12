/**
 * \class proposition
 * \brief Class used a support to elaborate the proposition of the \ref action(s) in the domain.
 *
 * A \ref proposition identifies and specifies an \ref action behavior(executability, effects, observability).
 * For details on the types of proposition look \ref proposition_type.
 * 
 * The content of a \ref proposition is not grounded as will be grounded once is inserted in an \ref action.
 *
 * \see action.
 * 
 * \todo public fields into private and getter and setter.
 * 
 * \todo move into the action package.
 * 
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 1, 2019
 */

#pragma once

#include <list>

#include "../formulae/belief_formula.h"
#include "../utilities/define.h"
#include "../utilities/printer.h"

/**
 * \brief The possible types of proposition founded in the domain description.
 */
enum proposition_type
{
    EXECUTABILITY, /**< \brief Used when the proposition specifies an action executability condition -- *act* **exec if** *phi* */
    ONTIC, /**< \brief Used when the proposition specifies the effects of an ontic action -- *act* **causes** *f*.*/
    SENSING, /**< \brief Used when the proposition specifies the effects of a sensing action -- *act* **sensed** *f*.*/
    ANNOUNCEMENT, /**< \brief Used when the proposition specifies the effects of a announcement action -- *act* **annaunces** *ff* */
    OBSERVANCE, /**< \brief Used when the proposition specifies the full observability conditions of an action -- *ag* **observes** *act* */
    AWARENESS, /**< \brief Used when the proposition specifies the partial observability conditions of an action -- *ag* **aware** *act* */
    EXECUTOR, /**< \brief Used when the proposition specifies the executor of an action -- *ag* **executor** *act* */
    NOTSET, /**< \brief Default case.*/
    /***************DOXASTIC REASONING***************/
    LIES,
    /***************END DOXASTIC***************/
};

class proposition
{
private:
    /** The \ref proposition_type of this.*/
    proposition_type m_type;
    /** The name of this (used as id).*/
    std::string m_action_name;
    /** If *this* is \ref ONTIC, \ref SENSING or \ref ANNOUNCEMENT the effects are stored here.*/
    string_set_set m_action_effect;
    /** If *this* is \ref OBSERVANCE or \ref AWARENESS the relative \ref agent is stored here.*/
    std::string m_agent;
    /** \brief If *this* is \ref OBSERVANCE or \ref AWARENESS the \ref fluent_formula condition for the observability is stored here (not grounded).*/
    belief_formula m_observability_conditions;
    /** \brief If *this* is \ref ONTIC, \ref SENSING or \ref ANNOUNCEMENT the \ref belief_formula condition for the act is stored here.*/
    belief_formula m_executability_conditions;



public:

    /** Getter of the field \ref m_type.
     * 
     * @return the value of \ref m_type.*/
    proposition_type get_type() const;
    /** Getter of the field \ref m_action_name.
     * 
     * @return the value of \ref m_action_name.*/
    const std::string & get_action_name() const;
    /** Getter of the field \ref m_action_effect grounded.
     * 
     * @return the grounded value of \ref m_action_effect.*/
    fluent_formula get_action_effect() const;
    /** Getter of the field \ref m_agent grounded.
     * 
     * @return the grounded value of \ref m_agent.*/
    agent get_agent() const;
    /** Getter of the field \ref m_observability_conditions.
     * 
     * @return the grounded value of \ref m_observability_conditions.*/
    const belief_formula & get_observability_conditions() const;
    /** Getter of the field \ref m_executability_conditions.
     * 
     * @return the grounded value of \ref m_executability_conditions.*/
    const belief_formula & get_executability_conditions() const;

    /** Getter of the field \ref m_observability_conditions grounded.
     * 
     * @return the grounded value of \ref m_observability_conditions.*/
    const belief_formula & get_grounded_observability_conditions();

    /** Getter of the field \ref m_executability_conditions grounded.
     * 
     * @return the grounded value of \ref m_executability_conditions.*/
    const belief_formula & get_grounded_executability_conditions();


    /** Setter for the field \ref m_type.
     * 
     * @param[in] to_set: the value to assign to \ref m_type.*/
    void set_type(proposition_type to_set);
    /** Setter for the field \ref m_action_name.
     * 
     * @param[in] to_set: the value to assign to \ref m_action_name.*/
    void set_action_name(const std::string & to_set);


    /** Setter for the field \ref m_action_effect.
     * 
     * @param[in] to_set: the value to assign to \ref m_action_effect.*/
    void set_action_effect(const string_set_set & to_set);
    /** Function that adds an action effect to \ref m_action_effect.
     * 
     * @param[in] to_add: the value to add to \ref m_action_effect.*/
    void add_action_effect(const string_set & to_add);

    /** Setter for the field \ref m_agent.
     * 
     * @param[in] to_set: the value to assign to \ref m_agent.*/
    void set_agent(const std::string & to_set);
    /** Setter for the field \ref m_observability_conditions.
     * 
     * @param[in] to_set: the value to assign to \ref m_observability_conditions.*/
    void set_observability_conditions(const belief_formula & to_set);
    /** Setter for the field \ref m_executability_conditions.
     * 
     * @param[in] to_set: the value to assign to \ref m_executability_conditions.*/
    void set_executability_conditions(const belief_formula & to_set);

    /** \brief Function that print *this*.*/
    void print() const;

    /** \brief Function that grounds *this*.*/
    void ground();
};

typedef std::list<proposition> proposition_list; /**< \brief A list of proposition to simplify the usage.*/