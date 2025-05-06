/**
 * \class belief_formula
 * \brief Class that implements a Belief Formula.
 *
 * \details A \ref belief_formula can have several form:
 *    - \ref FLUENT_FORMULA -- \ref fluent_formula;
 *    - \ref BELIEF_FORMULA -- B(\ref agent, *phi*);
 *    - \ref PROPOSITIONAL_FORMULA -- \ref BF_NOT(*phi*) or (*phi_1* \ref BF_AND *phi_2*) or (*phi_1* \ref BF_OR *phi_2*);
 *    - \ref E_FORMULA -- E([set of \ref agent], *phi*);
 *    - \ref C_FORMULA -- C([set of \ref agent], *phi*);
 *    - \ref D_FORMULA -- D([set of \ref agent], *phi*);
 * 
 * 
 * @see reader, domain
 * 
 * \todo Maybe implement the "move" so the reader actually move the object instead of copying them.
 *
 * \copyright GNU Public License.
 * 
 * \author Francesco Fabiano.
 * \date March 31, 2019
 */
#pragma once

#include <memory>

#include "../utilities/define.h"

#include "../domain/grounder.h"

/** \brief The possible types of \ref belief_formula.*/
enum bf_type
{
    FLUENT_FORMULA, /**< \brief A \ref belief_formula is also a \ref fluent_formula.
                     * 
                     *  This is the base case for the recursion.*/
    BELIEF_FORMULA, /**< \brief A \ref belief_formula of the form B(\ref agent, *phi*).
                     * 
                     *  This represent the belief of an \ref agent about a \ref belief_formula *phi*.*/
    PROPOSITIONAL_FORMULA, /**< \brief A \ref belief_formula composed with logical operators and \ref belief_formula(e).
                            * 
                            * This uses the operator: \ref BF_NOT, \ref BF_AND, \ref BF_OR
                            * 
                            * @see bf_operator.*/
    E_FORMULA, /**< \brief A \ref belief_formula of the form E([set of \ref agent], *phi*).
                     * 
                     *  This represent the belief of a set of \ref agent about a \ref belief_formula *phi*.*/
    C_FORMULA, /**< \brief A \ref belief_formula of the form C([set of \ref agent], *phi*).
                     * 
                     *  This represent the Common Knowledge of a set of \ref agent of a \ref belief_formula *phi*.*/
    D_FORMULA, /**< \brief A \ref belief_formula of the form D([set of \ref agent], *phi*).
                     * 
                     *  This represent the Distributed Knowledge of a set of \ref agent of a \ref belief_formula *phi*.*/
    BF_EMPTY, /**< \brief When the belief formula is empty.*/
    BF_TYPE_FAIL /**< \brief The failure case.*/
};

/** \brief The logical operator for \ref belief_formula(e).
 *
 * These are used in the case that the \ref bf_type of a \ref belief_formula is \ref PROPOSITIONAL_FORMULA.*/
enum bf_operator
{
    BF_AND, /**< \brief The AND between \ref belief_formula(e).*/
    BF_OR, /**< \brief The OR between \ref belief_formula(e).*/
    BF_NOT, /**< \brief The NOT of a \ref belief_formula.*/
    BF_INPAREN, /**< \brief When the \ref belief_formula is only surrounded by "()".*/
    BF_FAIL /**< \brief When the \ref belief_formula is not set PROPERLY (shouldn't be accessed if not \ref PROPOSITIONAL_FORMULA).*/
};

class belief_formula
{
private:

    /**A boolean value that tells if *this* has been grounded*/
    bool m_is_grounded = false;

    /*-******STRING FIELDs (private GETTER (for the grounding) and SETTER public for the reading)*************-*/
    /** \brief If *this* is a \ref FLUENT_FORMULA in this field is contained the 
     * string description of the actual \ref fluent_formula.*/
    string_set_set m_string_fluent_formula;

    /** \brief If *this* is a \ref BELIEF_FORMULA in this field is contained the
     *  string description of the \ref agent of the formula.*/
    std::string m_string_agent;

    /** \brief If this is a \ref E_FORMULA, \ref C_FORMULA or a \ref D_FORMULA here is contained the
     *  string description of the relative set of \ref agent.*/
    string_set m_string_group_agents;

    /** \brief Getter of the field \ref m_string_fluent_formula.
     * 
     * Only used inside *this* for the grounding process, so it's private.
     * 
     * @return the value of \ref m_string_fluent_formula.*/
    const string_set_set & get_string_fluent_formula() const;
    /** \brief Getter of the field \ref m_string_agent.
     * 
     * Only used inside *this* for the grounding process, so it's private.
     * 
     * @return the value of \ref m_string_agent.*/
    const std::string & get_string_agent() const;
    /** \brief Getter of the field \ref m_string_group_agents.
     * 
     * Only used inside *this* for the grounding process, so it's private.
     * 
     * @return the value of \ref m_string_group_agents.*/
    const string_set & get_string_group_agents() const;

    /*-******GROUNDED FIELDs (private SETTER (for the grounding/reading part) and GETTER public)*************-*/
    /**\brief The \ref bf_type of *this*.*/
    bf_type m_formula_type = BF_EMPTY; // = BF_TYPE_FAIL; //Which case of belief formula this is
    /**\brief If *this* is a \ref FLUENT_FORMULA in this field is contained the actual \ref fluent_formula.*/
    fluent_formula m_fluent_formula;
    /** \brief If *this* is a \ref BELIEF_FORMULA in this field is contained the \ref agent of the formula.
     *
     * Given the \ref belief_formula B(**ag**, *phi*) this field contains **ag**.*/
    agent m_agent;
    /** \brief In this field is contained the operator if *this* is \ref PROPOSITIONAL_FORMULA.
     *
     * - Given the \ref PROPOSITIONAL_FORMULA *phi_1* **op** *phi_2* \ref m_operator contains **op**.*/
    bf_operator m_operator;
    /** \brief If this is a \ref E_FORMULA, \ref C_FORMULA or a \ref D_FORMULA here is contained the relative set of \ref agent.
     *
     * - If *this* is \ref E_FORMULA, i.e., E(**ags**, *phi*), \ref m_group_agents contains **ags**;
     * - if *this* is \ref C_FORMULA, i.e., C(**ags**, *phi*), \ref m_group_agents contains **ags**;
     * - If *this* is \ref D_FORMULA, i.e., E(**ags**, *phi*), \ref m_group_agents contains **ags**.*/
    agent_set m_group_agents;


    /*-******RECOURSIVE FIELDs (private SETTER (for the reading part) and GETTER public)*************-*/
    /** \brief In this field is contained the first nested \ref belief_formula of *this*.
     *
     * - If *this* is \ref BELIEF_FORMULA, i.e., B(ag, *phi*), \ref m_bf1 contains *phi*;
     * - if *this* is \ref PROPOSITIONAL_FORMULA, i.e. *phi_1* **op** *phi_2*, \ref m_bf1 contains *phi_1*;
     * - if *this* is \ref E_FORMULA, i.e., E(ags, *phi*), \ref m_bf1 contains *phi*;
     * - if *this* is \ref C_FORMULA, i.e., C(ags, *phi*), \ref m_bf1 contains *phi*;
     * - if *this* is \ref D_FORMULA, i.e., D(ags, *phi*), \ref m_bf1 contains *phi*;
     * - null_ptr otherwise.*/
    std::shared_ptr<belief_formula> m_bf1;
    /** \brief In this field is contained the second nested \ref belief_formula of *this*.
     *
     * - If *this* is \ref PROPOSITIONAL_FORMULA, i.e. *phi_1* **op** *phi_2*, \ref m_bf2 contains *phi_2*;
     * - null_ptr otherwise.*/
    std::shared_ptr<belief_formula> m_bf2;



public:

    /**Empty Constructor*/
    belief_formula();

    /**Copy Constructor
     *
     * @param[in] to_copy: the \ref belief_formula to copy in *this*.*/
    belief_formula(const belief_formula & to_copy);

    /** \brief Getter for the field \ref m_is_grounded.
     *
     * @return true: if *this* has been already grounded
     * @return false: otherwise.*/
    bool get_is_grounded() const;

    /** \brief Setter for the field \ref m_is_grounded.
     *
     * Used when we create extra belief_formula directly grounded
     * @param[in] is_grounded: the value to assign to m_is_grounded.*/
    void set_is_grounded(bool is_grounded);

    /** \brief Setter for the field \ref m_fluent_formula.
     *
     * @param[in] to_set: the \ref fluent_formula object to copy in \ref m_fluent_formula.*/
    void set_fluent_formula(const fluent_formula& to_set);

    /** \brief Setter for the field \ref m_fluent_formula.
     *
     * @param[in] to_set: the \ref fluent object to insert in \ref m_fluent_formula.*/
    void set_fluent_formula_from_fluent(fluent to_set);

    /** \brief Setter of the field \ref m_string_fluent_formula.
     * 
     * public so the \ref reader can access and modify the string
     * fields of *this*. We don't have public getters because we want to able to retrieve only the grounded values.
     * 
     * @param[in] to_set: the \ref string_set_set to set as *this* \ref fluent_formula in string description.*/
    void set_string_fluent_formula(const string_set_set & to_set);


    /** \brief Setter for the field \ref m_agent.
     *
     * @param[in] to_set: the \ref agent object to copy in \ref m_agent.*/
    void set_agent(agent to_set);

    /** \brief Setter for the field \ref m_group_agents.
     *
     * @param[in] to_set: the \ref agent_set object to copy in \ref m_group_agents.*/
    void set_group_agents(const agent_set& to_set);

    /** \brief Setter of the field \ref m_string_agent.
     * 
     * public so the \ref reader can access and modify the string
     * fields of *this*. We don't have public getters because we want to able to retrieve only the grounded values.
     * 
     * @param[in] to_set: the string to set as *this* \ref agent in string description.*/
    void set_string_agent(std::string to_set);

    /** \brief Setter of the field \ref m_string_group_agents.
     * 
     * public so the \ref reader can access and modify the string
     * fields of *this*. We don't have public getters because we want to able to retrieve only the grounded values.
     * 
     * @param[in] to_set: the \ref string_set to set as *this* \ref agent_set in string description.*/
    void set_string_group_agents(const string_set & to_set);


    /** \brief Setter of the field \ref m_bf1.
     * 
     * This setter takes a \ref belief_formula and set \ref m_bf1 to be its pointer.
     * 
     * @param[in] to_set: the \ref belief_formula to be pointed by \ref m_bf1.*/
    void set_bf1(const belief_formula & to_set);

    /** \brief Setter of the field \ref m_bf2.
     * 
     * This setter takes a \ref belief_formula and set \ref m_bf2 to be its pointer.
     * 
     * @param[in] to_set: the \ref belief_formula to be pointed by \ref m_bf2.*/
    void set_bf2(const belief_formula & to_set);


    /** \brief Setter for the field \ref m_formula_type.
     *
     * @param[in] to_set: the \ref bf_type object to copy in \ref m_formula_type.*/
    void set_formula_type(bf_type to_set);
    /** \brief Setter for the field \ref m_operator.
     *
     * @param[in] to_set: the \ref bf_operator object to copy in \ref m_operator.*/
    void set_operator(bf_operator to_set);


    /** \brief Getter for the field \ref m_formula_type.
     *
     * @return the value of the field \ref m_formula_type.*/
    bf_type get_formula_type() const;
    /** \brief Getter for the field \ref m_fluent_formula.
     *
     * @return the value of the field \ref m_fluent_formula.*/
    const fluent_formula& get_fluent_formula() const;
    /** \brief Getter for the field \ref m_agent.
     *
     * @return the value of the field \ref m_agent.*/
    agent get_agent() const;
    /** \brief Getter of the \ref belief_formula pointed by \ref m_bf1.
     *
     * @return the \ref belief_formula pointed by \ref m_bf1.*/
    const belief_formula & get_bf1() const;
    /** \brief Getter of the \ref belief_formula pointed by \ref m_bf2.
     *
     * @return the \ref belief_formula pointed by \ref m_bf2.*/
    const belief_formula & get_bf2() const;
    /** \brief Getter for the field \ref m_operator.
     *
     * @return the value of the field \ref m_operator.*/
    bf_operator get_operator() const;
    /** \brief Getter for the field \ref m_group_agents.
     *
     * @return the value of the field \ref m_group_agents.*/
    const agent_set& get_group_agents() const;

    /**Setter from a fluent_formula*/
    void set_from_ff(const fluent_formula & to_build);


    /** \brief Function that print *this* (std::string parameters representation).
     *
     * \todo Since the string parameter should be removed print thanks to a \ref grounder.*/
    void print() const;

    /** \brief Function that print *this* (std::string parameters representation) degrounding the parameters.
     *
     * \todo Since the string parameter should be removed print thanks to a \ref grounder.*/
    void print_deground();

    /* \brief Function that print *this* where *this* is grounded.
     *
     * @param[in] gr: the \ref grounder obj used to ground the fields of *this*.
     * 
     * \todo Make sure it's grounded at construction and remove the parameter.
    void print_grounded(const grounder& gr) const;*/

    /** \brief Function that grounds the fields of *this*.
     * 
     * \todo Call on costruction from outside, maybe a static function in \ref grounder.*/
    void ground();

    /** \brief Function that degrounds the fields of *this*.*/
    void deground();


    /** \brief The equal operator for\ref belief_formula.
     * 
     * @param[in]  to_compare: the \ref belief_formula to compare with *this*.
     * 
     * @return true: if the given \ref belief_formula is syntacticly equal to *this*.
     * @return false: otherwise.*/
    bool operator==(const belief_formula& to_compare) const;

    /** \brief The = operator.
     *   
     * @param [in] to_copy: the \ref belief_formula to assign to *this*.
     * @return true: if \p the assignment went ok.
     * @return false: otherwise.*/
    bool operator=(const belief_formula& to_copy);

    /** \brief The < operator for set operations.
     *   
     * @param [in] to_compare: the \ref state to to_compare to *this*.
     * @return true: if *this* is smaller then to_compare.
     * @return false: otherwise.*/
    bool operator<(const belief_formula& to_compare) const;
};