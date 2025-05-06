/**
 * \class attitude
 * \brief Class used a support to elaborate the attitudes of the \ref agent(s) in the domain.
 *
 * An \ref attitude identifies and specifies an \ref agents attitude w.r.t. an executor and some conditions.
 * 
 * The content of an \ref attitude is not grounded as will be grounded once is inserted in \ref attitudes_table.
 *
 * \see attitudes_table.
 * 
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 1, 2019
 */

#pragma once

#include "../formulae/belief_formula.h"
#include "../utilities/define.h"

class attitude
{
private:
    /** The \ref agents_attitudes of this.*/
    agents_attitudes m_type;
    /** The agent for which is specified the attitude.*/
    std::string m_agent;
    /** The agent w.r.t. \ref m_agents has the \ref m_type attitude if the conditions are met.*/
    std::string m_executor;
    /** The condition for the attitude to be true.*/
    belief_formula m_attitude_conditions;

public:

    /** Getter of the field modified \ref m_type.
     * 
     * @return the value of \ref m_type.*/
    agents_attitudes get_type() const;

    /** Getter of the field \ref m_type.
     * 
     * @return the value of \ref m_type.*/
    std::string get_string_type() const;
    /** Getter of the field \ref m_agent grounded.
     * 
     * @return the grounded value of \ref m_agent.*/
    agent get_agent() const;

    /** Getter of the field \ref m_executor grounded.
     * 
     * @return the grounded value of \ref m_executor.*/
    agent get_executor() const;

    /** Getter of the field \ref m_attitude_conditions grounded.
     * 
     * @return the grounded value of \ref m_attitude_conditions.*/
    const belief_formula & get_attitude_conditions();


    /** Getter of the field \ref m_attitude_conditions.
     * 
     * @return the value of \ref m_attitude_conditions.*/
    const belief_formula & get_original_attitude_conditions() const;

    /** Setter for the field \ref m_type.
     * 
     * @param[in] to_set: the value to assign to \ref m_type.*/
    void set_type(agents_attitudes to_set);

    /** Setter for the field \ref m_agent.
     * 
     * @param[in] to_set: the value to assign to \ref m_agent.*/
    void set_agent(const std::string & to_set);

    /** Setter for the field \ref m_executor.
     * 
     * @param[in] to_set: the value to assign to \ref m_executor.*/
    void set_executor(const std::string & to_set);
    /** Setter for the field \ref m_attitude_conditions.
     * 
     * @param[in] to_set: the value to assign to \ref m_attitude_conditions.*/
    void set_attitude_conditions(const belief_formula & to_set);


    /** \brief Function that print *this*.*/
    void print(const belief_formula & cond) const;

    /** \brief The = operator.
     *   
     * @param [in] to_copy: the \ref belief_formula to assign to *this*.
     * @return true: if \p the assignment went ok.
     * @return false: otherwise.*/
    bool operator=(const attitude& to_copy);

};

typedef std::list<attitude> attitudes_list; /**< \brief A list of attitude to simplify the usage.*/