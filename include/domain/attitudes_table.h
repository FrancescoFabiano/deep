/**
 * \brief Class that store the info related to a the agents attitude.
 * 
 * This class stores the info of all the agents' attitude.
 * This attitude depends on both the action executors and on the state of the world.
 *  
 * An instance of this class contains the info for each agent.
 * Moreover, given a state and an executor, this class will return all the agents attitudes.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date December 9, 2020
 */

#pragma once

#include "attitude.h"
#include "../utilities/define.h"

typedef std::map<agent, agents_attitudes> single_attitudes_map;
typedef std::map<agent, std::map<agent, std::map<agents_attitudes, belief_formula>>> complete_attitudes_map;

class attitudes_table
{
private:

    /**A map that contains the fully_observants attitudes of m_agent when the exe_agent is the executor.
     * 
     * each attitude is associated to a condition. Only one attitude can be true at the time;
     * we have some order in the attitudes and this make us select the first one that is activated.
     * 
     * The order is the following: F_TRUSTY, F_MISTRUSTY, F_UNTRUSTY, F_STUBBORN
     */
    complete_attitudes_map m_F_attitude_wrt_exec;

    /**A map that contains the partially_observants attitudes of m_agent when the exe_agent is the executor.
     * 
     * each attitude is associated to a condition. Only one attitude can be true at the time;
     * we have some order in the attitudes and this make us select the first one that is activated
     * 
     * The order is the following: P_KEEPER, P_INSECURE
     */
    complete_attitudes_map m_P_attitude_wrt_exec;

    /**Function that add the attitude for \ref m_agent w.r.t, \ref executor and an \ref attitude_condition to meet
     * @param[in] m_agent: the agent of whom we want to analyze the attitude.
     * @param[in] executor: the agent with respects with \ref m_agent has \ref attitude when \ref attitude_condition is met.
     * @param[in] attitude: the \ref attitude that the agent will have w.r.t. \ref executor when \ref attitude_condition is met.
     * @param[in] attitude_condition: the condition that the state must require for the agent to have the attitude.
     * @param[in] table: the map where to add the attitudes (fully or partially).
     */
    void add_attitudes(agent m_agent, agent executor, agents_attitudes attitude, const belief_formula & attitude_condition, complete_attitudes_map & table);

    /**Function that add the fully_obs attitude for \ref m_agent w.r.t, \ref executor when \ref attitude_condition is met
     * @param[in] m_agent: the agent of whom we want to analyze the attitude.
     * @param[in] executor: the agent with respects with \ref m_agent has \ref attitude when \ref attitude_condition is met.
     * @param[in] attitude: the \ref attitude that the agent will have w.r.t. \ref executor when \ref attitude_condition is met.
     * @param[in] attitude_condition: the condition that the state must require for the agent to have the attitude.
     */
    void add_F_attitudes(agent m_agent, agent executor, agents_attitudes attitude, const belief_formula & attitude_condition);

    /**Function that add the partially_obs attitude for \ref m_agent w.r.t, \ref executor when \ref attitude_condition is met
     *
     * @param[in] m_agent: the agent of whom we want to analyze the attitude.
     * @param[in] executor: the agent with respects with \ref m_agent has \ref attitude when \ref attitude_condition is met.
     * @param[in] attitude: the \ref attitude that the agent will have w.r.t. \ref executor when \ref attitude_condition is met.
     * @param[in] attitude_condition: the condition that the state must require for the agent to have the attitude.
     */
    void add_P_attitudes(agent m_agent, agent executor, agents_attitudes attitude, const belief_formula & attitude_condition);


    void print_table(const complete_attitudes_map & table) const;
public:
    /**Class that initialize the internal maps with default values
     * 
     * @param[in] tot_ags: the set of all the agents from the domain.
     * @param[in] f1: the first fluent of the domain.
     */
    void set_attitudes_table(const agent_set & tot_ags, fluent f1);

    /**Function that add the an attitude
     * @param[in] att: the attitude to add to *this*.
     */
    void add_attitude(attitude & att);

    /*Getter of the field \ref m_F_attitude_wrt_exec
     * 
     * @return: the field \ref m_F_attitude_wrt_exec
     */
    const complete_attitudes_map & get_F_attitudes() const;

    /*Getter of the field \ref m_P_attitude_wrt_exec
     * 
     * @return: the field \ref m_P_attitude_wrt_exec
     */
    const complete_attitudes_map & get_P_attitudes() const;

    void print() const;
};