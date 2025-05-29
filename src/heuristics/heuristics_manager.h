/**
 * \class heuristics_manager
 * \brief Class that set the score to each state by using the given heuristics.
 *
 * \copyright GNU Public License.
 * 
 * \author Francesco Fabiano.
 * \date November 28, 2019
 */
//#include "satisfied_goals.h"

#pragma once


#include "../utilities/Define.h"
#include "../domain/Domain.h"
#include "satisfied_goals.h"
#include "planning_graph.h"




class heuristics_manager
{
private:
    /**The type of heuristic used*/
    heuristics m_used_heur;
    /*The goal description, possibly expanded to exploit the heuristic even more*/
    FormulaeList m_goals;
    pg_f_map m_fluents_score;
    pg_bf_map m_bf_score;
    /*A map which contains the value of each "grounded" bf w.r.t. the classical planning BisGraph.*/
    // pg_bfs_score m_bf_score;


    /** \brief Function that expands the group formulae to have more sub_goals
     * 
     * For example: We have C([a,b], \phi) we then transform it in:
     *      - B(a, \phi);
     *      - B(b, \phi);
     *      - B(a, (B(b, \phi)));
     *      - B(b, (B(a, \phi)));
     *      - B(a, (B(b, B(a, \phi))));
     *      - etc. (Following \ref depth)
     *      - C([a,b], \phi).
     * 
     * @param[in] nesting: The max_depth of the generated subgoals. It is set to two for now so everybody know that everybody know.
     * 
     */
    void expand_goals(unsigned short nesting = 2);

    /**The recursive function to generate the various nested goals
     *
     * @param[in] nesting: The max_depth of the generated subgoals.
     * @param[in] depth: The depth of the currently generating subgoal.
     * @param[in] to_explore: The belief formula from which generate the next subgoal.
     * @param[in] agents: The agents for whom the new subgoals have to be generated.
     */
    void produce_subgoals(unsigned short nesting, unsigned short depth, const BeliefFormula & to_explore, const AgentsSet & agents);



public:
    /** \brief Class constructor that uses the chosen heuristic to perform the operation
     * 
     * @param[in] used_heur: Which heuristic has to be used.*/
    heuristics_manager(heuristics used_heur);

    template <class T>
    heuristics_manager(heuristics used_heur, const T & eState);

    /** \brief The function that compute heuristic value
     * 
     * @param[out] eState: The state to update with the calculated heuristic value.*/
    template <class T>
    void set_heuristic_value(T & eState);

    /**Setter for the field \ref m_used_heur
     *
     * @param[in] used_h: The heuristics to assign to \ref m_used_heur
     */
    void set_used_h(heuristics used_h);
    /**Getter for the field \ref m_used_heur
     *
     * @return: The heuristics assigned to \ref m_used_heur
     */
    heuristics get_used_h() const;


    /**Setter of the field \ref m_goals
     *
     * @param[in] to_set: the cnf of expanded sub goals to set as \ref m_goals.
     */
    void set_goals(const FormulaeList & to_set);

    /**Getter of the field \ref m_goals
     *
     * @return: the cnf of expanded sub goals.
     */
    const FormulaeList & get_goals() const;

    /** \brief The = operator.
     *   
     * @param [in] to_copy: the \ref heuristics_manager to assign to *this*.
     * @return true: if \p the assignment went ok.
     * @return false: otherwise.*/
    bool operator=(const heuristics_manager& to_copy);
};