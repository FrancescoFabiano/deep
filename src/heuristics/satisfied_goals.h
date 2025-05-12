/**
 * \class satisfied_goals
 * \brief Class that implements the heuristic "number of satisfied sub_goals".
 *
 * \details In this class we simply check how many sub_goals 
 * (elements of domain::get_instance.get_goal_description()) each state satisfies
 * and we pick the states which satisfies the highest number of sub goals as the next to
 * expand.
 * As extension we should break each complex belief formula in more than one so we
 * have more sub goals and more accuracy. Especially we should break C in multiple components.
 * 
 *
 * \copyright GNU Public License.
 * 
 * \author Francesco Fabiano.
 * \date November 28, 2019
 */

#pragma once
#include "../utilities/define.h"
#include "../formulae/belief_formula.h"

class satisfied_goals
{
private:
    /** \brief The number of subgoals.*/
    unsigned short m_goals_number;

    /** \brief A local copy of the goal that is possibly modified (Added more subgoals).*/
    formula_list m_goals;

    /**Private constructor because is a Singleton class*/
    satisfied_goals();

    /**Setter of the field \ref m_goals
     *
     * @param[in] to_set: the cnf of expanded sub goals to set as \ref m_goals.
     */
    void set_goals(const formula_list & to_set);
    /**Setter of the field \ref m_goals_number
     *
     * @return: the number of expanded sub goals to set.
     */
    void set_goals_number(unsigned short to_set);
public:

    /** \brief To get always (the same instance of) *this* and the same instantiated fields.
     * \warning the \ref set has to called in the heuristics_manager file only.*/
    static satisfied_goals& get_instance();

    /** \brief Class setter that assign the goals and the size to *this*.
     *
     * @param[in] goals: The goal description, expanded by \ref heuristics_manager.
     */
    void set(const formula_list & goals);

    /** \brief Function that return the number of unsatisifed goals of \ref eState
     * 
     * @param[in] eState: The (templatic) state where to check for unsatisfied goals
     * @return: The number of unsatisfied goals in \ref eState.*/
    template< class T>
    unsigned short get_unsatisfied_goals(const T & eState) const;


    /** \brief Function that return the number of satisifed goals of \ref eState
     * 
     * @param[in] eState: The (templatic) state where to check for satisfied goals
     * @return: The number of satisfied goals in \ref eState.*/
    template< class T>
    unsigned short get_satisfied_goals(const T & eState) const;

    /**Getter of the field \ref m_goals
     *
     * @return: the cnf of expanded sub goals.
     */
    const formula_list & get_goals() const;

    /**Getter of the field \ref m_goals_number
     *
     * @return: the number of expanded sub goals.
     */
    unsigned short get_goals_number() const;


    /* \brief The = operator.
     *   
     * @param [in] to_copy: the \ref s to assign to *this*.
     * @return true: if \p the satisfied_goals went ok.
     * @return false: otherwise.
    bool operator=(const satisfied_goals<T>& to_copy);*/


    /** \brief Copy constructor removed since is Singleton class. */
    satisfied_goals(satisfied_goals const&) = delete;
    /** \brief Copy operator removed since Singleton class. */
    void operator=(satisfied_goals const&) = delete;
};
