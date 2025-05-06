/**
 * \class pstore_opt
 * \brief Singleton class that stores the unique copy of each \ref pstate_opt created.
 * 
 * \details All the other class only store pointers of \ref pstate_opt and this class takes care of creating new
 * one in case the requested one doesn't exist.
 * 
 * @see pstate_opt.
 * 
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date September 14, 2019
 */

#pragma once

#include <set>
#include "pstate_opt.h"

class pstore_opt
{
private:
    /** \brief All the created \ref pstate_opt, all the other class only have pointers to elements of this set.
     * 
     * \todo should it be static?*/
    pstate_opt_set m_created_states;

    /** \brief Private constructor since it is a Singleton class.*/
    pstore_opt();
public:

    /** \brief To get always (the same instance of) *this* and the same instantiated fields.*/
    static pstore_opt& get_instance();

    /** \brief Function that return the pointer to the given \ref pstate_opt.
     * 
     * If the \ref pstate_opt didn't exist then it is inserted to \ref m_created_states and then the pointer is returned;
     * if it existed a pointer to the already existing version is returned.
     * 
     * @param[in] to_add: the \ref pstate_opt to add to the collection of created states.
     * 
     * @return the \ref pstate_opt_ptr to \p to_add.
     * 
     * \todo Param ok because set makes copy?*/
    const pstate_opt_ptr add_state(const pstate_opt & to_add);

    /*\brief Function that add a created \ref pstate_opt without returning its pointer.
     * 
     * @param[in] to_add: the \ref pstate_opt to add to the collection of created states. */
    /*void add_state_no_ret(const pstate_opt & to_add);*/

    /** \brief Copy constructor removed since is Singleton class. */
    pstore_opt(pstore_opt const&) = delete;
    /** \brief Copy operator removed since Singleton class. */
    void operator=(pstore_opt const&) = delete;

};

