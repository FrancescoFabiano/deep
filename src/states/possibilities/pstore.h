/**
 * \class pstore
 * \brief Singleton class that stores the unique copy of each \ref pworld created.
 * 
 * \details All the other class only store pointers of \ref pworld and this class takes care of creating new
 * one in case the requested one doesn't exist.
 * 
 * @see pworld.
 * 
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date September 14, 2019
 */

#pragma once

#include <set>
#include "pworld.h"

typedef std::set<pworld> pworld_set; /**< \brief A set of \ref pworld, used to store all the created ones.*/

class pstore
{
private:
    /** \brief All the created \ref pworld, all the other class only have pointers to elements of this set.
     * 
     * \todo should it be static?*/
    pworld_set m_created_worlds;

    /** \brief Private constructor since it is a Singleton class.*/
    pstore();
public:

    /** \brief To get always (the same instance of) *this* and the same instantiated fields.*/
    static pstore& get_instance();
    
    /** \brief Function that return the pointer to the given \ref pworld.
     * 
     * If the \ref pworld didn't exist then it is inserted to \ref m_created_worlds and then the pointer is returned;
     * if it existed a pointer to the already existing version is returned.
     * 
     * @param[in] to_add: the \ref pworld to add to the collection of created worlds.
     * 
     * @return the \ref pworld_ptr to \p to_add.
     * 
     * \todo Param ok because set makes copy?*/
    const pworld_ptr add_world(const pworld & to_add);

    /*\brief Function that add a created \ref pworld without returning its pointer.
     * 
     * @param[in] to_add: the \ref pworld to add to the collection of created worlds. */
    /*void add_world_no_ret(const pworld & to_add);*/

    /** \brief Copy constructor removed since is Singleton class. */
    pstore(pstore const&) = delete;
    /** \brief Copy operator removed since Singleton class. */
    void operator=(pstore const&) = delete;

};

