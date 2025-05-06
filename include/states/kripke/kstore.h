/**
 * \class kstore
 * \brief Singleton class that stores the unique copy of each \ref kedge and \ref kworld created.
 * 
 * \details All the other class only store pointers of \ref kedge or \ref kworld and this class takes care of creating new
 * one in case the requested one doesn't  exist.
 * 
 * @see kedge and kworld.
 *
 *
 * \todo It is better to create a new world each time or to search for the key and in case
 * they don't exist create the world and add the world (double insert basically) or the function
 * const kworld& add_world(const fluent_formula &)?
 * Use function in case building world is way more expensive than searching the set.
 * 
 * \todo Implement the hash here statically.
 * 
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 9, 2019
 */

#pragma once

#include <set>
#include "kworld.h"
#include "kedge.h"


typedef std::set<kedge> kedge_set; /**< \brief A set of \ref kedge, used to store all the created ones.*/
typedef std::set<kworld> kworld_set; /**< \brief A set of \ref kworld, used to store all the created ones.*/

class kstore
{
private:
    /** \brief All the created \ref kedge, all the other class only have pointers to elements of this set.
     * 
     * \todo should it be static?*/
    kedge_set m_created_edges;
    /** \brief All the created \ref kworld, all the other class only have pointers to elements of this set.
     * 
     * \todo should it be static?*/
    kworld_set m_created_worlds;

    /** \brief Private constructor since it is a Singleton class.*/
    kstore();
public:

    /** \brief To get always (the same instance of) *this* and the same instantiated fields.*/
    static kstore& get_instance();
    
    /** \brief Function that return the pointer to the given \ref kedge.
     * 
     * If the \ref kedge didn't exist then it is inserted to \ref m_created_edges and then the pointer is returned;
     * if it existed a pointer to the already existing version is returned.
     * 
     * @param[in] to_add: the \ref kedge to add to the collection of created edges.
     * 
     * @return the \ref kedge_ptr to \p to_add.
     * 
     * \todo Param ok because set makes copy?*/
    const kedge_ptr add_edge(const kedge & to_add);
    /** \brief Function that return the pointer to the given \ref kworld.
     * 
     * If the \ref kworld didn't exist then it is inserted to \ref m_created_worlds and then the pointer is returned;
     * if it existed a pointer to the already existing version is returned.
     * 
     * @param[in] to_add: the \ref kworld to add to the collection of created worlds.
     * 
     * @return the \ref kworld_ptr to \p to_add.
     * 
     * \todo Param ok because set makes copy?*/
    const kworld_ptr add_world(const kworld & to_add);

    /*\brief Function that add a created \ref kworld without returning its pointer.
     * 
     * @param[in] to_add: the \ref kworld to add to the collection of created worlds. */
    /*void add_world_no_ret(const kworld & to_add);*/

    /** \brief Copy constructor removed since is Singleton class. */
    kstore(kstore const&) = delete;
    /** \brief Copy operator removed since Singleton class. */
    void operator=(kstore const&) = delete;

};

