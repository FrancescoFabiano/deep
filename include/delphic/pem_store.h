/**
 * \class pem_store
 * \brief Singleton class that stores the unique copy of each \ref event created.
 * 
 * \details All the other class only store pointers of \ref event and this class takes care of creating new
 * one in case the requested one doesn't exist.
 * 
 * @see event.
 * 
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 28, 2021
 */

#pragma once

#include <set>
#include "event.h"
#include "pem.h"

class pem_store
{
private:
    /** \brief The event representing the perspective of oblivious agents.*/
    event_ptr m_epsilon;
    /** \brief A set that contains all the generated pems (even the pems that are "internal")*/
    event_set m_created_events;

    /** \brief All the event that represent an action (given as input).
     *
     * These pems are pointer to the ones in \ref m_created_events.*/
    pem_set m_created_pems;

    std::map<std::string, agent_group> m_agent_group_ids;

    /** \brief Private constructor since it is a Singleton class.*/
    pem_store();
public:
    /**Function that populates the list m_created_pems with the event model given as input in \ref file*/
    void generate(const std::string & file);

    /** \brief To get always (the same instance of) *this* and the same instantiated fields.*/
    static pem_store& get_instance();
    //    /** \brief Function that inserts a \ref event given its id to \ref m_created_pems.
    //     * @param[in] to_add: The event model that represent the graph of an action.
    //     * @parma[in] id: The id associated to \ref to_add.
    //     */
    //    void add_action_pem(const event & to_add, event_id id);
    //
    //    /** \brief Function that inserts a \ref event given its id to \ref m_created_pems.
    //     * @param[in] to_add: The event model (pointer) that represent the graph of an action.
    //     * @parma[in] id: The id associated to \ref to_add.
    //     */
    //    void add_action_pem(const event_ptr & to_add, event_id id);
    //
    //    /** \brief Function that retrieves a \ref event from its \ref event_id.
    //     * @parma[in] id: The id of the \ref event that needs to be returned.
    //     * @return: The event model that represent the graph of the action associated to \ref id.
    //     */
    //    const event_ptr & get_action_pem(event_id id);


    /** \brief Function that return the pointer to the given \ref event.
     * 
     * If the \ref event didn't exist then it is inserted to \ref m_created_events and then the pointer is returned;
     * if it existed a pointer to the already existing version is returned.
     * 
     * @param[in] to_add: the \ref event to add to the collection of created events.
     * 
     * @return the \ref event_ptr to \p to_add.*/
    const event_ptr add_event(const event & to_add);
    /** \brief Function that return the pointer to the given \ref pem.
     *
     * If the \ref pem didn't exist then it is inserted to \ref m_created_pems and then the pointer is returned;
     * if it existed a pointer to the already existing version is returned.
     *
     * @param[in] to_add: the \ref pem to add to the collection of created pems.
     *
     * @return the \ref pem_ptr to \p to_add.*/
    const pem_ptr add_pem(const pem & to_add);


    /** \brief Function that add a new agent group name to the map name->id.
     *
     * @return the \ref string to \p to_add in m_agent_group_ids.*/
    void add_agent_group(const std::string & to_add);


    /** \brief Function that return the pointer to the given \ref event from its id.
     * 
     * @return the \ref event_ptr to that has ref \id id.*/
    const event_ptr get_event(event_id id) const;

    /** \brief Function that return the pointer to the given \ref model from its id.
     * 
     * @return the \ref pem_ptr to that has ref \id id.*/
    const pem_ptr get_pem(pem_id id) const;



    /** \brief Function that returns the id given an agent group name.
     *
     * @return the \ref id of \p to_get.*/
    agent_group get_agent_group(const std::string & to_get) const;

    std::string get_agent_group_name(agent_group id) const;


    /** \brief Copy constructor removed since is Singleton class. */
    pem_store(pem_store const&) = delete;
    /** \brief Copy operator removed since Singleton class. */
    void operator=(pem_store const&) = delete;

};

