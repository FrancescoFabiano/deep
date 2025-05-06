/**
 * \brief Class that describes an edge of a Kripke structure (\ref kstate).
 *
 * \details   An edge in a \ref kstate labeled with the \ref agent *ag* represent
 *            the information that the \ref agent *ag* cannot distinguish between the two worlds
 *            that the edge connects.
 * 
 *            Each label is provided with an id to save and retrieve them from \ref kstore.
 * 
 * @see kstate, kworld, and kstore.
 *
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date March 31, 2019
 */
#pragma once

#include "kworld.h"

#include "../../utilities/define.h"

class kedge
{
    friend class kedge_ptr;
private:
    /**
     * \brief The \ref kworld_ptr to the world where *this* is from.
     */
    kworld_ptr m_from;
    /**
     * \brief The \ref kworld_ptr to the world where *this* is directed.
     */
    kworld_ptr m_to;
    /**
     * \brief The label of *this*, that is an \ref agent.
     */
    agent m_label;
    /**
     * \brief The unique id of *this* computed with \ref hash_info_into_id().
     */
    kedge_id m_edge_id;

    /**
     * \brief Function used to hash the the info of an edge in a unique id.
     *
     *
     *
     * @param[in] from: the world where the edge is coming from.
     * @param[in] to: the world where the edge is going to.
     * @param[in] label: the \ref agent that is the label of the edge.
     * @return the unique id of the edge.
     * 
     * @warning Useless if not moved to \ref kstore.
     */
    kedge_id hash_info_into_id(const kworld_ptr & from, const kworld_ptr & to, agent label);

    /**
     * \brief Function used to hash the the info of *this* in a unique id.
     *
     * @return the unique id of *this*.
     */
    kedge_id hash_info_into_id();

    /** \brief Setter for the field \ref m_from.
     * 
     * It takes a world and set its \ref kworld_ptr, it is private
     * because we want to do it only in the constructor.
     *
     * @param[in] from: the \ref kworld to set as \ref m_from.
     * 
     * \todo is the parameter passing the best one?*/
    void set_from(const kworld & from);
    /** \brief Setter for the field \ref m_to.
     * 
     * It takes a world and set its \ref kworld_ptr, it is private
     * because we want to do it only in the constructor.
     *
     * @param[in] to: the \ref kworld to set as \ref m_to.
     * 
     * \todo is the parameter passing the best one?*/
    void set_to(const kworld & to);


    /** \brief Setter for the field \ref m_from.
     * 
     * It is private because we want to do it only in the constructor.
     *
     * @param[in] from: the \ref kworld_ptr to set as \ref m_from.*/
    void set_from(const kworld_ptr & from);
    /** \brief Setter for the field \ref m_to.
     * 
     * It is private because we want to do it only in the constructor.
     *
     * @param[in] to: the \ref kworld_ptr to set as \ref m_to.*/
    void set_to(const kworld_ptr & to);

    /**
     * \brief Setter for the field \ref m_label.
     * 
     * It is private because we want to do it only in the constructor,
     *
     * @param[in] ag: the \ref agent to set as \ref m_label.
     * 
     */
    void set_label(agent ag);

    /**
     * \brief Function that uses the info of  *this* to set its  \ref m_edge_id.
     */
    void set_id();

    /** \brief Getter of \ref m_from.
     * 
     * Only accessible by the \ref kedge_ptr.
     *  
     * @return the \ref kworld_ptr to the world where *this* is from.*/
    const kworld_ptr & get_from() const;
    /** \brief Getter of \ref m_to.
     * 
     * Only accessible by the \ref kedge_ptr.
     *     
     * @return the \ref kworld_ptr to the world where *this* is directed.*/
    const kworld_ptr & get_to() const;
    /** \brief Getter of \ref m_label.
     * 
     * Only accessible by the \ref kedge_ptr.
     *  
     * @return the \ref agent that is the label of *this*.*/
    agent get_label() const;
    /**\brief Getter of \ref m_edge_id.
     * 
     * Only accessible by the \ref kedge_ptr.
     *      
     * @return the int that is the unique id of *this*.*/
    kedge_id get_id() const;

public:
    /**
     * \brief Empty constructor, call the default constructor of all the fields.
     */
    kedge();
    /** \brief Constructor with parameters.
     * 
     * Construct an object with the given info and then set the unique id.
     * @param[in] from: the world where *this* is coming from.
     * @param[in] to: the world where *this* is going to.
     * @param[in] label: the \ref agent that is the label of *this*.*/
    kedge(const kworld & from, const kworld & to, agent label);


    /** \brief Constructor with parameters.
     * 
     * Construct an object with the given info and then set the unique id.
     * @param[in] from: the pointer to the world where *this* is coming from.
     * @param[in] to: the pointer to the world where *this* is going to.
     * @param[in] label: the \ref agent that is the label of *this*.*/
    kedge(const kworld_ptr & from, const kworld_ptr & to, agent label);


    /** \brief The < operator based on the field \ref m_edge_id.
     * Implemented to allow the ordering on set of \ref kedge (used in \ref kstore).
     * 
     * @see kstore.
     *     
     * @param [in] to_compare: the \ref kedge to compare with *this*.
     * @return true: if \p to_compare is smaller than *this*
     * @return false: otherwise.*/
    bool operator<(const kedge& to_compare) const;

    /** \brief The > operator based on the field \ref m_edge_id. 
     * @see kstore.
     *     
     * @param [in] to_compare: the \ref kedge to compare with *this*.
     * @return true: if \p to_compare is bigger than *this*
     * @return false: otherwise.*/
    bool operator>(const kedge& to_compare) const;

    /** \brief The == operator based on the field \ref m_edge_id.
     *     
     * @param [in] to_compare: the \ref kedge to compare with *this*.
     * @return true: if \p to_compare is equal to *this*
     * @return false: otherwise.*/
    bool operator==(const kedge& to_compare) const;


    /** \brief The = operator.
     *   
     * @param [in] to_assign: the \ref kedge to assign to *this*.
     * @return true: if \p the assignment went ok.
     * @return false: otherwise.*/
    bool operator=(const kedge& to_assign);

    /** \brief Function used to print all the information of *this*.*/
    void print() const;
};

/**
 * \class kedge_ptr
 * 
 * \brief A wrapper class for a std::shared_pointer to a \ref kedge usually stored in \ref kstore.
 * 
 * This class is necessary so we can use the < operator in the set of \ref kedge_ptr.
 * If we dom't use the wrapper pointers to the same elements are different.
 *
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 29, 2019
 */

class kedge_ptr
{
private:
    /**\brief the pointer that is wrapped by *this*.*/
    std::shared_ptr<const kedge> m_ptr;

public:
    /**\brief Constructor without parameters.*/
    kedge_ptr();
    /**\brief Constructor with parameters.
     *
     * This constructor uses const parameter, this means that the pointer is copied
     * and the counter of the shared pointer is increased (std implementation).
     * 
     * @param[in] ptr: the pointer to assign to \ref m_ptr.*/
    kedge_ptr(const std::shared_ptr<const kedge> & ptr);
    /**\brief Constructor with parameters.
     *
     * This constructor uses non-const parameter, this means that the pointer is copied
     * in \ref m_ptr and \p ptr becomes empty (std implementation).
     *  
     * @param[in] ptr: the pointer to assign to \ref m_ptr.*/
    kedge_ptr(std::shared_ptr<const kedge>&& ptr);

    /**\brief Setter for the field \ref m_ptr.
     *
     * This setter uses const pointer, this means that the pointer is copied
     * and the counter of the shared pointer is increased (std implementation).
     * 
     * @param[in] ptr: the pointer to assign to \ref m_ptr.*/
    void set_ptr(const std::shared_ptr<const kedge> & ptr);
    /**\brief Setter for the field \ref m_ptr (move constructor).
     * 
     * This setter uses non-const pointer, this means that the pointer is copied
     * in \ref m_ptr and \p ptr becomes empty (std implementation).
     *  
     * @param[in] ptr: the pointer to assign to \ref m_ptr.*/
    void set_ptr(std::shared_ptr<const kedge>&& ptr);

    /**\brief Getter for the field \ref m_ptr.
     *  
     * @return a copy of the pointer \ref m_ptr.*/
    std::shared_ptr<const kedge> get_ptr() const;

    /** \brief Function that return the field m_from of the pointed \ref kworld. 
     *     
     * @return the \ref kworld_ptr to the world where the \ref kworld pointed by \ref m_ptr is from.*/
    const kworld_ptr & get_from() const;
    /** \brief Function that return the field m_to of the pointed \ref kworld. 
     *      
     * @return the \ref kworld_ptr to the world where the \ref kworld pointed by \ref m_ptr is directed.*/
    const kworld_ptr & get_to() const;
    /** \brief Function that return the field m_label of the pointed \ref kworld.
     *     
     * @return the \ref agent that is the label of the \ref kworld pointed by \ref m_ptr.*/
    agent get_label() const;

    /** \brief Function that return the field m_id of the pointed \ref kedge.
     *     
     * @return the \ref kedge_id that is the id of the \ref kedge pointed by \ref m_ptr.*/
    kedge_id get_id() const;

    /**\brief The operator =.
     *
     * This operator assign the parameter without destroying \p to_copy.
     * 
     * @param[in] to_copy: the \ref kedge_ptr to assign to *this*.
     * @return true: if the assignment went through.
     * @return false: otherwise.*/
    bool operator=(const kedge_ptr & to_copy);
    /**\brief The operator < for set operation.
     *
     * The ordering is based on the pointed object and not on the pointer itself so we have one
     * copy of each pointed object.
     * 
     * @param[in] to_compare: the \ref kedge_ptr to check for odering.
     * 
     * @return true: if *this* is smaller than \p to_compare.
     * @return false: otherwise.*/
    bool operator<(const kedge_ptr & to_compare) const;

    /**\brief The operator > \ref kstate operator <.
     * 
     * @param[in] to_compare: the \ref kedge_ptr to check for odering.
     * 
     * @return true: if *this* is bigger than \p to_compare.
     * @return false: otherwise.*/
    bool operator>(const kedge_ptr & to_compare) const;
    /**\brief The operator ==.
     * 
     * @param[in] to_compare: the \ref kedge_ptr to confront with *this*.
     * @return true: if *this* is equal to \p to_compare.
     * @return false: otherwise.*/
    bool operator==(const kedge_ptr & to_compare) const;

    /** \brief Function used to print all the information of \ref m_ptr.*/
    void print() const;
};
