/**
 * \brief Class that represents a Possibility-based Event Model (PEM).
 *
 * \details  A PEM is a representation for actions in epistemic planning
 *
 * @see action and event.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 05, 2021
 */
#pragma once

#include "../utilities/define.h"

class pem
{
    friend class pem_ptr;
private:
    /** \brief The id of *this*.*/
    pem_id m_id;
    /** \brief The map containing the edges of the action.
     *
     * Each edge consists of <\ref agent, \ref event_ptr_set> and link each \ref agent to a set of events.
     *
     * @see pem_store and agent.*/
    pem_edges m_edges;

    /** \brief The id of the pointed event of *this*.*/
    event_id m_pointed_id;



    /** \brief Getter of \ref m_id.
     *
     * Only accessible by the \ref pem_ptr.
     *
     * @return the \ref pem_id of *this*.*/
    pem_id get_id() const;


    /** \brief Getter of \ref m_pointed_id.
     *
     * Only accessible by the \ref pem_ptr.
     *
     * @return the \ref m_pointed_id of *this*.*/
    event_id get_pointed_id() const;

    /** \brief Getter of \ref m_edges.
     *
     * Only accessible by the \ref pem_ptr.
     *
     * @return the map of edges of *this*.*/
    const pem_edges & get_edges() const;

public:
    /** \brief Empty constructor, call the default constructor of all the fields.*/
    pem();
    /** \brief Constructor with parameters.
     * Construct an object with the given info.
     *
     * @param[in] id: the \ref id to set as \ref m_id.*/
    pem(pem_id id);
    /** \brief Copy constructor.
     *
     * @param[in] to_copy: the \ref pem to copy into *this*.*/
    pem(const pem & to_copy);

    /** \brief Setter for the field \ref m_id.
     *
     * Only accessible by the \ref pem_ptr.
     *
     * @param[in] to_set: the \ref pem_id to set as \ref m_id.*/
    void set_id(pem_id to_set);


    /** \brief Setter for the field \ref m_pointed_id.
     *
     * Only accessible by the \ref pem_ptr.
     *
     * @param[in] to_set: the \ref event_id to set as \ref m_pointed_id.*/
    void set_pointed_id(event_id to_set);


    /** \brief Setter for the field \ref m_edges.
     *
     * Only accessible by the \ref pem_ptr.
     *
     * @param[in] to_set: the map of \ref pem_edges to set as \ref m_edges.*/
    void set_edges(const pem_edges & to_set);

    /**
     *\brief The < operator based on the field \ref m_id.
     * Implemented to allow the ordering on set of \ref pem (used in \ref pem_store).
     *
     * @see pem_store.
     *
     * @param [in] to_compare: the \ref pem to compare with *this*.
     * @return true: if \p to_compare is smaller than *this*
     * @return false: otherwise.*/
    bool operator<(const pem & to_compare) const;
    /**
     *\brief The > operator based on the field \ref m_id.
     * Implemented to allow the ordering on set of \ref pem (used in \ref pem_store).
     *
     * @see pem_store.
     *
     * @param [in] to_compare: the \ref pem to compare with *this*.
     * @return true: if \p to_compare is bigger than *this*
     * @return false: otherwise.*/
    bool operator>(const pem & to_compare) const;
    /**
     *\brief The == operator based on the field \ref m_id.
     * Implemented to allow the ordering on set of \ref pem (used in \ref pem_store).
     *
     * @see pem_store.
     *
     * @param [in] to_compare: the \ref pem to compare with *this*.
     * @return true: if \p to_compare is equal to *this*
     * @return false: otherwise.*/
    bool operator==(const pem & to_compare) const;
    /** \brief The copy operator.
     *
     * @param [in] to_assign: the \ref pem to assign to *this*.
     * @return true: if \p the assignment went ok.
     * @return false: otherwise.*/
    bool operator=(const pem & to_copy);

    void print()const;
};

/**
 * \class pem_ptr
 * 
 * \brief A wrapper class for a std::shared_pointer to a \ref pem usually stored in \ref pem_store.
 * 
 * This class is necessary so we can use the < operator in the set of \ref pem_ptr.
 * If we don't use the wrapper pointers to the same elements are different.
 *
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 05, 2021
 */
class pem_ptr
{
private:
    /**\brief the pointer that is wrapped by *this*.*/
    std::shared_ptr<pem> m_ptr;

public:
    /**\brief Constructor without parameters.*/
    pem_ptr();

    /**\brief Constructor with parameters.
     *
     * This constructor uses const pointer, this means that the pointer is copied
     * and the counter of the shared pointer is increased (std implementation).
     * 
     * @param[in] ptr: the pointer to assign to \ref m_ptr.
     */
    pem_ptr(const std::shared_ptr<pem> &ptr);

    /**\brief Constructor with parameters.
     *
     * This constructor uses non-const pointer, this means that the pointer is copied
     * in \ref m_ptr and \p ptr becomes empty (std implementation).
     *  
     * @param[in] ptr: the pointer to assign to \ref m_ptr.
     */
    pem_ptr(std::shared_ptr<pem> &&ptr);

    /** \brief Constructor with parameters.
     *
     * This constructor build a pointer to the given parameter.
     *  
     * @param[in] state: the \ref pem that *this* (\ref m_ptr) should point.*/
    pem_ptr(const pem &state);

    /**\brief Setter for the field \ref m_ptr.
     *
     * This setter uses const parameter, this means that the pointer is copied
     * and the counter of the shared pointer is increased (std implementation).
     * 
     * @param[in] ptr: the pointer to assign to \ref m_ptr.*/
    void set_ptr(const std::shared_ptr<pem> &ptr);

    /**\brief Setter for the field \ref m_ptr (move constructor).
     * 
     * This setter uses non-const parameter, this means that the pointer is copied
     * in \ref m_ptr and \p ptr becomes empty (std implementation).
     *  
     * @param[in] ptr: the pointer to assign to \ref m_ptr.*/
    void set_ptr(std::shared_ptr<pem> &&ptr);

    /**\brief Getter for the field \ref m_ptr.
     *
     * @return a copy of the pointer \ref m_ptr.*/
    std::shared_ptr<pem> get_ptr() const;

    /**\brief Setter for the field \ref m_id of \ref m_ptr.
     *
     * @param[in] to_set: the \ref pem_id to assign to \ref m_id.*/
    void set_id(pem_id to_set);

    /** \brief Setter for the field \ref m_pointed_id.
     *
     *
     * @param[in] to_set: the \ref event_id to set as \ref m_pointed_id.*/
    void set_pointed_id(event_id to_set);

    /**\brief Setter for the field \ref m_edges of \ref m_ptr.
     *
     * @param[in] to_set: the \ref pem_edges to assign to \ref m_edges.*/
    void set_edges(const pem_edges & to_set);

    /** \brief Function that return the field m_id \ref m_ptr.
     *
     * @return the \ref m_id of the \ref pem pointed by \ref m_ptr.*/
    pem_id get_id() const;

    /** \brief Getter of \ref m_pointed_id.
     *
     *
     * @return the \ref pem_id of *this*.*/
    event_id get_pointed_id() const;

    /** \brief Function that return the field m_edges of the pointed \ref event.
     *
     * @return the \ref m_edges of the \ref event pointed by \ref m_ptr.*/
    const pem_edges& get_edges() const;

    /**
     *\brief The < operator based on the field \ref m_id.
     * Implemented to allow the ordering on set of \ref pem_ptr (used in \ref pem_store).
     *
     * @see pem_store.
     *
     * @param [in] to_compare: the \ref pem_ptr to compare with *this*.
     * @return true: if \p to_compare is smaller than *this*
     * @return false: otherwise.*/
    bool operator<(const pem_ptr & to_compare) const;
    /**
     *\brief The > operator based on the field \ref m_id.
     * Implemented to allow the ordering on set of \ref pem_ptr (used in \ref pem_store).
     *
     * @see pem_store.
     *
     * @param [in] to_compare: the \ref pem_ptr to compare with *this*.
     * @return true: if \p to_compare is bigger than *this*
     * @return false: otherwise.*/
    bool operator>(const pem_ptr & to_compare) const;
    /**
     *\brief The == operator based on the field \ref m_id.
     * Implemented to allow the ordering on set of \ref pem_ptr (used in \ref pem_store).
     *
     * @see pem_store.
     *
     * @param [in] to_compare: the \ref pem_ptr to compare with *this*.
     * @return true: if \p to_compare is equal to *this*
     * @return false: otherwise.*/
    bool operator==(const pem_ptr & to_compare) const;
    /** \brief The copy operator.
     *
     * @param [in] to_assign: the \ref pem_ptr to assign to *this*.
     * @return true: if \p the assignment went ok.
     * @return false: otherwise.*/
    bool operator=(const pem_ptr & to_copy);

    void print()const;

};
