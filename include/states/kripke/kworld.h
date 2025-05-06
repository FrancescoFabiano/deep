/**
 * \class kworld
 * 
 * \brief Class that describes a world (node) of a Kripke structure (\ref kstate).
 *
 * \details   A world in a \ref kstate represent a consistent set of \ref fluent (AKA a \ref fluent_set);
 *            that is a possible configuration of the real world (only based on the \ref fluent).
 * 
 * @see kstate, kedge, and kstore.
 *
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date March 31, 2019
 */
#pragma once

#include <set>

#include "../../utilities/define.h"
#include "../../utilities/printer.h"

class kworld
{
    friend class kworld_ptr;

private:
    /** \brief The set of  \ref fluent that describes how these are interpreted in *this*.*/
    fluent_set m_fluent_set;
    /** \brief The unique id of *this* computed with \ref hash_fluents_into_id().*/
    kworld_id m_id;
    /** \brief The unique numerical id of *this* computed with \ref hash_fluents_into_numerical_id().*/
    unsigned int m_numerical_id;

    /** \brief Function used to hash the the info of an edge in a unique id.
     * 
     * @param[in] description: the interpretation of the \ref fluent in the world.
     * @return the unique id of the world.
     * 
     * @warning Useless if not moved to \ref kstore.*/
    kworld_id hash_fluents_into_id(const fluent_set& description);

    /** \brief Function used to hash the the info of *this* in a unique id.
     *
     * @return the unique id of *this*. */
    kworld_id hash_fluents_into_id();

    /** \brief Function used to hash the the info of an edge in a unique numerical id.
     * 
     * @param[in] description: the interpretation of the \ref fluent in the world.
     * @return the unique numerical id of the world.
     * 
     * @warning Useless if not moved to \ref kstore.*/
    unsigned int hash_fluents_into_numerical_id(const fluent_set& description);
    /** \brief Function used to hash the the info of *this* in a unique numerical id.
     *
     * @return the unique numerical id of *this*. */
    unsigned int hash_fluents_into_numerical_id();

    /** \brief Setter for the field \ref m_fluent_set.
     * 
     * This function also guarantees consistency.
     * 
     * @param[in] description: the set of \ref fluent to set as \ref m_fluent_set.
     * 
     * \todo is the parameter passing the best one? Copy?
     * \todo is the consistency useless?*/
    void set_fluent_set(const fluent_set & description);

    /**
     * \brief Function that checks if the given \ref fluent_set is consistent.
     * 
     * @param[in] to_check: the set of \ref fluent to check.
     * 
     * @return true if \p to_check is consistent.
     * @return false if \p to_check is not consistent.
     * 
     * \todo Add \ref static_law as static class and use it to check.
     */
    bool consistent(const fluent_set & to_check);

    /** \brief Function that uses the info of *this* to set its \ref m_id.*/
    void set_id();

    /** \brief Function that uses the info of *this* to set its \ref m_numerical_id.*/
    void set_numerical_id();

    /** \brief Getter of \ref m_fluent_set.
     * 
     * Only accessible by the \ref kworld_ptr.
     *     
     * @return the \ref kworld_ptr to the world where *this* is from.*/
    const fluent_set & get_fluent_set() const;

    /** \brief Getter of \ref m_id.
     *     
     * Only accessible by the \ref kworld_ptr.
     * 
     * @return the int that is the unique id of *this*.*/
    kworld_id get_id() const;
    /** \brief Getter of \ref m_numerical_id.
     *     
     * Only accessible by the \ref kworld_ptr.
     * 
     * @return the int that is the unique id of *this*.*/
    int get_numerical_id() const;

public:
    /** \brief Empty constructor, call the default constructor of all the fields.*/
    kworld();

    /** \brief Constructor with parameters.
     *Construct an object with the given info and then set the unique id.
     * the set of \ref fluent to set as \ref m_fluent_set.
     * 
     * @param[in] description: the set of \ref fluent to set as \ref m_fluent_set.
     * 
     * \todo is the parameter passing the best one? Copy?*/
    kworld(const fluent_set & description);

    /** \brief Copy constructor.
     * 
     * @param[in] world: the \ref kworld to copy into *this*.*/
    kworld(const kworld & world);

    /** \brief Function that check the entailment of a single \ref fluent in *this*.
     * 
     * A single \ref fluent is entailed if is present in *this*. Given that the each kworld
     * should only admit consistent set of \ref fluent if the \ref fluent itself is presented in *this*.
     * 
     * @param[in] to_check: the \ref fluent that has to be checked if entailed by *this* (if is present in *this*).
     *
     * @return true: \p to_check is entailed;
     * @return false: \p -to_check is entailed.
     * 
     * \todo To implement also whit \ref kworld_ptr to the \ref kworld?
     * \todo check consistency on constructor?*/
    bool entails(fluent to_check) const;
    /**
     *\brief Function that check the entailment of a conjunctive set of \ref fluent in *this*.
     * 
     * This is checked by calling recursively \ref entails(fluent) const on all the element
     * of \p to_check and returning true only if all the elements are entailed (conjunctive set of \ref fluent).
     *  
     * @param[in] to_check: the conjunctive set of \ref fluent that has to be checked if entailed by *this*.
     * 
     *
     * @return true: \p to_check is entailed;
     * @return false: \p -to_check is entailed.*/
    bool entails(const fluent_set& to_check) const;
    /**
     *\brief Function that check the entailment of a DNF \ref fluent_formula in *this*.
     * 
     * This is checked by calling recursively \ref entails(const fluent_set &) const on all the element
     * of \p to_check and returning true only if at least one element is entailed (DNF).
     *  
     * @param[in] to_check: the DNF formula that has to be checked if entailed by *this*.
     * 
     *
     * @return true: \p to_check is entailed;
     * @return false: \p -to_check is entailed.*/
    bool entails(const fluent_formula & to_check) const;

    /**
     *\brief The < operator based on the field \ref m_id.
     * Implemented to allow the ordering on set of \ref kworld (used in \ref kstore).
     * 
     * @see kstore.
     *     
     * @param [in] to_compare: the \ref kworld to compare with *this*.
     * @return true: if \p to_compare is smaller than *this*
     * @return false: otherwise.*/
    bool operator<(const kworld& to_compare) const;

    /**
     *\brief The > operator based on the field \ref m_id. 
     * @see kstore.
     *     
     * @param [in] to_compare: the \ref kworld to compare with *this*.
     * @return true: if \p to_compare is bigger than *this*
     * @return false: otherwise.*/
    bool operator>(const kworld& to_compare) const;

    /** \brief The == operator based on the field \ref m_id.
     *     
     * @param [in] to_compare: the \ref kworld to compare with *this*.
     * @return true: if \p to_compare is equal to *this*
     * @return false: otherwise.*/
    bool operator==(const kworld& to_compare) const;

    /** \brief The copy operator.
     *   
     * @param [in] to_assign: the \ref kworld to assign to *this*.
     * @return true: if \p the assignment went ok.
     * @return false: otherwise.*/
    bool operator=(const kworld& to_assign);

    /** \brief Function used to print all the information of *this*.*/
    void print() const;
};

/**
 * \class kworld_ptr
 * 
 * \brief A wrapper class for a std::shared_pointer to a \ref kworld usually stored in \ref kstore.
 * 
 * This class is necessary so we can use the < operator in the set of \ref kworld_ptr.
 * If we dom't use the wrapper pointers to the same elements are different.
 *
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 29, 2019
 */
class kworld_ptr
{
private:
    /**\brief the pointer that is wrapped by *this*.*/
    std::shared_ptr<const kworld> m_ptr;

    /** \brief The number of repetition of *this* in a \ref kstate due to oblivious obsv.*/
    unsigned short m_repetition = 0;

public:
    /**\brief Constructor without parameters.*/
    kworld_ptr();
    /**\brief Constructor with parameters.
     *
     * This constructor uses const pointer, this means that the pointer is copied
     * and the counter of the shared pointer is increased (std implementation).
     * 
     * @param[in] ptr: the pointer to assign to \ref m_ptr.
     * @param[in] repetition: the value to give to \ref m_repetition, default to 0.
     */
    kworld_ptr(const std::shared_ptr<const kworld> & ptr, unsigned short repetition = 0);
    /**\brief Constructor with parameters.
     *
     * This constructor uses non-const pointer, this means that the pointer is copied
     * in \ref m_ptr and \p ptr becomes empty (std implementation).
     *  
     * @param[in] ptr: the pointer to assign to \ref m_ptr.
     * @param[in] repetition: the value to give to \ref m_repetition, default to 0.
     */
    kworld_ptr(std::shared_ptr<const kworld>&& ptr, unsigned short repetition = 0);

    /** \brief Constructor with parameters.
     *
     * This constructor build a pointer to the given parameter.
     *  
     * @param[in] world: the \ref kworld that *this* (\ref m_ptr) should point.
     * @param[in] repetition: the value to give to \ref m_repetition, default to 0.*/
    kworld_ptr(const kworld & world, unsigned short repetition = 0);

    /**\brief Getter for the field \ref m_ptr.
     *  
     * @return a copy of the pointer \ref m_ptr.*/
    std::shared_ptr<const kworld> get_ptr() const;

    /**\brief Setter for the field \ref m_ptr.
     *
     * This setter uses const parameter, this means that the pointer is copied
     * and the counter of the shared pointer is increased (std implementation).
     * 
     * @param[in] ptr: the pointer to assign to \ref m_ptr.*/
    void set_ptr(const std::shared_ptr<const kworld> & ptr);
    /**\brief Setter for the field \ref m_ptr (move constructor).
     * 
     * This setter uses non-const parameter, this means that the pointer is copied
     * in \ref m_ptr and \p ptr becomes empty (std implementation).
     *  
     * @param[in] ptr: the pointer to assign to \ref m_ptr.*/
    void set_ptr(std::shared_ptr<const kworld>&& ptr);

    /** \brief Setter for the field \ref m_repetition.
     * 
     * @param[in] repetition: the value to give to \ref m_repetition.*/
    void set_repetition(unsigned short repetition);

    /** \brief Getter of the field \ref m_repetition.
     * 
     * @return the value to of \ref m_repetition.*/
    unsigned short get_repetition() const;

    /** \brief Function that return the field m_fluent_set of the pointed \ref kworld.
     *     
     * @return the \ref fluent_set that is the description of the \ref kworld pointed by \ref m_ptr.*/
    const fluent_set & get_fluent_set() const;

    /** \brief Function that return the field m_id of the pointed \ref kworld.
     *     
     * @return the \ref kworld_id that is the id of the \ref kworld pointed by \ref m_ptr*/
    kworld_id get_fluent_based_id() const;

    /** \brief Function that return the field m_id of the pointed \ref kworld + \ref m_repetition.
     *     
     * @return the \ref kworld_id that is the id of the \ref kworld pointed by \ref m_ptr + \ref m_repetition.*/
    kworld_id get_id() const;

    /** \brief Function that return the field m_numerical_id of the pointed \ref kworld + \ref m_repetition.
     *     
     * @return the numerical id that is the id of the \ref kworld pointed by \ref m_ptr + \ref m_repetition.*/
    int get_numerical_id() const;


    /** \brief Function that check the entailment of a single \ref fluent in \ref m_ptr.
     * 
     * 
     * @param[in] to_check: the \ref fluent that has to be checked if entailed by \ref m_ptr (if is present in \ref m_ptr).
     *
     * @return true: \p to_check is entailed;
     * @return false: \p -to_check is entailed.*/
    bool entails(fluent to_check) const;
    /**
     *\brief Function that check the entailment of a conjunctive set of \ref fluent in \ref m_ptr.
     * 
     * @param[in] to_check: the conjunctive set of \ref fluent that has to be checked if entailed by \ref m_ptr.
     *
     * @return true: \p to_check is entailed;
     * @return false: \p -to_check is entailed.*/
    bool entails(const fluent_set& to_check) const;
    /**
     *\brief Function that check the entailment of a DNF \ref fluent_formula in \ref m_ptr.
     * 
     * @param[in] to_check: the DNF formula that has to be checked if entailed by \ref m_ptr.
     *
     * @return true: \p to_check is entailed;
     * @return false: \p -to_check is entailed.*/
    bool entails(const fluent_formula & to_check) const;

    /**\brief The operator =.
     *
     * This operator assign the parameter without destroying \p to_copy.
     * 
     * @param[in] to_copy: the \ref kworld_ptr to assign to *this*.
     * @return true: if the assignment went through.
     * @return false: otherwise.*/
    bool operator=(const kworld_ptr & to_copy);
    /**\brief The operator < for set operation.
     *
     * The ordering is based on the pointed object and not on the pointer itself so we have one
     * copy of each pointed object.
     * 
     * @param[in] to_compare: the \ref kworld_ptr to check for odering.
     * 
     * @return true: if *this* is smaller than \p to_compare.
     * @return false: otherwise.*/

    bool operator<(const kworld_ptr & to_compare) const;
    /**\brief The operator > \ref kstate operator <.
     *
     * 
     * @param[in] to_compare: the \ref kworld_ptr to check for odering.
     * 
     * @return true: if *this* is bigger than \p to_compare.
     * @return false: otherwise.*/

    bool operator>(const kworld_ptr & to_compare) const;
    /**\brief The operator ==.
     * 
     * @param[in] to_compare: the \ref kworld_ptr to confront with *this*.
     * @return true: if *this* is equal to \p to_compare.
     * @return false: otherwise.*/
    bool operator==(const kworld_ptr & to_compare) const;
};

