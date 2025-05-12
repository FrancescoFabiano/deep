/**
 * \brief Class that contains template-based helper methods.
 *
 *  The class implements static methods to facilitate
 *  the modification of the formulae and other.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 30, 2021
 */

#pragma once

class helper_t {
public:
    /** \brief Function that return the sum_set of the two parameters by modifying the first one.
     *
     *
     * @param[out] to_modify: the set in which is added \p factor2.
     * @param[in] factor2: the set to add to \p to_modify.*/
    template <class T>
    static void sum_set(std::set<T> & to_modify, const std::set<T> & factor2);
    /** \brief Function that return the set difference of the two parameters by modifying the first one.
     *
     *
     * @param[out] to_modify: the set from which is removed \p factor2.
     * @param[in] factor2: the set to remove from \p to_modify.*/
    template <class T>
    static void minus_set(std::set<T> & to_modify, const std::set<T> & factor2);
};
