#pragma once

#include <set>

/**
 * \class SetHelper
 * \brief Template-based helper methods for set operations.
 *
 * Provides static methods to facilitate modification of sets.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date April 30, 2025
 */
class SetHelper {
public:
    /**
     * \brief Adds all elements from \p factor2 to \p to_modify.
     * \tparam T Type of set elements.
     * \param[in,out] to_modify The set to which elements are added.
     * \param[in] factor2 The set whose elements are added.
     */
    template <class T>
    static void sum_set(std::set<T>& to_modify, const std::set<T>& factor2) {
        for (const auto& elem : factor2) {
            to_modify.insert(elem);
        }
    }

    /**
     * \brief Removes all elements in \p factor2 from \p to_modify.
     * \tparam T Type of set elements.
     * \param[in,out] to_modify The set from which elements are removed.
     * \param[in] factor2 The set whose elements are removed.
     */
    template <class T>
    static void minus_set(std::set<T>& to_modify, const std::set<T>& factor2) {
        for (const auto& elem : factor2) {
            to_modify.erase(elem);
        }
    }
};