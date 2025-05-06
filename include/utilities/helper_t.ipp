/** \brief Implementation of helper_t.h
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 30, 2021
 */

#include "helper_t.h"

template <class T>
void helper_t::sum_set(std::set<T> & to_modify, const std::set<T> & factor2)
{
    typename std::set<T>::const_iterator it_pwset;
    for (it_pwset = factor2.begin(); it_pwset != factor2.end(); it_pwset++) {
        to_modify.insert(*it_pwset);
    }
}

template <class T>
void helper_t::minus_set(std::set<T> & to_modify, const std::set<T> & factor2)
{
    typename std::set<T>::const_iterator it_pwset;
    for (it_pwset = factor2.begin(); it_pwset != factor2.end(); it_pwset++) {
        to_modify.erase(*it_pwset);
    }
}
