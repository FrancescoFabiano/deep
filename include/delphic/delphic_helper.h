/**
 * \brief Class that contains the union update function.
 *
 * \details The union update returns the possibility after the execution of a PEM
 *
 * @see pstate and event.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 29, 2021
 */
#pragma once

#include "../states/kripke/kstate.h"

#include "../states/possibilities/pstate.h"
#include "../utilities/define.h"

class delphic_helper
{
public:
    /** \brief Calculates the PEM relative to the \ref action description and the current \ref pstate.
     *
     * @param [in] act: the \ref action description.
     * @return the resulting \ref pem.*/
    static pem_ptr get_pem(const action & act);
    /** \brief Calculates the \ref pstate resulting from the given current \ref pstate and \ref pem.
     *
     * @param [in] state: the current \ref pstate.
     * @param [in] e: the \ref pem to apply.
     * @return the updated \ref pstate.*/
    static const pstate & union_update(const pstate & state, const action & act);
    static const pworld_ptr & union_update_helper(pstate & ret, const pstate & state, const action & act, const pem_ptr & pem, const pworld_ptr & pw, const event_ptr & ev, update_map & u_map, const agent_group_map & a_map);

    static const kstate & union_update(const kstate & state, const action & act);


    //Francesco: addition
    static const pworld & world_cartesian_product(const pworld & world, const event_ptr & e);
    static fluent_formula get_total_effects(const pstate & state, const action & act, const event_ptr & e);
    static formula_list get_total_pre(const pstate & state, const action & act, const event_ptr & e);
};
