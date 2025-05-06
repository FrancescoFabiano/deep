/**
 * \brief Implementation of \ref delphic_helper.h.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 29, 2021
 */

#include "delphic_helper.h"
#include "../utilities/helper_t.ipp"
#include "event.h"
#include "pem.h"
#include "pem_store.h"

pem_ptr delphic_helper::get_pem(const action & act)
{
	pem_id p_id;
	switch ( act.get_type() ) {
	case ONTIC:
	{
		//Hard-coded
		p_id = 2;
		break;
	}
	case SENSING:
	case ANNOUNCEMENT:
	{
		//Hard-coded
		p_id = 1;
		break;
	}
	default:
		std::cerr << "\nError: wrong action specification.\n";
		break;
	}

	return pem_store::get_instance().get_pem(p_id);

}

const pstate & delphic_helper::union_update(const pstate & state, const action & act)
{
    pstate ret;
    pem_ptr pem = get_pem(act);
    pworld_ptr p_pw = state.get_pointed();
    event_ptr p_ev = pem_store::get_instance().get_event(pem.get_pointed_id());
    update_map u_map;
    agent_group_map a_map;

    if (!state.entails(get_total_pre(state, act, p_ev), p_pw)) {
        std::cerr << "Action is not executable." << std::endl;
        return ret;
    }

    pworld_ptr new_pointed = union_update_helper(ret, state, act, pem, p_pw, p_ev, u_map, a_map);
    ret.set_pointed(new_pointed);

    return ret;
/*	//DEBUG
	event_id e_id = 2;
	event_ptr event = pem_store::get_instance().get_event(e_id);
	std::cerr << "\nPrinting Action " << act.get_name() << " x Event " << e_id << "!\n";
	std::cerr << "*******Action*******";
	act.print();
	std::cerr << "\n*******Event*******";
	event.print();
	auto entailed_eff = helper::get_effects_if_entailed(act.get_effects(), state);

	std::cerr << "*******Effects*******\n";
	printer::get_instance().print_list(entailed_eff);
	std::cerr << "\n********************\n";

	std::cerr << "Real Preconditions (in the initial state): ";
	printer::get_instance().print_list(delphic_helper::get_total_pre(act.get_executability(), entailed_eff, event));
	std::cerr << "\nReal Postconditions (in the initial state): ";
	printer::get_instance().print_list(delphic_helper::get_total_effects(entailed_eff, event));




	return state;*/
}

const pworld_ptr & delphic_helper::union_update_helper(pstate & ret, const pstate & state, const action & act, const pem_ptr & pem, const pworld_ptr & pw, const event_ptr & ev, update_map & u_map, const agent_group_map & a_map)
{
    if (ev.get_meta_precondition().find(e_meta_condition::none) != ev.get_meta_precondition().end() && ev.get_ontic_change()) {
        u_map.insert(update_map::value_type({pw, ev}, pw));
        return pw;
    }
    
    fluent_set world_description = pw.get_fluent_set();
    // Execute the all the effects
    if (ev.get_ontic_change()) {
        fluent_formula effects = get_total_effects(state, act, ev);
        fluent_formula::const_iterator it_eff;

        for (it_eff = effects.begin(); it_eff != effects.end(); it_eff++) {
            helper::apply_effect(*it_eff, world_description);
        }
    }

    pworld_ptr new_pw = ret.add_rep_world(pworld(world_description), pw.get_repetition());
    u_map.insert(update_map::value_type({pw, ev}, new_pw));

    auto it_pwtm = state.get_beliefs().find(pw);

    if (it_pwtm != state.get_beliefs().end()) {
        pworld_map::const_iterator it_pwm;
        pworld_ptr_set::const_iterator it_pws;

        for (it_pwm = it_pwtm->second.begin(); it_pwm != it_pwtm->second.end(); it_pwm++) {
            auto it_eve = pem.get_edges().find(ev);

            if (it_eve != pem.get_edges().end()) {
                agent ag = it_pwm->first;
                auto it_agm = a_map.find(ag);

                if (it_agm != a_map.end()) {
                    auto it_evm = it_eve->second.find(it_agm->second);
    
                    if (it_evm != it_eve->second.end()) {
                        event_ptr_set::const_iterator it_evs;
    
                        for (it_pws = it_pwm->second.begin(); it_pws != it_pwm->second.end(); it_pws++) {
                            for (it_evs = it_evm->second.begin(); it_evs != it_evm->second.end(); it_evs++) {
                                if (u_map.find({*it_pws, *it_evs}) == u_map.end() &&
                                        state.entails(get_total_pre(state, act, *it_evs), *it_pws)) {
                                    pworld_ptr believed_pw = union_update_helper(ret, state, act, pem, *it_pws, *it_evs, u_map, a_map);
                                    ret.add_edge(new_pw, believed_pw, ag);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return new_pw;
}

const kstate & delphic_helper::union_update(const kstate & state, const action & act)
{
	std::cerr << "\nError: Union update not yet implmente for Kripke structures\n";
	exit(1);
}

const pworld & delphic_helper::world_cartesian_product(const pworld & world, const event_ptr & e)
{
	return world;
}

fluent_formula delphic_helper::get_total_effects(const pstate & state, const action & act, const event_ptr & e)
{
    fluent_formula action_eff = helper::get_effects_if_entailed(act.get_effects(), state);
	auto meta_post = e.get_meta_postconditions();

	if (meta_post.size() > 1) {
		std::cerr << "Error: malformed action postcondition in event " << e.get_id() << std::endl;
		exit(1);
	} else if (meta_post.size() == 0) {
		fluent_formula empty;
		return empty;
	} else {
		switch ( *meta_post.begin() ) {
		case act_eff:
			return action_eff;
		case neg_act_eff:
			return helper::negate_fluent_formula(action_eff);
			//Not break because it means it is empty
			//break;
		case none:
		default:
			fluent_formula empty;
			return empty;
			break;
		}
	}

	//Need to insert the merge with specific postcondition
}

formula_list delphic_helper::get_total_pre(const pstate & state, const action & act, const event_ptr & e)
{

	formula_list ret;
	formula_list action_pre = act.get_executability();
    fluent_formula action_eff = helper::get_effects_if_entailed(act.get_effects(), state);

	auto meta_pre = e.get_meta_precondition();

	for (auto it_meta_pre = meta_pre.begin(); it_meta_pre != meta_pre.end(); ++it_meta_pre) {
		belief_formula tmp_bf;

		switch ( *it_meta_pre ) {
		case act_eff:
			tmp_bf.set_from_ff(action_eff);
			ret.push_back(tmp_bf);
			break;
		case neg_act_eff:
			tmp_bf.set_from_ff(helper::negate_fluent_formula(action_eff));
			ret.push_back(tmp_bf);
			break;
		case act_pre:
			for (auto it_act_pre = action_pre.begin(); it_act_pre != action_pre.end(); ++it_act_pre) {
				ret.push_back(*it_act_pre);
				
			}
			break;
		case none:
		default:
			ret.clear();
			return ret;
			break;
		}
	}

	return ret;

}
