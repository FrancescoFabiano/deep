#include "asp_maker.h"

asp_maker::asp_maker()
{
	m_grounder = domain::get_instance().get_grounder();
	m_already_printed_predicate.insert("");



	m_all_agents = "";
	agent_set::const_iterator it_ags;
	agent_set agents = domain::get_instance().get_agents();
	for (it_ags = agents.begin(); it_ags != agents.end(); it_ags++) {
		m_all_agents += m_grounder.deground_agent(*it_ags);
	}
	m_already_printed_agents_sets.insert("");
}

std::string asp_maker::print_ag_set(const agent_set & ags, std::ofstream & to_print)
{
	std::string ret = "";
	agent_set::const_iterator it_ags;

	for (it_ags = ags.begin(); it_ags != ags.end(); it_ags++) {
		ret += m_grounder.deground_agent(*it_ags);
	}

	if (ret.compare(m_all_agents) == 0) {
		ret = "all_agents";
	}

	if (m_already_printed_agents_sets.insert(ret).second) {
		to_print << "agent_set(" << ret << ").\n";
		for (it_ags = ags.begin(); it_ags != ags.end(); it_ags++) {
			to_print << "contains_ag(" << ret << "," << m_grounder.deground_agent(*it_ags) << ").\n";
		}
	}

	return ret;
}

std::string asp_maker::print_subformula(const fluent & fl, std::ofstream & to_print, const std::string & predicate)
{
	std::string ret = m_grounder.deground_fluent(fl);

	if (m_already_printed_predicate.insert(predicate + ret).second) {
		to_print << predicate << "(" << ret << ").\n";
	}

	return ret;
}

std::string asp_maker::print_subformula(const fluent_set & fluents, std::ofstream & to_print, const std::string & predicate)
{
	std::string ret = "";
	
	fluent_set tmp = fluents;

	if (tmp.size() > 1) {
		
		ret += "and(" + print_subformula(*tmp.begin(), to_print, predicate) + ",";
		tmp.erase(*tmp.begin());
		ret += print_subformula(tmp, to_print, predicate) + ")";

	} else if (tmp.size() == 1) {
		ret += print_subformula(*tmp.begin(), to_print, predicate);
	} else {
		//ret = "";
		std::cerr << "\nError in parsing fluent_set for asp conversion\n";
		exit(1);
	}

	if (m_already_printed_predicate.insert(predicate + ret).second) {
		to_print << predicate << "(" << ret << ").\n";
	}
	return ret;
}

std::string asp_maker::print_subformula(const fluent_formula & ff, std::ofstream & to_print, const std::string & predicate)
{

	std::string ret = "";

	fluent_formula tmp = ff;

	
	if (tmp.size() > 1) {

		ret += "or(" + print_subformula(*tmp.begin(), to_print, predicate) + ",";
		tmp.erase(*tmp.begin());
		ret += print_subformula(tmp, to_print, predicate) + ")";

	} else if (tmp.size() == 1) {
		ret += print_subformula(*tmp.begin(), to_print, predicate);
		//return ret;
	} else {
		ret += "";
	}

	if (m_already_printed_predicate.insert(predicate + ret).second) {
		to_print << predicate << "(" << ret << ").\n";
	}
	return ret;
}


std::string asp_maker::print_subformula(const formula_list & fl, std::ofstream & to_print, const std::string & predicate)
{
	std::string ret = "";

	if (fl.size() > 1) {
		formula_list::const_iterator it_fl;
		for (it_fl = fl.begin(); it_fl != fl.end(); it_fl++) {
			if (std::next(it_fl, 1) != fl.end()) {
				ret += "and(" + print_subformula(*it_fl, to_print, predicate) + ",";
			} else {
				ret += print_subformula(*it_fl, to_print, predicate) + ")";
			}
		}
	} else if (fl.size() == 1) {
		ret += print_subformula(*fl.begin(), to_print, predicate);
		//	return ret;
	} else {
		std::cerr << "\nError in parsing fluent_list for asp conversion\n";
		exit(1);
	}

	if (m_already_printed_predicate.insert(predicate + ret).second) {
		to_print << predicate << "(" << ret << ").\n";
	}
	return ret;
}

std::string asp_maker::print_subformula(const belief_formula & bf, std::ofstream & to_print, const std::string & predicate)
{

	std::string ret = "";

	switch ( bf.get_formula_type() ) {

	case FLUENT_FORMULA:
	{
		//std::cout << "\nDEBUG:FF\n" << std::endl;
		ret += print_subformula(bf.get_fluent_formula(), to_print, predicate);
		break;
	}

	case BELIEF_FORMULA:
		//ret += parse_formula(bf.get_fluent_formula());
		//std::cout << "\nDEBUG:BF\n" << std::endl;
		ret += "b(" + m_grounder.deground_agent(bf.get_agent()) + "," + print_subformula(bf.get_bf1(), to_print, predicate) + ")";

		break;

	case PROPOSITIONAL_FORMULA:
		switch ( bf.get_operator() ) {
		case BF_NOT:
			//std::cout << "\nDEBUG:NOT\n" << std::endl;
			ret += "neg(" + print_subformula(bf.get_bf1(), to_print, predicate) + ")";
			//return !entails(bf.get_bf1(), world);
			break;
		case BF_OR:
			//std::cout << "\nDEBUG:OR\n" << std::endl;
			ret += "or(" + print_subformula(bf.get_bf1(), to_print, predicate) + "," + print_subformula(bf.get_bf2(), to_print, predicate) + ")";
			//return entails(bf.get_bf1(), world) || entails(bf.get_bf2(), world);
			break;
		case BF_AND:
			//std::cout << "\nDEBUG:AND\n" << std::endl;
			ret += "and(" + print_subformula(bf.get_bf1(), to_print, predicate) + "," + print_subformula(bf.get_bf2(), to_print, predicate) + ")";

			//return entails(bf.get_bf1(), world) && entails(bf.get_bf2(), world);
			break;
		case BF_FAIL:
		default:
			std::cerr << "Something went wrong in transforming asp for Propositional formula";
			exit(1);
		}
		break;
	case C_FORMULA:
		//std::cout << "\nDEBUG:COMMON\n" << std::endl;
		ret += "c(" + print_ag_set(bf.get_group_agents(), to_print) + "," + print_subformula(bf.get_bf1(), to_print, predicate) + ")";
		break;
	case BF_EMPTY:
	{
		//ret += print_subformula(bf.get_bf2(), to_print);
		////std::cout << "\nDEBUG:HERE\n" << std::endl;
		break;
	}
	case BF_TYPE_FAIL:
	default:
		std::cerr << "Something went wrong in transforming asp for Propositional formula";
		exit(1);
	}

	if (m_already_printed_predicate.insert(predicate + ret).second) {
		to_print << predicate << "(" << ret << ").\n";
	}
	return ret;
}

void asp_maker::print_all()
{
	std::ofstream result;
	std::string folder = "out/ASP_encoding/";
	system(("mkdir -p " + folder).c_str());
	std::string filename = folder + domain::get_instance().get_name() + ".lp";
	remove(filename.c_str());
	result.open(filename, std::ofstream::out | std::ofstream::app);

	std::string s = domain::get_instance().get_name();
	std::regex r(".+__pl_(\\d+)");
	std::smatch m;
	bool found = std::regex_search(s, m, r);

	int maxT = 0;
	if (found) {
		maxT = std::stoi(m[1]);
	}

	result << "#const maxT  = " << maxT << ".\n";
	result << "#const n_fl  = " << domain::get_instance().get_fluent_number() << ".\n";
	result << "#const n_ag  = " << domain::get_instance().get_agents().size() << ".\n";
	result << "#const n_ags = " << exp2(domain::get_instance().get_agents().size()) - 1 << ".\n\n";

	result << "%time(0..maxT).\n\n";

	print_fluents(result);
	print_agents(result);
	// print_agent_set(result);
	print_actions(result);
	print_initially(result);
	print_goals(result);
	// print_worlds(result);

	//print_formulae(result);

	result.close();

}

void asp_maker::print_fluents(std::ofstream & to_print)
{
	to_print << "% *** FLUENTS ***\n";
	fluent_set::const_iterator it_fls;
	std::string predicate = "formula";
	fluent_set fluents = domain::get_instance().get_fluents();
	for (it_fls = fluents.begin(); it_fls != fluents.end(); it_fls++) {
		if (!helper::is_negate(*it_fls)) {
			to_print << "fluent(" << m_grounder.deground_fluent(*it_fls) << ").\n";
			m_already_printed_predicate.insert(predicate);
			print_subformula(*it_fls, to_print, predicate);
		}
	}
	to_print << "\n\n";
}

void asp_maker::print_actions(std::ofstream & to_print)
{
	to_print << "% *** ACTIONS ***\n";
	action_set::const_iterator it_acs;
	std::string act_name;
	action_set actions = domain::get_instance().get_actions();
	for (it_acs = actions.begin(); it_acs != actions.end(); it_acs++) {
		act_name = it_acs->get_name();
		to_print << "action(" << act_name << ").\n";
		std::string effect;
		std::string cond_eff;
		fluent_formula::const_iterator it_flf;
		fluent_set::const_iterator it_fls;

		std::string predicate1 = "formula";
		m_already_printed_predicate.insert(predicate1);

		std::string predicate2 = "ce_formula";
		m_already_printed_predicate.insert(predicate2);



		effects_map::const_iterator it_mapeff;
		switch ( it_acs->get_type() ) {
		case ONTIC:
			to_print << "ontic(" << act_name << ").\n";
			if (it_acs->get_effects().size() > 0) {


				for (it_mapeff = it_acs->get_effects().begin(); it_mapeff != it_acs->get_effects().end(); it_mapeff++) {
					for (it_flf = it_mapeff->first.begin(); it_flf != it_mapeff->first.end(); it_flf++) {
						for (it_fls = it_flf->begin(); it_fls != it_flf->end(); it_fls++) {

							effect = print_subformula(*it_fls, to_print, predicate1);
							cond_eff = print_subformula(it_mapeff->second, to_print, predicate2);

							to_print << "causes(" << act_name << ", ";

							to_print << effect << ", ";



							if (!cond_eff.empty()) {
								to_print << cond_eff;
							} else {
								to_print << "true";
							}

							to_print << ").\n";
						}
					}
				}
			}

			break;
		case SENSING:
			to_print << "sensing(" << act_name << ").\n";
			if (it_acs->get_effects().size() > 0) {


				for (it_mapeff = it_acs->get_effects().begin(); it_mapeff != it_acs->get_effects().end(); it_mapeff++) {
					for (it_flf = it_mapeff->first.begin(); it_flf != it_mapeff->first.end(); it_flf++) {
						for (it_fls = it_flf->begin(); it_fls != it_flf->end(); it_fls++) {

							effect = print_subformula(*it_fls, to_print, predicate1);
							cond_eff = print_subformula(it_mapeff->second, to_print, predicate2);
							to_print << "determines(" << act_name << ", ";

							to_print << effect << ", ";



							if (!cond_eff.empty()) {
								to_print << cond_eff;
							} else {
								to_print << "true";
							}

							to_print << ").\n";
						}
					}
				}
			}


			break;
		case ANNOUNCEMENT:
			to_print << "announcement(" << act_name << ").\n";

			if (it_acs->get_effects().size() > 0) {


				for (it_mapeff = it_acs->get_effects().begin(); it_mapeff != it_acs->get_effects().end(); it_mapeff++) {
					for (it_flf = it_mapeff->first.begin(); it_flf != it_mapeff->first.end(); it_flf++) {
						for (it_fls = it_flf->begin(); it_fls != it_flf->end(); it_fls++) {

							effect = print_subformula(*it_fls, to_print, predicate1);
							cond_eff = print_subformula(it_mapeff->second, to_print, predicate2);
							to_print << "announces(" << act_name << ", ";

							to_print << effect << ", ";



							if (!cond_eff.empty()) {
								to_print << cond_eff;
							} else {
								to_print << "true";
							}

							to_print << ").\n";
						}
					}
				}
			}

			break;
		default:
			to_print << "ontic(" << act_name << ").\n";
			break;
		}

		std::string predicate3 = "formula";
		m_already_printed_predicate.insert(predicate3);

		//it_acs->get_executability()
		if (it_acs->get_executability().size() > 0) {
			std::string ex_conditions = print_subformula(it_acs->get_executability(), to_print, predicate3);
			print_subformula(it_acs->get_executability(), to_print, predicate1);
			to_print << "executable(" << act_name << ", ";

			if (!ex_conditions.empty()) {
				to_print << ex_conditions;
			} else {
				to_print << "true";
			}
			to_print << ").\n";
		}



		std::string predicate4 = "formula";
		m_already_printed_predicate.insert(predicate4);
		agent_set oblivious = domain::get_instance().get_agents();

		observability_map::const_iterator it_obs;
		observability_map tmp_map = it_acs->get_fully_observants();
		for (it_obs = tmp_map.begin(); it_obs != tmp_map.end(); it_obs++) {

			oblivious.erase(it_obs->first);

			std::string obs_conditions = print_subformula(it_obs->second, to_print, predicate4);
			print_subformula(it_obs->second, to_print, predicate1);
			to_print << "observes(" << act_name << ", ";
			to_print << m_grounder.deground_agent(it_obs->first);
			if (!obs_conditions.empty()) {
				to_print << ", " << obs_conditions;
			} else {
				to_print << ", true";
			}
			to_print << ").\n";
		}

		tmp_map = it_acs->get_partially_observants();
		for (it_obs = tmp_map.begin(); it_obs != tmp_map.end(); it_obs++) {

			oblivious.erase(it_obs->first);
			std::string obs_conditions = print_subformula(it_obs->second, to_print, predicate4);
			print_subformula(it_obs->second, to_print, predicate1);
			to_print << "aware_of(" << act_name << ", ";
			to_print << m_grounder.deground_agent(it_obs->first);
			if (!obs_conditions.empty()) {
				to_print << ", " << obs_conditions;
			} else {
				to_print << ", true";
			}
			to_print << ").\n";
		}

		agent_set::const_iterator it_ags;
		for (it_ags = oblivious.begin(); it_ags != oblivious.end(); it_ags++) {
			to_print << "oblivious(" << act_name << ", ";
			to_print << m_grounder.deground_agent(*it_ags);
			to_print << ").\n";
		}


		to_print << "\n\n";
	}
	to_print << "\n\n";
}

void asp_maker::print_agents(std::ofstream & to_print) const
{
	to_print << "% *** AGENTS ***\n";
	agent_set::const_iterator it_ags;
	agent_set agents = domain::get_instance().get_agents();
	for (it_ags = agents.begin(); it_ags != agents.end(); it_ags++) {
		to_print << "agent(" << m_grounder.deground_agent(*it_ags) << ").\n";
	}
	to_print << "\n\n";
}

void asp_maker::print_initially(std::ofstream & to_print)
{
	to_print << "% *** INITIALLY ***\n";
	initially ini_conditions = domain::get_instance().get_initial_description();

	to_print << "\n% ** initial pointed **\n";
	//fluent_formula ini_ff = ini_conditions.get_pointed_world_conditions();
	fluent_set::const_iterator it_fs;
	fluent_set fs = *ini_conditions.get_pointed_world_conditions().begin();


	std::string predicate1 = "formula";
	m_already_printed_predicate.insert(predicate1);
	std::string predicate2 = "formula";
	m_already_printed_predicate.insert(predicate2);
	std::string predicate3 = "init_sf";
	m_already_printed_predicate.insert(predicate3);

	std::string ini_string;
	for (it_fs = fs.begin(); it_fs != fs.end(); it_fs++) {
		ini_string = print_subformula(*it_fs, to_print, predicate1);
		to_print << "initially(" << ini_string << ").\n";
	}

	to_print << "\n% ** initial beliefs **\n";

	formula_list::const_iterator it_fl;
	for (it_fl = ini_conditions.get_initial_conditions().begin(); it_fl != ini_conditions.get_initial_conditions().end(); it_fl++) {
		ini_string = print_subformula(*it_fl, to_print, predicate2);
		to_print << "initially(" << ini_string << ").\n";
	}
	to_print << "\n\n";

	to_print << "\n% ** commonly known fluent formulae **\n";
	//formula_list::const_iterator it_fl;
	for (it_fl = ini_conditions.get_initial_conditions().begin(); it_fl != ini_conditions.get_initial_conditions().end(); it_fl++) {
		if (it_fl->get_formula_type() == C_FORMULA) {
			//ini_string = 
			if (it_fl->get_bf1().get_formula_type() == FLUENT_FORMULA) {
				print_subformula(it_fl->get_bf1(), to_print, predicate3);
				//to_print << "init_sf(" << ini_string << ").\n";
			}
			//if (ini_string.compare("") != 0) {
			//	to_print << "init_sf(" << ini_string << ").\n";
			//}
		}
	}
	to_print << "\n\n";
}

void asp_maker::print_goals(std::ofstream & to_print)
{
	to_print << "% *** GOALS ***\n";

	formula_list goals = domain::get_instance().get_goal_description();
	formula_list::const_iterator it_fl;

	std::string predicate = "formula";
	m_already_printed_predicate.insert(predicate);

	std::string goal_string;
	for (it_fl = goals.begin(); it_fl != goals.end(); it_fl++) {
		goal_string = print_subformula(*it_fl, to_print, predicate);
		to_print << "goal(" << goal_string << ").\n";
	}
	to_print << "\n\n";
}

void asp_maker::print_agent_set(std::ofstream & to_print) const
{
	to_print << "% *** AGENT SETS ***\n";

	// agent_set permutation;
	// int ag_number = 1;
	// print_all_agent_set(permutation, 0, ag_number, to_print);

	int n = domain::get_instance().get_agents().size(), k, ags = 1;

	to_print << "agent_set(1..n_ags).\n\n";

	for (k = 1; k <= n; k++) {
		std::string bitmask(k, 1); // K leading 1's
		bitmask.resize(n, 0); // N-K trailing 0's

		// print integers and permute bitmask
        int size_agent = helper::lenght_to_power_two(n);
		do {
			for (int i = 0; i < n; ++i) // [0..N-1] integers
			{
                boost::dynamic_bitset<> agent_bits(size_agent,i);
				if (bitmask[i]) to_print << "contains_ag(" << ags << ", " << m_grounder.deground_agent(agent_bits) << ").\n";
			}
			to_print << "\n";
			ags++;
		} while (std::prev_permutation(bitmask.begin(), bitmask.end()));
	}
}

/*From https://www.geeksforgeeks.org/generate-all-the-binary-strings-of-n-bits/ since is like generating all the binary numbers.*/
void asp_maker::print_all_fluent_set(fluent_set& permutation, unsigned int index, int & permutation_number, std::ofstream & to_print) const
{
    int bit_size = helper::lenght_to_power_two(domain::get_instance().get_fluent_number());
	fluent_set::const_iterator it_fls;
	if (index / 2 == domain::get_instance().get_fluent_number()) {
		to_print << "fluent_set(" << permutation_number << ").\t%";
		for (it_fls = permutation.begin(); it_fls != permutation.end(); it_fls++) {
			if (!helper::is_negate(*it_fls)) {
				to_print << " ";
			}
			to_print << m_grounder.deground_fluent(*it_fls);
			if (std::next(it_fls, 1) != permutation.end()) {
				to_print << ", ";
			}
		}
		to_print << "\n";

		for (it_fls = permutation.begin(); it_fls != permutation.end(); it_fls++) {
			if (!helper::is_negate(*it_fls)) {
				to_print << "holds(" << permutation_number << ", " << m_grounder.deground_fluent(*it_fls) << ").\n";
			}
		}
		to_print << "\n";

		permutation_number++;

		return;
	}
	fluent_set permutation_2 = permutation;
	//Add the \ref fluent in positive version
    boost::dynamic_bitset<> bitSetToFindPositve(bit_size,index);
    bitSetToFindPositve.set(bitSetToFindPositve.size()-1,0);
	permutation.insert(bitSetToFindPositve);
	print_all_fluent_set(permutation, index + 2, permutation_number, to_print);

	//Add the \ref fluent in negative version
    boost::dynamic_bitset<> bitSetToFindNegative(bit_size,index);
    bitSetToFindNegative.set(bitSetToFindPositve.size() - 1, 1);
	permutation_2.insert(bitSetToFindNegative);
	print_all_fluent_set(permutation_2, index + 2, permutation_number, to_print);
}

void asp_maker::print_worlds(std::ofstream & to_print) const
{
	to_print << "% *** WORLDS ***\n";

	fluent_set permutation;
	int fl_number = 1;
	print_all_fluent_set(permutation, 0, fl_number, to_print);
}

//
//void asp_maker::minus_set(agent_set & to_modify, const agent_set & factor2) const
//{ /**\todo move to helper*/
//
//	agent_set::const_iterator it_kwset;
//	for (it_kwset = factor2.begin(); it_kwset != factor2.end(); it_kwset++) {
//
//		to_modify.erase(*it_kwset);
//	}
//}