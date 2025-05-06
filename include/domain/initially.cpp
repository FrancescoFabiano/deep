/**
 * \brief Implementation of \ref initially.h
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 6, 2019
 */
#include <iostream>

#include "initially.h"
#include "../utilities/printer.h"

initially::initially()
{
	m_ini_restriction = RESTRICTION_FAIL;
}

initially::initially(domain_restriction ini_restriction)
{
	m_ini_restriction = ini_restriction;
}

domain_restriction initially::get_ini_restriction() const
{
	return m_ini_restriction;
}

bool initially::check_restriction(const belief_formula & bf) //Apply the restriction
{
	bool ret = false;
	switch ( m_ini_restriction ) {
		//We only admit C(belief)
		/**\todo Check if all the agents have to be in the C.*/
		/* The possible cases are:
		 * - *phi* -> all worlds must entail *phi*.
		 * - C(B(i,*phi*)) -> all worlds must entail *phi*.
		 * - C(B(i,*phi*) \ref BF_OR B(i,-*phi*)) -> only edges conditions.
		 * - C(-B(i,*phi*) \ref BF_AND -B(i,-*phi*)) -> only edges conditions.*/
	case S5:
	{
		switch ( bf.get_formula_type() ) {
			//To admit initially [S5_ok AND S5_ok AND ...]  (Also taken care of in the add_initial_condition)

		case PROPOSITIONAL_FORMULA:
		{
			if (bf.get_operator() != BF_AND) {
				ret = false;
			} else {
				ret = check_restriction(bf.get_bf1()) && check_restriction(bf.get_bf2());
			}
			break;
		}

			//*phi*
		case FLUENT_FORMULA:
		{
			ret = true;
			break;
		}

			//C(..)
		case C_FORMULA:
		{
			//C(B(i, *phi*))
			belief_formula tmp = bf.get_bf1();
			switch ( tmp.get_formula_type() ) {

				//C(*phi*)
			case FLUENT_FORMULA:
			{
				ret = true;
				break;
			}
				//C(B(i,*phi*)
			case BELIEF_FORMULA:
			{
				if (tmp.get_bf1().get_formula_type() == FLUENT_FORMULA) {
					ret = true;
				} else {
					ret = false;
				}
				break;
			}
			case PROPOSITIONAL_FORMULA:
			{
				//C(B(i,*phi*) \ref BF_OR B(i,-*phi*))
				//Check if C ( x1 **OR** x2 )
				if (tmp.get_operator() == BF_OR) {
					//Check if **x1** and **x2** are B(i,phi) and B(i,-phi)
					
					ret = helper::check_Bff_notBff(tmp.get_bf1(), tmp.get_bf2(), nullptr);
				} else if (tmp.get_operator() == BF_AND) { //C(-B(i,*phi*) \ref BF_AND -B(i,-*phi*))
					//Check if C ( x1 **AND** x2 )
					//Check if **x1** and **x2** are **NOT**(...)
					belief_formula tmp_nested1 = tmp.get_bf1();
					belief_formula tmp_nested2 = tmp.get_bf2();
					if (tmp_nested1.get_formula_type() == PROPOSITIONAL_FORMULA && tmp_nested2.get_formula_type() == PROPOSITIONAL_FORMULA) {

						if (tmp_nested1.get_operator() == BF_NOT && tmp_nested2.get_operator() == BF_NOT) {
							//Check tmp_nested1 and tmp_nested2 are B(i,phi) and B(i,-phi)
							ret = helper::check_Bff_notBff(tmp_nested1.get_bf1(), tmp_nested2.get_bf1(), nullptr);
						}
					}
				} else {
					ret = false;
				}
				break;
			}
			case BF_EMPTY:
			{
				ret = true;
				break;
			}
			default:
			{
				ret = false;
				break;
			}
			}
			/**\todo Check all the agents are present in C.
			 * \todo Check that all the fluent are considered.*/

			break;
		}
		case BF_EMPTY:
		{
			ret = true;
			break;
		}
		default:
		{
			ret = false;
			break;
		}
		}
		break;
	}
	case K45:
	{
		ret = false;
		break;
	}
	case NONE:
	{
		ret = true;
		break;
	}
	default:
	{
		ret = false;
		break;
	}
	}

	return ret;
}
//This type of parameter is fine because of we add the return value and not ff

void initially::add_pointed_condition(const fluent_formula & ff)
{
	//Is in DNF form so you have to add these to the fluent of before (all of them)
	m_pointed_world_conditions = helper::and_ff(m_pointed_world_conditions, ff);
}

//This type of parameter is fine because of the push back (makes a copy))

void initially::add_initial_condition(const belief_formula & bf)
{
	//Normalize the and and keep only bf_1, bf_2 and not (bf_1 AND bf_2)
	if (bf.get_formula_type() == PROPOSITIONAL_FORMULA && bf.get_operator() == BF_AND) {
		add_initial_condition(bf.get_bf1());
		add_initial_condition(bf.get_bf2());
	} else if (check_restriction(bf)) {
		m_bf_intial_conditions.push_back(bf);
	} else {
		std::cerr << "The initial state does not respect the conditions\n";
		exit(1);
	}

}

const fluent_formula & initially::get_pointed_world_conditions() const
{
	return m_pointed_world_conditions;
}

const formula_list & initially::get_initial_conditions() const
{
	return m_bf_intial_conditions;
}

const fluent_formula & initially::get_ff_forS5() const
{
	return m_ff_forS5;
}

void initially::set_ff_forS5()
{
	if (m_ini_restriction == S5) {
		//The consistency with S5 is already checked
		fluent_formula ret;
		formula_list::const_iterator it_fl;
		for (it_fl = m_bf_intial_conditions.begin(); it_fl != m_bf_intial_conditions.end(); it_fl++) {
			switch ( (*it_fl).get_formula_type() ) {
				//*phi*
			case FLUENT_FORMULA:
			{
				ret = helper::and_ff(ret, (*it_fl).get_fluent_formula());
				break;
			}
				//C(...)
			case C_FORMULA:
			{
				belief_formula tmp = (*it_fl).get_bf1();
				switch ( tmp.get_formula_type() ) {
					//C(*phi*)
				case FLUENT_FORMULA:
				{
					ret = helper::and_ff(ret, tmp.get_fluent_formula());
					fluent_formula tmp_ff = tmp.get_fluent_formula();
					if (tmp_ff.size() == 1) {
						fluent_set tmp_fs = *tmp_ff.begin();
						fluent_set::const_iterator it_fs_tmp;
						for (it_fs_tmp = tmp_fs.begin(); it_fs_tmp != tmp_fs.end(); it_fs_tmp++) {
							m_initially_known_fluents.insert(*it_fs_tmp);
							//std::cerr << "\nDEBUG: "<< *it_fs_tmp << std::endl;
						}
						}
					break;
				}
					//C(B(i,*phi*))
				case BELIEF_FORMULA:
				{
					if (tmp.get_bf1().get_formula_type() == FLUENT_FORMULA) {
						ret = helper::and_ff(ret, tmp.get_bf1().get_fluent_formula());
					} else {
						std::cerr << "\nError in the type of initial formulae (FIRST).\n";
						exit(1);
					}
					break;
				}
					//Only for edges -- expresses that someone is ignorant - just edges.
				case PROPOSITIONAL_FORMULA:
				{
					break;
				}
				default:
				{
					/*The S5_OK and S5_OK is taken care of so no need for PROP_FORM*/
					std::cerr << "\nError in the type of initial formulae (SECOND).\n";
					exit(1);
					break;
				}
				}
				break;
			}
			default:
			{
				std::cerr << "\nError in the type of initial formulae (THIRD).\n";
				exit(1);
				break;
			}
			}
		}
		//So if called twice is still correct
		m_ff_forS5 = ret;
	}
}

const fluent_set & initially::get_initially_known_fluents() const
{
	return m_initially_known_fluents;
}
