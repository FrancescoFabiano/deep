/**
 * \brief Implementation of \ref pstore_opt.h
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date September 14, 2019
 */

#include "pstore_opt.h"

pstore_opt::pstore_opt()
{
}

pstore_opt& pstore_opt::get_instance()
{
	static pstore_opt instance;
	return instance;
}

const pstate_opt_ptr pstore_opt::add_state(const pstate_opt & to_add)
{
	//It returns the pointer to the newly inserted element if it doesn't exit (The old one otherwise)
	//The pair is <iterator,bool>
	//@TODO: Make sure is a shared_ptr
	//return *((m_created_states.insert(state))->first);



	/*// \bug No need to create the pstate_opt pp or the iterator.
	 * pstate_opt_set::iterator it_pwset;
	 * it_pwset = std::get<0>(m_created_states.insert(to_add));
	 * pstate_opt pp = *it_kpset;
	 * return std::shared_ptr<const pstate_opt>(&pp);*/

	//return std::shared_ptr<const pstate_opt>(&(*(std::get<0>(m_created_states.insert(to_add)))));
	auto tmp_ptr = std::make_shared<pstate_opt>(*(std::get<0>(m_created_states.insert(to_add))));
	//*tmp_ptr = tmp;
	return tmp_ptr;
}
