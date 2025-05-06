/*
 * \brief Implementation of \ref pstore.h
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date September 14, 2019
 */


#include "pstore.h"

pstore::pstore()
{
}

pstore& pstore::get_instance()
{
	static pstore instance;
	return instance;
}

const pworld_ptr pstore::add_world(const pworld & to_add)
{
	//It returns the pointer to the newly inserted element if it doesn't exit (The old one otherwise)
	//The pair is <iterator,bool>
	//@TODO: Make sure is a shared_ptr
	//return *((m_created_worlds.insert(world))->first);



	// \bug No need to create the pworld pp or the iterator.
	/*  pworld_set::iterator it_pwset;
	  it_pwset = std::get<0>(m_created_worlds.insert(to_add));
	  //pworld pp = *it_kpset;
	  return std::shared_ptr<const pworld>(&pp);*/

	//return std::shared_ptr<const pworld>(&(*(std::get<0>(m_created_worlds.insert(to_add)))));
	//return std::make_shared<const pworld>(*(std::get<0>(m_created_worlds.insert(to_add))));
	//*tmp_ptr = tmp;
	//return tmp_ptr;

    auto tmp_ptr = std::make_shared<const pworld>(*(std::get<0>(m_created_worlds.insert(to_add))));
    //*tmp_ptr = tmp;
    return tmp_ptr;
}
