/*
 * \brief Implementation of \ref pworld.h and \ref pworld_ptr.h.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date September 14, 2019
 */
#include <boost/dynamic_bitset.hpp>
#include "pworld.h"

#include "pstore.h"

#include <stdexcept>
#include "../../utilities/helper.h"
pworld::pworld()
{
}

pworld::pworld(const fluent_set & description)
{
	set_fluent_set(description);
	set_id();
}
//generate an unique id given the state information -> the literals

pworld::pworld(const pworld & world)
{
	set_fluent_set(world.get_fluent_set());
	set_id();
}

boost::dynamic_bitset<> concatStringDyn( const boost::dynamic_bitset<>& bs1,const boost::dynamic_bitset<>& bs2)
{
    std::string s1;
    std::string s2;

    to_string(bs1,s1);
    to_string(bs2,s2);
    boost::dynamic_bitset<> res(s1+s2);
    return res;
}

boost::dynamic_bitset<> concatOperatorsDyn( const boost::dynamic_bitset<>& bs1,const boost::dynamic_bitset<>& bs2)
{
    boost::dynamic_bitset<> bs1Copy(bs1);
    boost::dynamic_bitset<> bs2Copy(bs2);
    size_t totalSize=bs1.size()+bs2.size();
    bs1Copy.resize(totalSize);
    bs2Copy.resize(totalSize);
    bs1Copy<<=bs2.size();
    bs1Copy|=bs2Copy;
    return bs1Copy;
}

boost::dynamic_bitset<> concatLoopDyn( const boost::dynamic_bitset<>& bs1,const boost::dynamic_bitset<>& bs2)
{
    boost::dynamic_bitset<> res(bs1);
    res.resize(bs1.size()+bs2.size());
    size_t bs1Size=bs1.size();

    for(size_t i=0;i<bs2.size();i++)
        res[i+bs1Size]=bs2[i];
    return res;
}

pworld_id pworld::hash_fluents_into_id(const fluent_set& fl)
{
    fluent_set fl2 = fl;
    std::size_t hash;
    hash = boost::hash_range(fl2.begin(),fl2.end());
    return hash;

}

pworld_id pworld::hash_fluents_into_id()
{
	return hash_fluents_into_id(m_fluent_set);
}

void pworld::set_fluent_set(const fluent_set & description)
{
	/*
	 * \throw std::invalid_argument whenever \p description is not consistent.
	 *
	if (consistent(description))*/
	m_fluent_set = description;
	/*else
		throw std::invalid_argument("Non consistent set of fluent");*/
}

bool pworld::consistent(const fluent_set & to_check)
{
	fluent_set::const_iterator it_flset;
	fluent_set::const_iterator it_flset_tmp;

	for (it_flset = to_check.begin(); it_flset != to_check.end(); it_flset++) {
		/* If the pointed fluent is in modulo 2 it means is the positive and if
		 * its successor (the negative version) is in the set then is not consistent.*/
		if (it_flset->test(it_flset->size()-1)) {
			//The std::set has is elements ordered so we can just check its successor.
			it_flset_tmp = it_flset;
			it_flset_tmp++;
		}
		if (it_flset_tmp != to_check.end()) {
			if (it_flset_tmp == ++((it_flset))) {
				std::cout << "\nCheck: " << *it_flset_tmp << " and " << (*(++it_flset)) << std::endl;
				return false;
			}
		}
		/**\todo check that all the possible \ref fluent are there.
		 * \bug change and use find.
		 */
	}
	return true;
}

void pworld::set_id()
{
	m_id = hash_fluents_into_id();
}

const fluent_set & pworld::get_fluent_set() const
{
	return m_fluent_set;
}

pworld_id pworld::get_id() const
{
	return m_id;
}

bool pworld::entails(fluent to_check) const
{
	return(m_fluent_set.find(to_check) != m_fluent_set.end());
}

/**
 * \todo Check for the size = 0?
 */
bool pworld::entails(const fluent_set & to_check) const
{
	//fluent_set expresses conjunctive set of \ref fluent
	if (to_check.size() == 0) {
		return true;
	}
	fluent_set::const_iterator it_fl;
	for (it_fl = to_check.begin(); it_fl != to_check.end(); it_fl++) {
		if (!entails(*it_fl)) {
			return false;
		}
	}
	return true;
}

/**
 * \todo Check for the size = 0?
 */
bool pworld::entails(const fluent_formula & to_check) const
{
	if (to_check.size() == 0) {
		return true;
	}
	fluent_formula::const_iterator it_fl;
	for (it_fl = to_check.begin(); it_fl != to_check.end(); it_fl++) {
		if (entails(*it_fl)) {
			return true;
		}
	}
	return false;
}

bool pworld::operator<(const pworld& to_compare) const
{

	if (m_id < to_compare.get_id())
		return true;

	return false;
}

bool pworld::operator>(const pworld& to_compare) const
{
	if (m_id > (to_compare.get_id()))
		return true;
	return false;
}

bool pworld::operator==(const pworld& to_compare) const
{
	/**std way*/
	if (!((*this) < to_compare) && !(to_compare < (*this))) {
		return true;
	}
	return false;
}

bool pworld::operator=(const pworld & to_assign)
{
	set_fluent_set(to_assign.get_fluent_set());
	set_id();
	return true;
}

void pworld::print() const
{
	std::cout << "\nFluents: " << get_id();
	printer::get_instance().print_list(m_fluent_set);
}

/*-***************************************************************************************************************-*/

pworld_ptr::pworld_ptr()
{
}

pworld_ptr::pworld_ptr(const std::shared_ptr<const pworld> & ptr, unsigned short repetition)
{
	set_ptr(ptr);
	set_repetition(repetition);
}

pworld_ptr::pworld_ptr(std::shared_ptr<const pworld>&& ptr, unsigned short repetition)
{
	set_ptr(ptr);
	set_repetition(repetition);

}

pworld_ptr::pworld_ptr(const pworld & world, unsigned short repetition)
{
	m_ptr = std::make_shared<pworld>(world);
	set_repetition(repetition);

}

void pworld_ptr::set_ptr(const std::shared_ptr<const pworld> & ptr)
{
	m_ptr = ptr;
}

void pworld_ptr::set_ptr(std::shared_ptr<const pworld>&& ptr)
{
	m_ptr = ptr;
}

std::shared_ptr<const pworld> pworld_ptr::get_ptr() const
{
	return m_ptr;
}

const fluent_set & pworld_ptr::get_fluent_set() const
{
	if (m_ptr != nullptr) {
		return get_ptr()->get_fluent_set();
	}
	std::cerr << "Error in creating a pworld_ptr\n";
	exit(1);
}

pworld_id pworld_ptr::get_fluent_based_id() const
{
	if (m_ptr != nullptr) {
		return(get_ptr()->get_id());
	}
	std::cerr << "\nError in creating a pworld_ptr\n";
	exit(1);
}

pworld_id pworld_ptr::get_id() const
{
	if (m_ptr != nullptr) {
	    pworld_id id = (get_ptr()->get_id());

	    //moltiplico * 10 id + get_repetion() TODO test con shift 
        return boost::hash_value((1000*id)+get_repetition());
	}
	std::cerr << "\nError in creating a pworld_ptr\n";
	exit(1);
}


pworld_id pworld_ptr::get_numerical_id() const
{
	if (m_ptr != nullptr) {
	    pworld_id id = (get_ptr()->get_id());

        return boost::hash_value(id);
	}
	std::cerr << "\nError in creating a pworld_ptr\n";
	exit(1);
}

void pworld_ptr::set_repetition(unsigned short to_set)
{
	m_repetition = to_set;
}

void pworld_ptr::increase_repetition(unsigned short to_increase)
{
	m_repetition = m_repetition + to_increase;
	//m_repetition = m_repetition;
}

unsigned short pworld_ptr::get_repetition() const
{
	return m_repetition;

}

bool pworld_ptr::entails(fluent to_check) const
{
	return m_ptr->entails(to_check);
}

bool pworld_ptr::entails(const fluent_set& to_check) const
{
	return m_ptr->entails(to_check);
}

bool pworld_ptr::entails(const fluent_formula & to_check) const
{
	return m_ptr->entails(to_check);
}

bool pworld_ptr::operator<(const pworld_ptr & to_compare) const
{
	if (get_id() < (to_compare.get_id())) {
		return true;
	}
	return false;
}

bool pworld_ptr::operator>(const pworld_ptr & to_compare) const
{
	if (get_id() > to_compare.get_id()) {
		return true;
	}
	return false;
}

bool pworld_ptr::operator==(const pworld_ptr & to_compare) const
{
	/**std way*/
	if (!((*this) < to_compare) && !(to_compare < (*this))) {
		return true;
	}
	return false;
}

bool pworld_ptr::operator=(const pworld_ptr & to_copy)
{
	set_ptr(to_copy.get_ptr());
	set_repetition(to_copy.get_repetition());
	return true;
}