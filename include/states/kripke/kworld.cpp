/*
 * \brief Implementation of \ref kworld.h and \ref kworld_ptr.h.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date March 31, 2019
 */

#include "kworld.h"

#include "kstore.h"
#include "../../utilities/helper.h"

#include <stdexcept>

kworld::kworld()
{
}

kworld::kworld(const fluent_set & description)
{
	/*
	 * No need because consistent for construction
	 * \throw std::invalid_argument whenever \p description is not consistent.
	 *
	 *try {*/
	set_fluent_set(description);
	set_id();
	set_numerical_id();
	/*} catch (const std::invalid_argument& ia) {
	 *	throw ia;
	 *} */
}
//generate an unique id given the state information -> the literals

kworld::kworld(const kworld & world)
{
	set_fluent_set(world.get_fluent_set());
	set_id();
	set_numerical_id();
}

kworld_id kworld::hash_fluents_into_id(const fluent_set& fl)
{
	kworld_id ret;
	fluent_set::const_iterator it_fl;
	for (it_fl = fl.begin(); it_fl != fl.end(); it_fl++) {
		ret.append(std::to_string(it_fl->to_ulong()) + "-");
	}
	//Remove last "-" that is useless
	//ret.pop_back();
	return ret;
}

kworld_id kworld::hash_fluents_into_id()
{
	return hash_fluents_into_id(m_fluent_set);
}

unsigned int kworld::hash_fluents_into_numerical_id(const fluent_set& fl)
{
	std::string ret;
	fluent_set::const_iterator it_fl;
	for (it_fl = fl.begin(); it_fl != fl.end(); it_fl++) {
		ret.append(std::to_string(helper::is_negate(*it_fl)));
	}
	return std::stoi(ret, 0, 2);
}

unsigned int kworld::hash_fluents_into_numerical_id()
{
	return hash_fluents_into_numerical_id(m_fluent_set);
}

void kworld::set_fluent_set(const fluent_set & description)
{
	/*
	 * \throw std::invalid_argument whenever \p description is not consistent.
	 *
	if (consistent(description))*/
	m_fluent_set = description;
	/*else
		throw std::invalid_argument("Non consistent set of fluent");*/
}

bool kworld::consistent(const fluent_set & to_check)
{
	for (auto it_flset = to_check.begin(); it_flset != to_check.end(); it_flset++) {
		if (to_check.find(helper::negate_fluent(*it_flset)) == to_check.end()) {
			return false;
		}

		/* If the pointed fluent is in modulo 2 it means is the positive and if
		 * its successor (the negative version) is in the set then is not consistent.
		if ((*it_flset) % 2 == 0) {
			//The std::set has is elements ordered so we can just check its successor.
			it_flset_tmp = it_flset;
			it_flset_tmp++;
		}
		if (it_flset_tmp != to_check.end()) {
			if (*it_flset_tmp == ((*it_flset) + 1)) {
				std::cout << "\nCheck: " << *it_flset_tmp << " and " << (*it_flset) + 1 << std::endl;
				return false;
			}
		}
		 */
	}
	return true;
}

void kworld::set_id()
{
	m_id = hash_fluents_into_id();
}

void kworld::set_numerical_id()
{
	m_numerical_id = hash_fluents_into_numerical_id();
}

const fluent_set & kworld::get_fluent_set() const
{
	return m_fluent_set;
}

kworld_id kworld::get_id() const
{
	return m_id;
}

int kworld::get_numerical_id() const
{
	return m_numerical_id;
}

bool kworld::entails(fluent to_check) const
{
	return(m_fluent_set.find(to_check) != m_fluent_set.end());
}

/**
 * \todo Check for the size = 0?
 */
bool kworld::entails(const fluent_set & to_check) const
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
bool kworld::entails(const fluent_formula & to_check) const
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

bool kworld::operator<(const kworld& to_compare) const
{
	if (m_id.compare(to_compare.get_id()) < 0)
		return true;
	return false;
}

bool kworld::operator>(const kworld& to_compare) const
{
	if (m_id.compare(to_compare.get_id()) > 0)
		return true;
	return false;
}

bool kworld::operator==(const kworld& to_compare) const
{
	/**std way*/
	if (!((*this) < to_compare) && !(to_compare < (*this))) {
		return true;
	}
	return false;
}

bool kworld::operator=(const kworld & to_assign)
{
	set_fluent_set(to_assign.get_fluent_set());
	set_id();
	return true;
}

void kworld::print() const
{
	std::cout << "\nFluents: " << get_id();
	printer::get_instance().print_list(m_fluent_set);
}

/*-***************************************************************************************************************-*/

kworld_ptr::kworld_ptr()
{
}

kworld_ptr::kworld_ptr(const std::shared_ptr<const kworld> & ptr, unsigned short repetition)
{
	set_ptr(ptr);
	set_repetition(repetition);
}

kworld_ptr::kworld_ptr(std::shared_ptr<const kworld>&& ptr, unsigned short repetition)
{
	set_ptr(ptr);
	set_repetition(repetition);

}

kworld_ptr::kworld_ptr(const kworld & world, unsigned short repetition)
{
	m_ptr = std::make_shared<kworld>(world);
	set_repetition(repetition);

}

void kworld_ptr::set_ptr(const std::shared_ptr<const kworld> & ptr)
{
	m_ptr = ptr;
}

void kworld_ptr::set_ptr(std::shared_ptr<const kworld>&& ptr)
{
	m_ptr = ptr;
}

std::shared_ptr<const kworld> kworld_ptr::get_ptr() const
{
	return m_ptr;
}

const fluent_set & kworld_ptr::get_fluent_set() const
{
	if (m_ptr != nullptr) {
		return get_ptr()->get_fluent_set();
	}
	std::cerr << "Error in creating a kworld_ptr (fluent_set)\n";
	exit(1);
}

kworld_id kworld_ptr::get_fluent_based_id() const
{
	if (m_ptr != nullptr) {
		return(get_ptr()->get_id());
	}
	std::cerr << "\nError in creating a kworld_ptr (id)\n";
	exit(1);
}

kworld_id kworld_ptr::get_id() const
{
	if (m_ptr != nullptr) {
		return(get_ptr()->get_id()).append(std::to_string(get_repetition()));
	}
	std::cerr << "\nError in creating a kworld_ptr (id)\n";
	exit(1);
}

int kworld_ptr::get_numerical_id() const
{
	if (m_ptr != nullptr) {
		return(get_ptr()->get_numerical_id());
	}
	std::cerr << "\nError in creating a kworld_ptr (id)\n";
	exit(1);
}

void kworld_ptr::set_repetition(unsigned short to_set)
{
	m_repetition = to_set;
}

unsigned short kworld_ptr::get_repetition() const
{
	return m_repetition;

}

bool kworld_ptr::entails(fluent to_check) const
{
	return m_ptr->entails(to_check);
}

bool kworld_ptr::entails(const fluent_set& to_check) const
{
	return m_ptr->entails(to_check);
}

bool kworld_ptr::entails(const fluent_formula & to_check) const
{
	return m_ptr->entails(to_check);
}

bool kworld_ptr::operator<(const kworld_ptr & to_compare) const
{
	if (get_id().compare(to_compare.get_id()) < 0) {
		return true;
	}
	return false;
}

bool kworld_ptr::operator>(const kworld_ptr & to_compare) const
{
	if (get_id().compare(to_compare.get_id()) > 0) {
		return true;
	}
	return false;
}

bool kworld_ptr::operator==(const kworld_ptr & to_compare) const
{
	/**std way*/
	if (!((*this) < to_compare) && !(to_compare < (*this))) {
		return true;
	}
	return false;
}

bool kworld_ptr::operator=(const kworld_ptr & to_copy)
{
	set_ptr(to_copy.get_ptr());
	set_repetition(to_copy.get_repetition());
	return true;
}