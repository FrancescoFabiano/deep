/**
 * \brief Implementation of \ref event.h and \ref event_ptr.h.
 *
 * \copyright GNU Public License.
 * 
 * @todo: Redefine the insert so that we have the bulk check (Bisimulation with all the graphs). 
 *
 * \author Francesco Fabiano.
 * \date April 28, 2021
 */

#include <iostream>
#include <stdexcept>

#include "event.h"
#include "../utilities/printer.h"

event::event()
{
	m_id = -1;
}

/*event::event(event_id id, bool ontic_change)
{
	set_id(id);
	set_ontic_change(ontic_change);
}

event::event(event_id id, bool ontic_change, const formula_list & pre)
{

	set_id(id);
	set_ontic_change(ontic_change);
	set_precondition(pre);
}

event::event(event_id id, bool ontic_change, const formula_list & pre, const event_postconditions & post)
{
	set_id(id);
	set_ontic_change(ontic_change);
	set_precondition(pre);
	set_postconditions(post);
}*/


event::event(event_id id, const formula_list & pre, const event_metacond & meta_pre, const event_postconditions & post, const event_metacond & meta_post)
{
	set_id(id);
	set_precondition(pre);
	set_meta_precondition(meta_pre);
	set_postconditions(post);
	set_meta_postconditions(meta_post);
	set_ontic_change();
}

event::event(event_id id, const event_metacond & meta_pre, const event_metacond & meta_post)
{
	set_id(id);
	set_meta_precondition(meta_pre);
	set_meta_postconditions(meta_post);
	set_ontic_change();
}

event::event(const event & to_copy)
{
	set_id(to_copy.get_id());
	set_precondition(to_copy.get_precondition());
	set_meta_precondition(to_copy.get_meta_precondition());
	set_postconditions(to_copy.get_postconditions());
	set_meta_postconditions(to_copy.get_meta_postconditions());
	set_ontic_change(to_copy.get_ontic_change());
}

void event::set_id(const event_id to_set)
{
	m_id = to_set;
}

void event::set_precondition(const formula_list & to_set)
{
	m_pre = to_set;
}

void event::set_meta_precondition(const event_metacond & to_set)
{
	m_meta_pre = to_set;
}

void event::set_postconditions(const event_postconditions & to_set)
{
	if (to_set.size() > 1) {
		std::cerr << "\n\n***Non determinism in event postconditions is not yet supported.***\n\n";
		exit(1);
	}
	m_post = to_set;
}

void event::set_meta_postconditions(const event_metacond & to_set)
{
	m_meta_post = to_set;

}

void event::set_ontic_change()
{

	if (m_meta_post.size() == 0) {
		set_ontic_change(false);
	} else if (m_meta_post.size() > 1) {
		set_ontic_change(true);
	} else if (*m_meta_post.begin() == none) {
		set_ontic_change(false);
	} else {
		set_ontic_change(true);
	}

}

void event::set_ontic_change(bool to_set)
{

	m_ontic_change = to_set;

}

const event_id event::get_id() const
{
	return m_id;
}

const formula_list & event::get_precondition() const
{
	return m_pre;
}

const event_metacond & event::get_meta_precondition() const
{
	return m_meta_pre;
}

const event_postconditions & event::get_postconditions() const
{
	return m_post;
}

const event_metacond & event::get_meta_postconditions() const
{
	return m_meta_post;
}

const bool event::get_ontic_change() const
{
	return m_ontic_change;
}

bool event::operator<(const event & to_compare) const
{
	if (m_id < to_compare.get_id()) { // We compare the ids since we assume that each PEM has at most one event for each id
		return true;
	}
	return false;
}

bool event::operator>(const event & to_compare) const
{
	if (m_id > to_compare.get_id()) {
		return true;
	}
	return false;
}

bool event::operator==(const event & to_compare) const
{
	/**std way*/
	if (!(*this < to_compare) && !(to_compare < *this)) {
		return true;
	}
	return false;
}

bool event::operator=(const event & to_copy)
{
	set_id(to_copy.get_id());
	set_precondition(to_copy.get_precondition());
	set_postconditions(to_copy.get_postconditions());
	return true;
}

void event::print()const
{
	std::cout << "\nEvent (" << get_id() << ") has meta_preconditions: ";


	for (auto it_fl = m_meta_pre.begin(); it_fl != m_meta_pre.end(); ++it_fl) {
		std::cout << " | " << *it_fl;
		//std::cout << std::endl;
	}

	std::cout << " and has meta_postconditions: ";
	for (auto it_fl = m_meta_post.begin(); it_fl != m_meta_post.end(); ++it_fl) {
		std::cout << " | " << *it_fl;
		//std::cout << std::endl;
	}

	//	for (auto it_fl = m_pre.begin(); it_fl != m_pre.end(); ++it_fl) {
	//		std::cout << " | ";
	//		it_fl->print();
	//		//std::cout << std::endl;
	//	}
	//	std::cout << " and has postconditions: ";
	//	for (auto it_effmap = m_post.begin(); it_effmap != m_post.end(); ++it_effmap) {
	//		std::cout << " | ";
	//		printer::get_instance().print_list(it_effmap->first);
	//		std::cout << " if ";
	//		it_effmap->second.print();
	//	}
	std::cout << std::endl;
}

/****************************************************************************/

event_ptr::event_ptr()
{
}

event_ptr::event_ptr(const std::shared_ptr <event> &ptr)
{
	set_ptr(ptr);
}

event_ptr::event_ptr(std::shared_ptr <event> &&ptr)
{
	set_ptr(ptr);
}

event_ptr::event_ptr(const event &action)
{
	m_ptr = std::make_shared<event>(action);
}

void event_ptr::set_ptr(const std::shared_ptr <event> &ptr)
{
	m_ptr = ptr;
}

void event_ptr::set_ptr(std::shared_ptr <event> &&ptr)
{
	m_ptr = ptr;
}

std::shared_ptr <event> event_ptr::get_ptr() const
{
	return m_ptr;
}

const event_id event_ptr::get_id() const
{
	if (m_ptr != nullptr) {
		return get_ptr()->get_id();
	}
	std::cerr << "Error in creating a event_ptr\n";
	exit(1);
}

const formula_list & event_ptr::get_precondition() const
{
	if (m_ptr != nullptr) {
		return get_ptr()->get_precondition();
	}
	std::cerr << "Error in creating a event_ptr\n";
	exit(1);
}

const event_postconditions & event_ptr::get_postconditions() const
{
	if (m_ptr != nullptr) {
		return get_ptr()->get_postconditions();
	}
	std::cerr << "Error in creating a event_ptr\n";
	exit(1);
}

const event_metacond & event_ptr::get_meta_precondition() const
{
	if (m_ptr != nullptr) {
		return get_ptr()->get_meta_precondition();
	}
	std::cerr << "Error in creating a event_ptr\n";
	exit(1);
}

const event_metacond & event_ptr::get_meta_postconditions() const
{
	if (m_ptr != nullptr) {
		return get_ptr()->get_meta_postconditions();
	}
	std::cerr << "Error in creating a event_ptr\n";
	exit(1);
}

bool event_ptr::get_ontic_change() const
{
	if (m_ptr != nullptr) {
		return get_ptr()->get_ontic_change();
	}
	std::cerr << "Error in creating a event_ptr\n";
	exit(1);
}

bool event_ptr::operator<(const event_ptr & to_compare) const
{
	if (get_id() < to_compare.get_id()) {
		return true;
	}
	return false;
}

bool event_ptr::operator>(const event_ptr & to_compare) const
{
	if (get_id() > to_compare.get_id()) {
		return true;
	}
	return false;
}

bool event_ptr::operator==(const event_ptr & to_compare) const
{
	/**std way*/
	if (!(*this < to_compare) && !(to_compare < (*this))) {
		return true;
	}
	return false;
}

bool event_ptr::operator=(const event_ptr & to_copy)
{
	set_ptr(to_copy.get_ptr());
	return true;
}

void event_ptr::print() const
{
	get_ptr()->print();
}
