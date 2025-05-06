/**
 * \brief Implementation of \ref pem.h and \ref pem_ptr.h.
 *
 * \copyright GNU Public License.
 *
 *
 * \author Francesco Fabiano.
 * \date May 05, 2021
 */

#include "pem.h"
#include "event.h"
#include "pem_store.h"

pem::pem()
{
}

pem::pem(pem_id id)
{
	set_id(id);
}

pem::pem(const pem & to_copy)
{
	set_id(to_copy.get_id());
	set_edges(to_copy.get_edges());
}

void pem::set_id(pem_id to_set)
{
	m_id = to_set;
}

void pem::set_pointed_id(event_id to_set)
{
	m_pointed_id = to_set;
}

void pem::set_edges(const pem_edges & to_set)
{
	m_edges = to_set;
}

pem_id pem::get_id() const
{
	return m_id;
}

event_id pem::get_pointed_id() const
{
	return m_pointed_id;
}

const pem_edges & pem::get_edges() const
{
	return m_edges;
}

bool pem::operator<(const pem & to_compare) const
{
	if (m_id < to_compare.get_id()) {
		return true;
	}
	return false;
}

bool pem::operator>(const pem & to_compare) const
{
	if (m_id > to_compare.get_id()) {
		return true;
	}
	return false;
}

bool pem::operator==(const pem & to_compare) const
{
	/**std way*/
	if (!(*this < to_compare) && !(to_compare < *this)) {
		return true;
	}
	return false;
}

bool pem::operator=(const pem & to_copy)
{
	set_id(to_copy.get_id());
	set_edges(to_copy.get_edges());
	return true;
}

void pem::print() const
{
	std::cout << "\nModel (" << get_id() << ") has pointed event: " << get_pointed_id() << " and edges: ";

//	for (auto it_fl = m_edges.begin(); it_fl != m_edges.end(); ++it_fl) {
//		for (auto it_edge = it_fl->second.begin(); it_edge != it_fl->second.end(); ++it_edge) {
//			std::cout << " | " << pem_store::get_instance().get_agent_group_name(it_fl->first) << "-" << it_edge->first.get_id() << "-" << it_edge->second.get_id();
//		}
		//std::cout << std::endl;
//	}
}

/****************************************************************************/

pem_ptr::pem_ptr()
{
}

pem_ptr::pem_ptr(const std::shared_ptr<pem> &ptr)
{
	set_ptr(ptr);
}

pem_ptr::pem_ptr(std::shared_ptr<pem> &&ptr)
{
	set_ptr(ptr);
}

pem_ptr::pem_ptr(const pem &action)
{
	m_ptr = std::make_shared<pem>(action);
}

void pem_ptr::set_ptr(const std::shared_ptr<pem> &ptr)
{
	m_ptr = ptr;
}

void pem_ptr::set_ptr(std::shared_ptr<pem> &&ptr)
{
	m_ptr = ptr;
}

std::shared_ptr<pem> pem_ptr::get_ptr() const
{
	return m_ptr;
}

void pem_ptr::set_id(pem_id to_set)
{
	m_ptr->set_id(to_set);
}

void pem_ptr::set_pointed_id(event_id to_set)
{
	m_ptr->set_pointed_id(to_set);
}

void pem_ptr::set_edges(const pem_edges & to_set)
{
	m_ptr->set_edges(to_set);
}

pem_id pem_ptr::get_id() const
{
	if (m_ptr != nullptr) {
		return get_ptr()->get_id();
	}
	std::cerr << "Error in creating a pem_ptr\n";
	exit(1);
}

event_id pem_ptr::get_pointed_id() const
{
	if (m_ptr != nullptr) {
		return get_ptr()->get_pointed_id();
	}
	std::cerr << "Error in creating a pem_ptr\n";
	exit(1);
}

const pem_edges & pem_ptr::get_edges() const
{
	if (m_ptr != nullptr) {
		return get_ptr()->get_edges();
	}
	std::cerr << "Error in creating a pem_ptr\n";
	exit(1);
}

bool pem_ptr::operator<(const pem_ptr & to_compare) const
{
	if (get_id() < to_compare.get_id()) {
		return true;
	}
	return false;
}

bool pem_ptr::operator>(const pem_ptr & to_compare) const
{
	if (get_id() > to_compare.get_id()) {
		return true;
	}
	return false;
}

bool pem_ptr::operator==(const pem_ptr & to_compare) const
{
	/**std way*/
	if (!(*this < to_compare) && !(to_compare < (*this))) {
		return true;
	}
	return false;
}

bool pem_ptr::operator=(const pem_ptr & to_copy)
{
	set_ptr(to_copy.get_ptr());
	return true;
}

void pem_ptr::print()const
{
	m_ptr->print();
}
