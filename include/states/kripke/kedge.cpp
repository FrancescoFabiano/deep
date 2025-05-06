
/*
 * \brief Class that implements kedge.h.
 *
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date March 31, 2019
 */
#include "kedge.h"

#include "kstore.h"

/*********************************************************************
 kedge implementation
 *********************************************************************/

kedge::kedge()
{
}

kedge::kedge(const kworld & from, const kworld & to, agent label)
{
	set_from(from);
	set_to(to);
	set_label(label);
	set_id();
}

kedge::kedge(const kworld_ptr & from, const kworld_ptr & to, agent label)
{
	set_from(from);
	set_to(to);
	set_label(label);
	set_id();
}

/**
 * \todo the hash function is maybe to simple.
 */
kedge_id kedge::hash_info_into_id(const kworld_ptr & from, const kworld_ptr & to, agent label)
{
	return(from.get_id() + "-" + to.get_id() + "-" + std::to_string(label.to_ulong()));
}

kedge_id kedge::hash_info_into_id()
{
	return(hash_info_into_id(m_from, m_to, m_label));
}

const kworld_ptr & kedge::get_from() const
{
	return m_from;
}

const kworld_ptr & kedge::get_to() const
{
	return m_to;
}

agent kedge::get_label() const
{
	return m_label;
}

kedge_id kedge::get_id() const
{
	return m_edge_id;
}

void kedge::set_from(const kworld & from)
{
	/*Get the reference of the world from the static class \ref kstore
	 *
	 * If the world doesn't exists it create a new one in \ref kstore and get its address
	 * otherwise it get the already existing one's address.*/
	m_from = kstore::get_instance().add_world(from);
}

void kedge::set_to(const kworld & to)
{
	/*Get the reference of the world from the static class \ref kstore
	 *
	 * If the world doesn't exists it create a new one in \ref kstore and get its address
	 * otherwise it get the already existing one's address.*/
	m_to = kstore::get_instance().add_world(to);
}

void kedge::set_from(const kworld_ptr & from)
{
	m_from = from;
}

void kedge::set_to(const kworld_ptr & to)
{
	m_to = to;
}

void kedge::set_label(agent label)
{
	m_label = label;
}

void kedge::set_id()
{
	m_edge_id = hash_info_into_id();
}

bool kedge::operator<(const kedge& to_compare) const
{
	if (m_edge_id.compare(to_compare.get_id()) < 0)
		return true;
	return false;
}

bool kedge::operator>(const kedge& to_compare) const
{
	if (m_edge_id.compare(to_compare.get_id()) > 0)
		return true;
	return false;
}

bool kedge::operator==(const kedge& to_compare) const
{
	/**std way*/
	if (!((*this) < to_compare) && !(to_compare < (*this))) {
		return true;
	}
	return false;
}

bool kedge::operator=(const kedge& to_assign)
{
	set_from(to_assign.get_from());
	set_to(to_assign.get_to());
	set_label(to_assign.get_label());
	//set_id();
	m_edge_id = to_assign.get_id();
	return true;
}

void kedge::print() const
{
	std::cout << "edge id " << get_id() << " label: " << get_label() << " from: " << get_from().get_id() << " to " << get_to().get_id() << std::endl;
}

/*-***************************************************************************************************************-*/

kedge_ptr::kedge_ptr()
{
}

kedge_ptr::kedge_ptr(const std::shared_ptr<const kedge> & ptr)
{
	set_ptr(ptr);
}

kedge_ptr::kedge_ptr(std::shared_ptr<const kedge>&& ptr)
{
	set_ptr(ptr);
}

void kedge_ptr::set_ptr(const std::shared_ptr<const kedge> & ptr)
{
	m_ptr = ptr;
}

void kedge_ptr::set_ptr(std::shared_ptr<const kedge>&& ptr)
{
	m_ptr = ptr;
}

std::shared_ptr<const kedge> kedge_ptr::get_ptr() const
{
	return m_ptr;
}

const kworld_ptr & kedge_ptr::get_from() const
{
	if (m_ptr != nullptr) {
		return get_ptr()->get_from();
	}
	std::cerr << "Error in creating a kedge_ptr\n";
	exit(1);
}

const kworld_ptr & kedge_ptr::get_to() const
{
	if (m_ptr != nullptr) {
		return get_ptr()->get_to();
	}
	std::cerr << "Error in creating a kedge_ptr\n";
	exit(1);
}

agent kedge_ptr::get_label() const
{
	if (m_ptr != nullptr) {
		return get_ptr()->get_label();
	}
	std::cerr << "Error in creating a kedge_ptr\n";
	exit(1);
}

kedge_id kedge_ptr::get_id() const
{
	if (m_ptr != nullptr) {
		return get_ptr()->get_id();
	}
	std::cerr << "Error in creating a kedge_ptr\n";
	exit(1);
}

bool kedge_ptr::operator=(const kedge_ptr & to_copy)
{
	set_ptr(to_copy.get_ptr());
	return true;
}

bool kedge_ptr::operator<(const kedge_ptr & to_compare) const
{
	if (get_id().compare(to_compare.get_id()) < 0) {
		return true;
	}
	return false;
}

bool kedge_ptr::operator>(const kedge_ptr & to_compare) const
{
	if (get_id().compare(to_compare.get_id()) > 0) {
		return true;
	}
	return false;
}

bool kedge_ptr::operator==(const kedge_ptr & to_compare) const
{
	/**std way*/
	if (!((*this) < to_compare) && !(to_compare < (*this))) {
		return true;
	}
	return false;
}

/** \brief Function used to print all the information of \ref m_ptr.*/
void kedge_ptr::print() const
{
	m_ptr->print();
}