/*
 * \file KripkeWorld.cpp
 * \brief Implementation of KripkeWorld and KripkeWorldPointer.
 * \copyright GNU Public License.
 * \author Francesco Fabiano.
 * \date September 14, 2019
 */
#include <boost/dynamic_bitset.hpp>
#include "KripkeWorld.h"

#include <iostream>

#include "FormulaHelper.h"
#include "HelperPrint.h"
#include "utilities/ExitHandler.h"

KripkeWorld::KripkeWorld(const FluentsSet& description) {
    set_fluent_set(description);
    set_id();
}

KripkeWorld::KripkeWorld(const KripkeWorld& world) {
    set_fluent_set(world.get_fluent_set());
    set_id();
}

KripkeWorldId KripkeWorld::hash_fluents_into_id() const {
    return FormulaHelper::hash_fluents_into_id(m_fluent_set);
}

void KripkeWorld::set_fluent_set(const FluentsSet& description) {
    if (!FormulaHelper::consistent(description)) {
        std::cerr << "  Fluent set: ";
        HelperPrint::get_instance().print_list(description,std::cerr);
        std::cerr << std::endl;
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::DomainInitialStateRestrictionError,
            "Error: Attempted to set an inconsistent set of fluents in KripkeWorld (printed above).\n"
        );
    }
    m_fluent_set = description;
}

void KripkeWorld::set_id() {
    m_id = hash_fluents_into_id();
}

const FluentsSet& KripkeWorld::get_fluent_set() const noexcept {
    return m_fluent_set;
}

KripkeWorldId KripkeWorld::get_id() const noexcept {
    return m_id;
}

bool KripkeWorld::operator<(const KripkeWorld& to_compare) const noexcept {
    return m_id < to_compare.get_id();
}

bool KripkeWorld::operator>(const KripkeWorld& to_compare) const noexcept {
    return m_id > to_compare.get_id();
}

bool KripkeWorld::operator==(const KripkeWorld& to_compare) const noexcept {
    /**std way*/
    if (!((*this) < to_compare) && !(to_compare < (*this))) {
        return true;
    }
    return false;
}

KripkeWorld& KripkeWorld::operator=(const KripkeWorld& to_assign) {
    if (this != &to_assign) {
        set_fluent_set(to_assign.get_fluent_set());
        set_id();
    }
    return *this;
}

void KripkeWorld::print(std::ostream& os) const {
    os << "\nFluents: " << get_id();
    HelperPrint::get_instance().print_list(m_fluent_set,os);
}

// *************************************************************************************************************** //

KripkeWorldPointer::KripkeWorldPointer(const std::shared_ptr<const KripkeWorld>& ptr, unsigned short repetition) {
    set_ptr(ptr);
    set_repetition(repetition);
}

KripkeWorldPointer::KripkeWorldPointer(std::shared_ptr<const KripkeWorld>&& ptr, unsigned short repetition) {
    set_ptr(std::move(ptr));
    set_repetition(repetition);
}

KripkeWorldPointer::KripkeWorldPointer(const KripkeWorld& world, unsigned short repetition) {
    m_ptr = std::make_shared<KripkeWorld>(world);
    set_repetition(repetition);
}

KripkeWorldPointer& KripkeWorldPointer::operator=(const KripkeWorldPointer& to_copy) {
    if (this != &to_copy) {
        set_ptr(to_copy.get_ptr());
        set_repetition(to_copy.get_repetition());
    }
    return *this;
}

std::shared_ptr<const KripkeWorld> KripkeWorldPointer::get_ptr() const noexcept {
    return m_ptr;
}

void KripkeWorldPointer::set_ptr(const std::shared_ptr<const KripkeWorld>& ptr) {
    m_ptr = ptr;
}

void KripkeWorldPointer::set_ptr(std::shared_ptr<const KripkeWorld>&& ptr) {
    m_ptr = std::move(ptr);
}

void KripkeWorldPointer::set_repetition(const unsigned short to_set) noexcept {
    m_repetition = to_set;
}

void KripkeWorldPointer::increase_repetition(const unsigned short to_increase) noexcept {
    m_repetition += to_increase;
}

unsigned short KripkeWorldPointer::get_repetition() const noexcept {
    return m_repetition;
}

const FluentsSet& KripkeWorldPointer::get_fluent_set() const {
    if (m_ptr) {
        return m_ptr->get_fluent_set();
    }
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::KripkeWorldPointerNullError,
        "Error: Null KripkeWorldPointer in get_fluent_set().\n  Tip: Ensure all KripkeWorldPointer objects are properly initialized before use."
    );
    static FluentsSet dummy;
    return dummy;
}

KripkeWorldId KripkeWorldPointer::get_fluent_based_id() const noexcept {
    if (m_ptr) {
        return m_ptr->get_id();
    }
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::KripkeWorldPointerNullError,
        "Error: Null KripkeWorldPointer in get_fluent_based_id().\n  Tip: Ensure all KripkeWorldPointer objects are properly initialized before use."
    );
    return 0;
}

KripkeWorldId KripkeWorldPointer::get_id() const noexcept {
    if (m_ptr) {
        const KripkeWorldId id = m_ptr->get_id();
        return boost::hash_value((1000 * id) + get_repetition());
    }
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::KripkeWorldPointerNullError,
        "Error: Null KripkeWorldPointer in get_id().\n  Tip: Ensure all KripkeWorldPointer objects are properly initialized before use."
    );
    return 0;
}

KripkeWorldId KripkeWorldPointer::get_internal_world_id() const noexcept {
    if (m_ptr) {
        /*OLD CODE
        const KripkeWorldId id = m_ptr->get_id();
        boost::hash_value(id);*/
        return m_ptr->get_id();
    }
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::KripkeWorldPointerNullError,
        "Error: Null KripkeWorldPointer in get_internal_world_id().\n  Tip: Ensure all KripkeWorldPointer objects are properly initialized before use."
    );
    return 0;
}


bool KripkeWorldPointer::operator<(const KripkeWorldPointer& to_compare) const noexcept {
    return get_id() < to_compare.get_id();
}

bool KripkeWorldPointer::operator>(const KripkeWorldPointer& to_compare) const noexcept {
    return get_id() > to_compare.get_id();
}

bool KripkeWorldPointer::operator==(const KripkeWorldPointer& to_compare) const noexcept {
    return get_id() == to_compare.get_id();
}
