/*
 * \file KripkeStorage.cpp
 * \brief Implementation of KripkeStorage.
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 17, 2025
 */

#include "KripkeStorage.h"
#include "utilities/ExitHandler.h"
#include <memory>

KripkeStorage &KripkeStorage::get_instance() noexcept {
  static KripkeStorage instance;
  return instance;
}

KripkeWorldPointer KripkeStorage::add_world(const KripkeWorld &to_add) {
  // Try to insert the world. If it already exists, get the existing one.
  auto [it, inserted] = m_created_worlds.insert(to_add);
  if (it == m_created_worlds.end()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::KripkeStorageInsertError,
        "Error: Failed to insert or find KripkeWorld in KripkeStorage.\n"
        "This should never happen. Please check the integrity of KripkeWorld "
        "comparison and hashing.");
  }
  // Return a shared pointer to the (possibly existing) KripkeWorld.
  return KripkeWorldPointer(std::make_shared<const KripkeWorld>(*it));
}
