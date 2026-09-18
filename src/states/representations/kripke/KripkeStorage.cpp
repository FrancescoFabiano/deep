/*
 * \file KripkeStorage.cpp
 * \brief Implementation of KripkeStorage.
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 17, 2025
 */

#include "KripkeStorage.h"

#include <memory>
#include <utility>
#ifdef DEBUG
#include <cassert>
#endif

KripkeStorage &
KripkeStorage::get_instance() noexcept {

  static KripkeStorage instance;

  return instance;
}


KripkeWorldPointer
KripkeStorage::add_world(
    const KripkeWorld &to_add) {

  /*
   * Heterogeneous lookup:
   *
   * Search using the KripkeWorld directly even though the unordered_set
   * stores shared_ptr<const KripkeWorld>.
   *
   * This avoids allocating a temporary shared KripkeWorld merely to
   * discover that the valuation has already been canonicalized.
   */
  const auto existing =
      m_created_worlds.find(
          to_add);

    if (existing !=
        m_created_worlds.end()) {

        const KripkeWorldPointer result(
            *existing);

#ifdef DEBUG
        assert(
            result.get_ptr().get() ==
            existing->get());
#endif

        return result;

  }

  /*
   * This valuation is genuinely new.
   *
   * Allocate exactly one immutable canonical KripkeWorld.
   */
  auto stored_world =
      std::make_shared<const KripkeWorld>(
          to_add);

  const auto [it, inserted] =
      m_created_worlds.insert(
          std::move(stored_world));

  /*
   * Under the current single-threaded storage usage, insertion should
   * succeed after the lookup above.
   *
   * Returning *it remains correct regardless.
   */
  (void)inserted;

  return KripkeWorldPointer(
      *it);
}


KripkeWorldPointer
KripkeStorage::add_world(
    KripkeWorld &&to_add) {

  /*
   * Lookup must happen before moving from to_add.
   */
  const auto existing =
      m_created_worlds.find(
          to_add);

    if (existing !=
        m_created_worlds.end()) {

        const KripkeWorldPointer result(
            *existing);

#ifdef DEBUG
        assert(
            result.get_ptr().get() ==
            existing->get());
#endif

        return result;

  }

  /*
   * New valuation.
   *
   * Move the complete KripkeWorld, including its fluent set, directly
   * into the canonical shared allocation.
   */
  auto stored_world =
      std::make_shared<const KripkeWorld>(
          std::move(to_add));

  const auto [it, inserted] =
      m_created_worlds.insert(
          std::move(stored_world));

  (void)inserted;

  return KripkeWorldPointer(
      *it);
}