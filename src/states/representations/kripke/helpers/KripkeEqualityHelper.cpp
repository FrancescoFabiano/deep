#include "KripkeEqualityHelper.h"
#include "Define.h"

#include "KripkeState.h"
#include "KripkeWorld.h"

bool KripkeEqualityHelper::world_ptr_equal(const KripkeWorldPointer &a,
                                           const KripkeWorldPointer &b) {
  return a.internal_equal(b);
}

bool KripkeEqualityHelper::world_ptr_smaller(const KripkeWorldPointer &a,
                                             const KripkeWorldPointer &b) {
  if (a.internal_equal(b)) {
    return false;
  }
  return a.internal_smaller(b);
}

KripkeWorldPointersVec KripkeEqualityHelper::canonicalize_worlds(
    const KripkeWorldPointersSet &worlds) {
  KripkeWorldPointersVec result;
  result.reserve(worlds.size());

  for (const auto &w : worlds) {
    result.push_back(w);
  }

  std::ranges::sort(result,
                    [](const KripkeWorldPointer &a, const KripkeWorldPointer &b) {
                      const auto ida = a.get_internal_world_id();
                      const auto idb = b.get_internal_world_id();

                      if (ida != idb) {
                        return ida < idb;
                      }

                      if (a.internal_equal(b)) {
                        return false;
                      }

                      return a.internal_smaller(b);
                    });

  return result;
}

KripkeWorldPointersMapVec KripkeEqualityHelper::canonicalize_agent_map(
    const KripkeWorldPointersMap &beliefs) {
  KripkeWorldPointersMapVec result;
  result.reserve(beliefs.size());

  for (const auto &[agent, worlds] : beliefs) {
    result.emplace_back(agent, canonicalize_worlds(worlds));
  }

  return result;
}

KripkeWorldPointersTransitiveMapVec
KripkeEqualityHelper::canonicalize_transitive_map(
    const KripkeWorldPointersTransitiveMap &beliefs) {
  KripkeWorldPointersTransitiveMapVec result;
  result.reserve(beliefs.size());

  for (const auto &[world, agent_map] : beliefs) {
    result.emplace_back(world, canonicalize_agent_map(agent_map));
  }

  std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
    const auto ida = a.first.get_internal_world_id();
    const auto idb = b.first.get_internal_world_id();

    if (ida != idb) {
      return ida < idb;
    }

    if (a.first.internal_equal(b.first)) {
      return false;
    }

    return a.first.internal_smaller(b.first);
  });

  return result;
}

bool KripkeEqualityHelper::internal_equal(const KripkeWorldPointersVec &lhs,
                                          const KripkeWorldPointersVec &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (!world_ptr_equal(lhs[i], rhs[i])) {
      return false;
    }
  }

  return true;
}

bool KripkeEqualityHelper::internal_smaller(const KripkeWorldPointersVec &lhs,
                                            const KripkeWorldPointersVec &rhs) {
  return std::ranges::lexicographical_compare(
      lhs, rhs, [](const KripkeWorldPointer &a, const KripkeWorldPointer &b) {
        return world_ptr_smaller(a, b);
      });
}

bool KripkeEqualityHelper::internal_equal(
    const KripkeWorldPointersMapVec &lhs,
    const KripkeWorldPointersMapVec &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].first != rhs[i].first) {
      return false;
    }

    if (!internal_equal(lhs[i].second, rhs[i].second)) {
      return false;
    }
  }

  return true;
}

bool KripkeEqualityHelper::internal_smaller(
    const KripkeWorldPointersMapVec &lhs,
    const KripkeWorldPointersMapVec &rhs) {
  return std::ranges::lexicographical_compare(
      lhs, rhs, [](const auto &a, const auto &b) {
        if (a.first != b.first) {
          return a.first < b.first;
        }

        if (internal_equal(a.second, b.second)) {
          return false;
        }

        return internal_smaller(a.second, b.second);
      });
}

bool KripkeEqualityHelper::internal_equal(
    const KripkeWorldPointersTransitiveMapVec &lhs,
    const KripkeWorldPointersTransitiveMapVec &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (!world_ptr_equal(lhs[i].first, rhs[i].first)) {
      return false;
    }

    if (!internal_equal(lhs[i].second, rhs[i].second)) {
      return false;
    }
  }

  return true;
}

bool KripkeEqualityHelper::internal_smaller(
    const KripkeWorldPointersTransitiveMapVec &lhs,
    const KripkeWorldPointersTransitiveMapVec &rhs) {
  return std::ranges::lexicographical_compare(
      lhs, rhs, [](const auto &a, const auto &b) {
        if (!world_ptr_equal(a.first, b.first)) {
          return world_ptr_smaller(a.first, b.first);
        }

        if (internal_equal(a.second, b.second)) {
          return false;
        }

        return internal_smaller(a.second, b.second);
      });
}

bool KripkeEqualityHelper::less_operator(
    const KripkeState &reference,
    const KripkeState &to_compare) {

  const auto reference_hash =
      reference.get_hash();

  const auto to_compare_hash =
      to_compare.get_hash();

  if (reference_hash != to_compare_hash)
    return reference_hash < to_compare_hash;

  /*
   * Fast mode assumes that equal state hashes identify
   * equivalent Kripke states.
   */
  if (s_fast_comparison)
    return false;

  // Hash collision / same hash:
  // perform the actual strong structural ordering.

  const auto &reference_designated =
      reference.get_designated_worlds_vec();

  const auto &to_compare_designated =
      to_compare.get_designated_worlds_vec();

  if (!internal_equal(
          reference_designated,
          to_compare_designated)) {
    return internal_smaller(
        reference_designated,
        to_compare_designated);
          }

  const auto &reference_worlds =
      reference.get_worlds_vec();

  const auto &to_compare_worlds =
      to_compare.get_worlds_vec();

  if (!internal_equal(
          reference_worlds,
          to_compare_worlds)) {
    return internal_smaller(
        reference_worlds,
        to_compare_worlds);
          }

  return internal_smaller(
      reference.get_beliefs_vec(),
      to_compare.get_beliefs_vec());
}

void KripkeEqualityHelper::set_fast_comparison(
    const bool value) noexcept {
  s_fast_comparison = value;
}


