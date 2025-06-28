#pragma once

/**
 * \brief Helper class for entailment checks in Kripke structures.
 *
 * Provides static methods to provide reachability on the estates.
 * We use this class to alleviate the logic of KripkeState so we make everything
 * private accessible only form KripkeState \author Francesco Fabiano \date May
 * 17, 2025
 */

#include "utilities/Define.h"

class KripkeReachabilityHelper {
public:
  KripkeReachabilityHelper() = default;

private:
  /**
   * \brief Get all worlds reachable by an agent from a starting world, using a
   * KripkeState. \param[in] ag The agent. \param[in] world The starting world.
   * \param[in] kstate The KripkeState to use for reachability.
   * \return Set of reachable worlds.
   */
  [[nodiscard]] static KripkeWorldPointersSet
  get_B_reachable_worlds(const Agent &ag, const KripkeWorldPointer &world,
                         const KripkeState &kstate);

  /**
   * \brief Recursive helper for get_B_reachable_worlds using a KripkeState.
   * \param[in] ag The agent.
   * \param[in] world The starting world.
   * \param[out] reached The set of reached worlds.
   * \param[in] kstate The KripkeState to use for reachability.
   * \return True if fixed point reached, false otherwise.
   */
  [[nodiscard]] static bool get_B_reachable_worlds_recursive(
      const Agent &ag, const KripkeWorldPointer &world,
      KripkeWorldPointersSet &reached, const KripkeState &kstate);

  /**
   * \brief Get all worlds reachable by a set of agents from a starting world,
   * using a KripkeState. \param[in] ags The set of agents. \param[in] world The
   * starting world. \param[in] kstate The KripkeState to use for reachability.
   * \return Set of reachable worlds.
   */
  [[nodiscard]] static KripkeWorldPointersSet
  get_E_reachable_worlds(const AgentsSet &ags, const KripkeWorldPointer &world,
                         const KripkeState &kstate);

  /**
   * \brief Recursive helper for get_E_reachable_worlds using a KripkeState.
   * \param[in] ags The set of agents.
   * \param[in] worlds The set of starting worlds.
   * \param[out] reached The set of reached worlds.
   * \param[in] kstate The KripkeState to use for reachability.
   * \return True if fixed point reached, false otherwise.
   */
  [[nodiscard]] static bool get_E_reachable_worlds_recursive(
      const AgentsSet &ags, const KripkeWorldPointersSet &worlds,
      KripkeWorldPointersSet &reached, const KripkeState &kstate);

  /**
   * \brief Get all common knowledge reachable worlds by a set of agents from a
   * starting world, using a KripkeState. \param[in] ags The set of agents.
   * \param[in] world The starting world.
   * \param[in] kstate The KripkeState to use for reachability.
   * \return Set of C-reachable worlds.
   */
  [[nodiscard]] static KripkeWorldPointersSet
  get_C_reachable_worlds(const AgentsSet &ags, const KripkeWorldPointer &world,
                         const KripkeState &kstate);

  /**
   * \brief Get all common knowledge reachable worlds by a set of agents from a
   * starting world, using a KripkeState. \param[in] world The starting world.
   * \param[out] reached_worlds the set of reachable worlds.
   * \param[out] reached_edges the set of reachable edges.
   * \param[out] kstate The KripkeState to use for reachability.
   */
  static void
  get_all_reachable_worlds(const KripkeWorldPointer &world,
                           KripkeWorldPointersSet &reached_worlds,
                           KripkeWorldPointersTransitiveMap &reached_edges,
                           const KripkeState &kstate);

  /**
   * @brief Removes all unreachable possible worlds from the given Kripke state.
   *
   * This function computes all worlds that are reachable from the pointed world
   * in the given KripkeState. It then updates the state to retain only the
   * reachable worlds and the belief edges among them.
   *
   * @param kstate The KripkeState object to be pruned of unreachable possible
   * worlds.
   */
  static void clean_unreachable_worlds(KripkeState &kstate);

  /*We use this class to alleviate the logic of KripkeState so we make
   * everything private accessible only form KripkeState*/
  friend class KripkeState;
  friend class KripkeEntailmentHelper;
};
