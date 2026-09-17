#pragma once

/**
 * \brief Helper class for equality and ordering checks in Kripke structures.
 *
 * The helper owns the comparison logic so KripkeState stays small. It can use
 * either the standard comparison or the stronger vectorized one, depending on
 * the selected comparison mode.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date April 2026
 */

#include "utilities/Define.h"

class KripkeEqualityHelper {
public:
  /**
   * \brief Default constructor.
   */
  KripkeEqualityHelper() = default;

  /**
   * \brief Default destructor.
   */
  ~KripkeEqualityHelper() = default;

  /**
   * \brief Copy constructor.
   */
  KripkeEqualityHelper(const KripkeEqualityHelper &) = default;

  /**
   * \brief Copy assignment operator.
   */
  KripkeEqualityHelper &operator=(const KripkeEqualityHelper &) = default;

  /**
   * \brief Move constructor.
   */
  KripkeEqualityHelper(KripkeEqualityHelper &&) = default;

  /**
   * \brief Move assignment operator.
   */
  KripkeEqualityHelper &operator=(KripkeEqualityHelper &&) = default;

    static void set_fast_comparison(bool value) noexcept;


private:
  static bool world_ptr_equal(const KripkeWorldPointer &a,
                              const KripkeWorldPointer &b);

  static bool world_ptr_smaller(const KripkeWorldPointer &a,
                                const KripkeWorldPointer &b);

  static KripkeWorldPointersVec
  canonicalize_worlds(const KripkeWorldPointersSet &worlds);

  static KripkeWorldPointersMapVec
  canonicalize_agent_map(const KripkeWorldPointersMap &beliefs);

  static KripkeWorldPointersTransitiveMapVec
  canonicalize_transitive_map(const KripkeWorldPointersTransitiveMap &beliefs);

  static bool internal_equal(const KripkeWorldPointersVec &lhs,
                             const KripkeWorldPointersVec &rhs);

  static bool internal_smaller(const KripkeWorldPointersVec &lhs,
                               const KripkeWorldPointersVec &rhs);

  static bool internal_equal(const KripkeWorldPointersMapVec &lhs,
                             const KripkeWorldPointersMapVec &rhs);

  static bool internal_smaller(const KripkeWorldPointersMapVec &lhs,
                               const KripkeWorldPointersMapVec &rhs);

  static bool internal_equal(const KripkeWorldPointersTransitiveMapVec &lhs,
                             const KripkeWorldPointersTransitiveMapVec &rhs);

  static bool internal_smaller(const KripkeWorldPointersTransitiveMapVec &lhs,
                               const KripkeWorldPointersTransitiveMapVec &rhs);


    inline static bool s_fast_comparison = false;

  /// \name Equality for KripkeState
  ///@{
  /**
   * \brief Check if a State is equal to another using Worlds ids (no
   * repetition). \param[in] reference The reference state. \param[in]
   * to_compare The state to check against reference. \return True if equal,
   * false otherwise.
   */
  static bool less_operator(const KripkeState &reference,
                                   const KripkeState &to_compare);



  ///@}
  ///

  /*We use this class to alleviate the logic of KripkeState so we make
   * everything private accessible only form KripkeState*/
  friend class KripkeState;
  friend class KripkeReachabilityHelper;
};
