#pragma once

#include "utilities/Define.h"

/**
 * \class InitialStateInformation
 * \brief Stores the initial state information, including pointed world and
 * belief conditions.
 *
 * \see domain
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 16, 2025
 */
class InitialStateInformation {
public:
  /// \brief Default constructor. Initializes all members with their default
  /// constructors.
  InitialStateInformation() = default;

  /**
   * \brief Adds a condition for the initial pointed world.
   * \param[in] to_add Fluent formula that the initial pointed world must
   * entail.
   */
  void add_pointed_condition(const FluentFormula &to_add);

  /**
   * \brief Adds a belief condition for the initial state.
   * \param[in] to_add Belief formula that the initial state must entail.
   */
  void add_initial_condition(const BeliefFormula &to_add);

  /**
   * \brief Returns the pointed world conditions.
   * \return The fluent formula for the pointed world.
   */
  [[nodiscard]] const FluentFormula &get_pointed_world_conditions() const;

  /**
   * \brief Returns the initial belief conditions.
   * \return The list of belief formulas for the initial state.
   */
  [[nodiscard]] const FormulaeList &get_initial_conditions() const;

  /**
   * \brief Returns the set of initially known fluents.
   * \return The set of fluents known initially.
   */
  [[nodiscard]] const FluentsSet &get_initially_known_fluents() const;

  /**
   * \brief Sets the S5 fluent formula for the initial state.
   *
   * Used to check possible worlds for the initial State under the finitary-S5
   * restriction.
   */
  void set_ff_forS5();

  /**
   * \brief Returns the S5 fluent formula for the initial state.
   * \return The fluent formula that all initial worlds must entail if S5 is
   * required.
   */
  [[nodiscard]] const FluentFormula &get_ff_forS5() const;

private:
  /// Formula containing all the fluents required by the possible initial
  /// pointed world.
  FluentFormula m_pointed_world_conditions;

  /// List of belief formulas describing the initial beliefs conditions.
  FormulaeList m_bf_initial_conditions;

  /// Fluent formula representation of m_bf_initial_conditions for finitary-S5.
  FluentFormula m_ff_forS5;

  /// Set of fluents known by formulas of the type C([ags], f).
  FluentsSet m_initially_known_fluents;

  /**
   * \brief Checks if a belief formula respects the initial restriction.
   * \param[in] to_check The belief formula to check.
   * \return true if the formula respects the restriction, false otherwise.
   */
  [[nodiscard]] static bool check_restriction(const BeliefFormula &to_check);
};
