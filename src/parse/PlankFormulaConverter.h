#pragma once

#include <vector>

#include "del/language/formulas.h"
#include "formulae/BeliefFormula.h"

/**
 * \class PlankFormulaConverter
 * \brief Converts grounded Plank DEL/EPDDL formulas into DEEP's
 * \ref BeliefFormula representation.
 *
 * \details The converter accepts the full formula fragment currently exposed by
 * Plank and normalizes it to the smaller set of node kinds used internally by
 * DEEP. In particular, implication, diamond, common-diamond, and
 * knowing-whether formulas are rewritten into combinations of NOT, OR, B, E,
 * and C formulas.
 */
class PlankFormulaConverter {
public:
  /**
   * \brief Builds a converter using the grounded fluent and agent order from
   * the current planning task.
   * \param fluents Grounded positive fluents indexed by Plank atom id.
   * \param agents Grounded agents indexed by Plank agent id.
   */
  PlankFormulaConverter(const std::vector<Fluent> &fluents,
                        const std::vector<Agent> &agents);

  /**
   * \brief Converts a grounded Plank formula into DEEP's normalized
   * \ref BeliefFormula representation.
   * \param formula The grounded Plank formula to convert.
   * \return The converted formula tree.
   */
  [[nodiscard]]
  BeliefFormula convert(const plank::del::formula_ptr &formula) const;

private:
  const std::vector<Fluent> &m_fluents;
  const std::vector<Agent> &m_agents;

  /// \brief Convert an atomic fluent formula.
  [[nodiscard]]
  BeliefFormula convert_atom(const plank::del::atom_formula_ptr &formula) const;

  /// \brief Convert a negation formula.
  [[nodiscard]]
  BeliefFormula convert_not(const plank::del::not_formula_ptr &formula) const;

  /// \brief Convert an n-ary conjunction.
  [[nodiscard]]
  BeliefFormula convert_and(const plank::del::and_formula_ptr &formula) const;

  /// \brief Convert an n-ary disjunction.
  [[nodiscard]]
  BeliefFormula convert_or(const plank::del::or_formula_ptr &formula) const;

  /// \brief Convert a box modality into either B or E form.
  [[nodiscard]]
  BeliefFormula convert_box(const plank::del::box_formula_ptr &formula) const;

  /// \brief Convert a common-knowledge box modality into C form.
  [[nodiscard]]
  BeliefFormula
  convert_common(const plank::del::c_box_formula_ptr &formula) const;

  /**
   * \brief Fold a list of formulas into a left-associated propositional tree.
   * \param formulas The operands to combine.
   * \param op The binary propositional operator to use.
   * \return The folded formula tree.
   */
  [[nodiscard]]
  BeliefFormula convert_nary(const plank::del::formula_deque &formulas,
                             BeliefFormulaOperator op) const;

  /**
   * \brief Convert a grounded Plank agent set into DEEP's grounded agent set.
   * \param agents The Plank agent identifiers to convert.
   * \return The corresponding DEEP agent set.
   */
  [[nodiscard]]
  AgentsSet convert_agents(const plank::del::agent_set &agents) const;

  /// \brief Convert implication by rewriting it as !phi OR psi.
  [[nodiscard]]
  BeliefFormula
  convert_imply(const plank::del::imply_formula_ptr &formula) const;

  /// \brief Convert diamond via duality with box.
  [[nodiscard]]
  BeliefFormula
  convert_diamond(const plank::del::diamond_formula_ptr &formula) const;

  /// \brief Convert knowing-whether box into a disjunction of knowledge
  /// formulas.
  [[nodiscard]]
  BeliefFormula
  convert_kw_box(const plank::del::kw_box_formula_ptr &formula) const;

  /// \brief Convert knowing-whether diamond via negated knowing-whether box.
  [[nodiscard]]
  BeliefFormula
  convert_kw_diamond(const plank::del::kw_diamond_formula_ptr &formula) const;

  /// \brief Convert common diamond via duality with common box.
  [[nodiscard]]
  BeliefFormula convert_common_diamond(
      const plank::del::c_diamond_formula_ptr &formula) const;

  /// \brief Build a propositional negation node around an existing formula.
  [[nodiscard]]
  static BeliefFormula make_not(const BeliefFormula &formula);

  /// \brief Build a propositional disjunction node from two formulas.
  [[nodiscard]]
  static BeliefFormula make_or(const BeliefFormula &left,
                               const BeliefFormula &right);
};
