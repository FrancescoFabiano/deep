/**
 * \class BeliefFormula
 * \brief Internal representation of grounded EPDDL goal, precondition, and
 * observability formulae.
 *
 * \details A \ref BeliefFormula stores the normalized formula tree used by
 * DEEP after Plank parses and grounds the source EPDDL. The representation can
 * encode:
 *    - \ref TRUE_FORMULA and \ref FALSE_FORMULA;
 *    - \ref FLUENT_FORMULA -- grounded propositional formulae over fluents;
 *    - \ref BELIEF_FORMULA -- B(a, *phi*);
 *    - \ref E_FORMULA -- E(G, *phi*);
 *    - \ref C_FORMULA -- C(G, *phi*);
 *    - \ref PROPOSITIONAL_FORMULA -- formulas built with \ref BF_NOT,
 *      \ref BF_AND, and \ref BF_OR.
 *
 * EPDDL surface syntax that is supported by the parser but not stored as a
 * dedicated node kind is lowered during conversion:
 *    - implication is rewritten as !phi OR psi;
 *    - diamond modalities are rewritten via duality with box modalities;
 *    - knowing-whether modalities are rewritten into disjunctions of
 *      knowledge formulas.
 *
 * This class therefore describes DEEP's semantic formula core rather than a
 * one-to-one AST of the original EPDDL text.
 *
 * \todo With new parser maybe implement the "move" so the reader actually moves
 * the object instead of copying them.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 16, 2025
 */
#pragma once

#include <memory>

#include "utilities/Define.h"

class BeliefFormula {
public:
  /// \name Constructors
  ///@{
  /** \brief Empty Constructor */
  BeliefFormula() = default;


  /** \brief Copy Constructor
   *  \param[in] to_copy The \ref BeliefFormula to copy in *this*.
   */
  BeliefFormula(const BeliefFormula &to_copy);

  /** \brief Move Constructor */
  BeliefFormula(BeliefFormula &&) noexcept = default;

  /** \brief Destructor */
  ~BeliefFormula() = default;

  /** \brief Copy Assignment */
  BeliefFormula &operator=(const BeliefFormula &to_copy);

  /** \brief Move Assignment */
  BeliefFormula &operator=(BeliefFormula &&) noexcept = default;
  ///@}

  /// \name Setters
  ///@{
  /** \brief Setter for the field m_fluent_formula.
   *  \param[in] to_set The FluentFormula object to copy in
   * m_fluent_formula.
   */
  void set_fluent_formula(const FluentFormula &to_set);

  /** \brief Setter for the field m_fluent_formula from a fluent.
   *  \param[in] to_set The Fluent object to insert in m_fluent_formula.
   */
  void set_fluent_formula_from_fluent(const Fluent &to_set);

  /** \brief Setter for the field m_agent.
   *  \param[in] to_set The Agent object to copy in m_agent.
   */
  void set_agent(const Agent &to_set);

  /** \brief Setter for the field m_group_agents.
   *  \param[in] to_set The AgentsSet object to copy in m_group_agents.
   */
  void set_group_agents(const AgentsSet &to_set);

  /** \brief Setter of the field m_bf1.
   *  This setter takes a \ref BeliefFormula and sets m_bf1 to be its pointer.
   *  \param[in] to_set The \ref BeliefFormula to be pointed by m_bf1.
   */
  void set_bf1(const BeliefFormula &to_set);

  /** \brief Setter of the field m_bf2.
   *  This setter takes a \ref BeliefFormula and sets m_bf2 to be its pointer.
   *  \param[in] to_set The \ref BeliefFormula to be pointed by m_bf2.
   */
  void set_bf2(const BeliefFormula &to_set);

  /** \brief Setter for the field m_formula_type.
   *  \param[in] to_set The \ref BeliefFormulaType object to copy in
   * m_formula_type.
   */
  void set_formula_type(BeliefFormulaType to_set);

  /** \brief Setter for the field m_operator.
   *  \param[in] to_set The \ref BeliefFormulaOperator object to copy in
   * m_operator.
   */
  void set_operator(BeliefFormulaOperator to_set);

  /** \brief Setter from a fluent_formula. */
  void set_from_ff(const FluentFormula &to_build);
  ///@}

  /// \name Getters
  ///@{
  /** \brief Getter for the field m_formula_type.
   *  \return The value of the field m_formula_type.
   */
  [[nodiscard]] BeliefFormulaType get_formula_type() const noexcept;

  /** \brief Getter for the field m_fluent_formula.
   *  \return The value of the field m_fluent_formula.
   */
  [[nodiscard]] const FluentFormula &get_fluent_formula() const noexcept;

  /** \brief Getter for the field m_agent.
   *  \return The value of the field m_agent.
   */
  [[nodiscard]] const Agent &get_agent() const noexcept;

  /** \brief Getter of the \ref BeliefFormula pointed by m_bf1.
   *  \return The \ref BeliefFormula pointed by m_bf1.
   */
  [[nodiscard]] const BeliefFormula &get_bf1() const;

  /** \brief Getter of the \ref BeliefFormula pointed by m_bf2.
   *  \return The \ref BeliefFormula pointed by m_bf2.
   */
  [[nodiscard]] const BeliefFormula &get_bf2() const;

  /** \brief Getter for the field m_operator.
   *  \return The value of the field m_operator.
   */
  [[nodiscard]] BeliefFormulaOperator get_operator() const noexcept;

  /** \brief Getter for the field m_group_agents.
   *  \return The value of the field m_group_agents.
   */
  [[nodiscard]] const AgentsSet &get_group_agents() const noexcept;
  ///@}

  // --- Other public methods ---
  /**
   * \brief Checks if the second belief formula (m_bf2) is empty.
   * \return true if m_bf2 is null, false otherwise.
   */
  [[nodiscard]] bool is_bf2_null() const;

  /** \brief Function that prints *this* (std::string parameters
   * representation).    */
  void print() const;

  /** \brief The equality operator for \ref BeliefFormula.
   *  \param[in] to_compare The \ref BeliefFormula to compare with *this*.
   *  \return true if the given \ref BeliefFormula is syntactically equal to
   * *this*, false otherwise.
   */
  [[nodiscard]] bool operator==(const BeliefFormula &to_compare) const;

  /** \brief The less-than operator for set operations.
   *  \param[in] to_compare The State to compare to *this*.
   *  \return true if *this* is smaller than to_compare, false otherwise.
   */
  [[nodiscard]] bool operator<(const BeliefFormula &to_compare) const;

private:
  // --- Data members ---
  /// \brief Discriminator for the normalized formula node stored in this object.
  BeliefFormulaType m_formula_type = BeliefFormulaType::BF_EMPTY;

  /// \brief Grounded propositional content used when \ref m_formula_type is
  /// \ref FLUENT_FORMULA.
  FluentFormula m_fluent_formula;

  /// \brief Single grounded agent used by \ref BELIEF_FORMULA nodes.
  Agent m_agent;

  /// \brief Propositional operator used when \ref m_formula_type is
  /// \ref PROPOSITIONAL_FORMULA.
  BeliefFormulaOperator m_operator{};

  /// \brief Grounded agent group used by \ref E_FORMULA and \ref C_FORMULA nodes.
  AgentsSet m_group_agents;

  /// \brief First child formula for unary and binary normalized nodes.
  std::unique_ptr<BeliefFormula> m_bf1; // Check if shared pointer is better

  /// \brief Second child formula for binary normalized nodes when present.
  std::unique_ptr<BeliefFormula> m_bf2; // Check if shared pointer is better
};
