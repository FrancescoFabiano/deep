/**
 * \class BeliefFormulaParsed
 * \brief Class that implements a parsed Belief Formula (string-based fields).
 *
 * \details A \ref BeliefFormulaParsed can have several forms:
 *    - \ref FLUENT_FORMULA -- \ref fluent_formula;
 *    - \ref BELIEF_FORMULA -- B(\ref agent, *phi*);
 *    - \ref PROPOSITIONAL_FORMULA -- \ref BF_NOT(*phi*) or (*phi_1* \ref BF_AND *phi_2*) or (*phi_1* \ref BF_OR *phi_2*);
 *    - \ref E_FORMULA -- E([set of \ref agent], *phi*);
 *    - \ref C_FORMULA -- C([set of \ref agent], *phi*);
 *    - \ref D_FORMULA -- D([set of \ref agent], *phi*);
 *
 * \see reader, domain
 *
 * \todo Maybe implement the "move" so the reader actually moves the object instead of copying them.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 16, 2025
 */
#pragma once

#include <memory>
#include "utilities/Define.h"
#include "formulae/BeliefFormula.h"


class BeliefFormulaParsed
{
public:
    /// \name Constructors
    ///@{
    BeliefFormulaParsed() = default;
    ~BeliefFormulaParsed() = default;
    ///@}

    /// \name Setters
    ///@{
    /** \brief Setter for the field m_string_fluent_formula. */
    void set_string_fluent_formula(const StringSetsSet & to_set);

    /** \brief Setter for the field m_string_agent. */
    void set_string_agent(const std::string & to_set);

    /** \brief Setter for the field m_string_group_agents. */
    void set_string_group_agents(const StringsSet & to_set);

    /** \brief Setter of the field m_bf1. */
    void set_bf1(const BeliefFormulaParsed & to_set);

    /** \brief Setter of the field m_bf2. */
    void set_bf2(const BeliefFormulaParsed & to_set);

    /** \brief Setter for the field m_formula_type. */
    void set_formula_type(BeliefFormulaType to_set);

    /** \brief Setter for the field m_operator. */
    void set_operator(BeliefFormulaOperator to_set);

    /** \brief Setter from a fluent_formula (string-based). */
    void set_from_ff(const StringSetsSet & to_build);
    ///@}

    /// \name Getters
    ///@{
    /** \brief Getter for the field m_formula_type. */
    [[nodiscard]] BeliefFormulaType get_formula_type() const noexcept;

    /** \brief Getter for the field m_string_fluent_formula. */
    [[nodiscard]] const StringSetsSet & get_string_fluent_formula() const noexcept;

    /** \brief Getter for the field m_string_agent. */
    [[nodiscard]] const std::string & get_string_agent() const noexcept;

    /** \brief Getter for the field m_string_group_agents. */
    [[nodiscard]] const StringsSet & get_string_group_agents() const noexcept;

    /** \brief Getter of the \ref BeliefFormulaParsed pointed by m_bf1. */
    [[nodiscard]] const BeliefFormulaParsed & get_bf1() const;

    /** \brief Getter of the \ref BeliefFormulaParsed pointed by m_bf2. */
    [[nodiscard]] const BeliefFormulaParsed & get_bf2() const;

    /** \brief Getter for the field m_operator. */
    [[nodiscard]] BeliefFormulaOperator get_operator() const noexcept;

    /** \brief Getter for the field m_string_group_agents (alias for group agents). */
    [[nodiscard]] const StringsSet & get_group_agents() const noexcept;
    ///@}

private:
    // --- Data members (all string-based) ---
    BeliefFormulaType m_formula_type = BeliefFormulaType::BF_EMPTY;
    StringSetsSet m_string_fluent_formula;
    std::string m_string_agent;
    BeliefFormulaOperator m_operator{};
    StringsSet m_string_group_agents;
    std::shared_ptr<BeliefFormulaParsed> m_bf1;
    std::shared_ptr<BeliefFormulaParsed> m_bf2;
};

using ParsedFormulaeList = std::list<BeliefFormulaParsed>; ///< CNF formula of BeliefFormula.
