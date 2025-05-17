/**
 * \class BeliefFormula
 * \brief Class that implements a Belief Formula.
 *
 * \details A \ref BeliefFormula can have several forms:
 *    - \ref FLUENT_FORMULA -- \ref fluent_formula;
 *    - \ref BELIEF_FORMULA -- B(\ref agent, *phi*);
 *    - \ref PROPOSITIONAL_FORMULA -- \ref BF_NOT(*phi*) or (*phi_1* \ref BF_AND *phi_2*) or (*phi_1* \ref BF_OR *phi_2*);
 *    - \ref E_FORMULA -- E([set of \ref agent], *phi*);
 *    - \ref C_FORMULA -- C([set of \ref agent], *phi*);
 *    - \ref D_FORMULA -- D([set of \ref agent], *phi*);
 * 
 * \see reader, domain
 * 
 * \todo With new parser maybe implement the "move" so the reader actually moves the object instead of copying them.
 *
 * \copyright GNU Public License.
 * 
 * \author Francesco Fabiano.
 * \date May 16, 2025
 */
#pragma once

#include <iostream>
#include <memory>

#include "BeliefFormulaParsed.h"
#include "utilities/Define.h"

/**
 * \brief The possible types of \ref BeliefFormula.
 */
enum class BeliefFormulaType
{
    FLUENT_FORMULA,           ///< A \ref BeliefFormula is also a \ref fluent_formula (base case for recursion).
    BELIEF_FORMULA,           ///< A \ref BeliefFormula of the form B(\ref agent, *phi*).
    PROPOSITIONAL_FORMULA,    ///< A \ref BeliefFormula composed with logical operators and \ref BeliefFormula(e).
    E_FORMULA,                ///< A \ref BeliefFormula of the form E([set of \ref agent], *phi*).
    C_FORMULA,                ///< A \ref BeliefFormula of the form C([set of \ref agent], *phi*).
    D_FORMULA,                ///< A \ref BeliefFormula of the form D([set of \ref agent], *phi*).
    BF_EMPTY,                 ///< When the belief formula is empty.
    BF_TYPE_FAIL              ///< The failure case.
};

/**
 * \brief The logical operator for \ref BeliefFormula(e).
 *
 * These are used in the case that the \ref bf_type of a \ref BeliefFormula is \ref PROPOSITIONAL_FORMULA.
 */
enum class BeliefFormulaOperator
{
    BF_AND,      ///< The AND between \ref BeliefFormula(e).
    BF_OR,       ///< The OR between \ref BeliefFormula(e).
    BF_NOT,      ///< The NOT of a \ref BeliefFormula.
    BF_INPAREN,  ///< When the \ref BeliefFormula is only surrounded by "()".
    BF_FAIL      ///< When the \ref BeliefFormula is not set properly (shouldn't be accessed if not \ref PROPOSITIONAL_FORMULA).
};

class BeliefFormula
{
public:
    /// \name Constructors
    ///@{
    /** \brief Empty Constructor */
    BeliefFormula() = default;

    /** \brief Constructor that takes a \ref BeliefFormulaParsed object.
     *  \details This constructor is used to create a \ref BeliefFormula from a parsed one.
     *
     *  \param[in] to_ground The \ref BeliefFormulaParsed to copy in *this*.
    */
    explicit BeliefFormula(const BeliefFormulaParsed& to_ground);

    /** \brief Copy Constructor
     *  \param[in] to_copy The \ref BeliefFormula to copy in *this*.
     */
    BeliefFormula(const BeliefFormula & to_copy);

    /** \brief Move Constructor */
    BeliefFormula(BeliefFormula &&) noexcept = default;

    /** \brief Destructor */
    ~BeliefFormula() = default;

    /** \brief Copy Assignment */
    BeliefFormula & operator=(const BeliefFormula & to_copy);

    /** \brief Move Assignment */
    BeliefFormula & operator=(BeliefFormula &&) noexcept = default;
    ///@}

    /// \name Setters
    ///@{
    /** \brief Setter for the field m_fluent_formula.
     *  \param[in] to_set The \ref fluent_formula object to copy in m_fluent_formula.
     */
    void set_fluent_formula(const FluentFormula& to_set);

    /** \brief Setter for the field m_fluent_formula from a fluent.
     *  \param[in] to_set The \ref fluent object to insert in m_fluent_formula.
     */
    void set_fluent_formula_from_fluent(const Fluent& to_set);

    /** \brief Setter for the field m_agent.
     *  \param[in] to_set The \ref agent object to copy in m_agent.
     */
    void set_agent(const Agent& to_set);

    /** \brief Setter for the field m_group_agents.
     *  \param[in] to_set The \ref agent_set object to copy in m_group_agents.
     */
    void set_group_agents(const AgentsSet& to_set);

    /** \brief Setter of the field m_bf1.
     *  This setter takes a \ref BeliefFormula and sets m_bf1 to be its pointer.
     *  \param[in] to_set The \ref BeliefFormula to be pointed by m_bf1.
     */
    void set_bf1(const BeliefFormula & to_set);

    /** \brief Setter of the field m_bf2.
     *  This setter takes a \ref BeliefFormula and sets m_bf2 to be its pointer.
     *  \param[in] to_set The \ref BeliefFormula to be pointed by m_bf2.
     */
    void set_bf2(const BeliefFormula & to_set);

    /** \brief Setter of the field m_bf1 given a \ref BeliefFormulaParsed.
     *  This setter takes a \ref BeliefFormulaParsed and sets m_bf1 to be its pointer.
     *  \param[in] to_set The \ref BeliefFormulaParsed to be pointed by m_bf1.
     */
    void set_bf1(const BeliefFormulaParsed & to_set);

    /** \brief Setter of the field m_bf2 given a \ref BeliefFormulaParsed.
     *  This setter takes a \ref BeliefFormulaParsed and sets m_bf2 to be its pointer.
     *  \param[in] to_set The \ref BeliefFormulaParsed to be pointed by m_bf2.
     */
    void set_bf2(const BeliefFormulaParsed & to_set);

    /** \brief Setter for the field m_formula_type.
     *  \param[in] to_set The \ref bf_type object to copy in m_formula_type.
     */
    void set_formula_type(BeliefFormulaType to_set);

    /** \brief Setter for the field m_operator.
     *  \param[in] to_set The \ref bf_operator object to copy in m_operator.
     */
    void set_operator(BeliefFormulaOperator to_set);

    /** \brief Setter from a fluent_formula. */
    void set_from_ff(const FluentFormula & to_build);
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
    [[nodiscard]] const FluentFormula& get_fluent_formula() const noexcept;

    /** \brief Getter for the field m_agent.
     *  \return The value of the field m_agent.
     */
    [[nodiscard]] const Agent & get_agent() const noexcept;

    /** \brief Getter of the \ref BeliefFormula pointed by m_bf1.
     *  \return The \ref BeliefFormula pointed by m_bf1.
     */
    [[nodiscard]] const BeliefFormula & get_bf1() const;

    /** \brief Getter of the \ref BeliefFormula pointed by m_bf2.
     *  \return The \ref BeliefFormula pointed by m_bf2.
     */
    [[nodiscard]] const BeliefFormula & get_bf2() const;


    /** \brief Getter for the field m_operator.
     *  \return The value of the field m_operator.
     */
    [[nodiscard]] BeliefFormulaOperator get_operator() const noexcept;

    /** \brief Getter for the field m_group_agents.
     *  \return The value of the field m_group_agents.
     */
    [[nodiscard]] const AgentsSet& get_group_agents() const noexcept;
    ///@}

    // --- Other public methods ---
    /**
     * \brief Checks if the second belief formula (m_bf2) is empty.
     * \return true if m_bf2 is null, false otherwise.
     */
    [[nodiscard]] bool is_bf2_null() const;

    /** \brief Function that prints *this* (std::string parameters representation).
     * \param os The output stream to print to (default: std::cout).
     */
    void print( std::ostream& os = std::cout) const;

    /** \brief The equality operator for \ref BeliefFormula.
     *  \param[in] to_compare The \ref BeliefFormula to compare with *this*.
     *  \return true if the given \ref BeliefFormula is syntactically equal to *this*, false otherwise.
     */
    [[nodiscard]] bool operator==(const BeliefFormula& to_compare) const;

    /** \brief The less-than operator for set operations.
     *  \param[in] to_compare The \ref state to compare to *this*.
     *  \return true if *this* is smaller than to_compare, false otherwise.
     */
    [[nodiscard]] bool operator<(const BeliefFormula& to_compare) const;

private:
    // --- Data members ---
    BeliefFormulaType m_formula_type = BeliefFormulaType::BF_EMPTY;
    FluentFormula m_fluent_formula;
    Agent m_agent;
    BeliefFormulaOperator m_operator{};
    AgentsSet m_group_agents;
    std::shared_ptr<BeliefFormula> m_bf1;
    std::shared_ptr<BeliefFormula> m_bf2;
};

