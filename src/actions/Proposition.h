#pragma once

#include <list>
#include <string>
#include "formulae/BeliefFormula.h"
#include "utilities/Define.h"

/**
 * \enum PropositionType
 * \brief The possible types of proposition found in the domain description.
 */
enum class PropositionType {
    EXECUTABILITY,  ///< Specifies an action executability condition: *act* exec if *phi*
    ONTIC,          ///< Specifies the effects of an ontic action: *act* causes *f*
    SENSING,        ///< Specifies the effects of a sensing action: *act* sensed *f*
    ANNOUNCEMENT,   ///< Specifies the effects of an announcement action: *act* announces *ff*
    OBSERVANCE,     ///< Specifies the full observability conditions: *ag* observes *act*
    AWARENESS,      ///< Specifies the partial observability conditions: *ag* aware *act*
    NOTSET          ///< Default case
};

/**
 * \class Proposition
 * \brief Support class to elaborate the proposition of the Action(s) in the domain.
 *
 * A Proposition identifies and specifies an Action behavior (executability, effects, observability).
 * The content of a Proposition is not grounded; it will be grounded once inserted in an Action.
 *
 * \see Action
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 16, 2025
 */
class Proposition {
public:
    /// \name Constructors
    ///@{
    Proposition() = default;
    ///@}

    /// \name Getters
    ///@{
    /** \brief Gets the type of this proposition. */
    [[nodiscard]] PropositionType get_type() const noexcept;

    /** \brief Gets the name of this proposition (used as id). */
    [[nodiscard]] const std::string& get_action_name() const noexcept;

    /** \brief Gets the grounded action effect. */
    [[nodiscard]] FluentFormula get_action_effect() const;

    /** \brief Gets the grounded agent. */
    [[nodiscard]] Agent get_agent() const;

    /** \brief Gets the observability conditions. */
    [[nodiscard]] const BeliefFormula& get_observability_conditions() const noexcept;

    /** \brief Gets the executability conditions. */
    [[nodiscard]] const BeliefFormula& get_executability_conditions() const noexcept;

    ///@}

    /// \name Setters
    ///@{
    /** \brief Sets the type of this proposition. */
    void set_type(PropositionType to_set) noexcept;

    /** \brief Sets the name of this proposition. */
    void set_action_name(const std::string& to_set);

    /** \brief Sets the action effect. */
    void set_action_effect(const StringSetsSet& to_set);

    /** \brief Adds an action effect. */
    void add_action_effect(const StringsSet& to_add);

    /** \brief Sets the agent. */
    void set_agent(const std::string& to_set);

    /** \brief Sets the observability conditions. */
    void set_observability_conditions(const BeliefFormula& to_set);

    /** \brief Sets the executability conditions. */
    void set_executability_conditions(const BeliefFormula& to_set);
    ///@}

    /**
     * \brief Converts a PropositionType to its string representation.
     * \param type The PropositionType to convert.
     * \return The string representation.
     */
    static std::string type_to_string(PropositionType type);

    /// \name Utilities
    ///@{
    /** \brief Prints this proposition.
     *
     * \param os The output stream to print to (default: std::cout).
     */
    void print(std::ostream &os = std::cout) const;

    /** \brief Grounds this proposition. */
    //void ground();
    ///@}

private:
    PropositionType m_type {PropositionType::NOTSET}; ///< The type of this proposition.
    std::string m_action_name;                        ///< The name/id of this proposition.
    StringSetsSet m_action_effect;                    ///< Effects (for ONTIC, SENSING, ANNOUNCEMENT).
    std::string m_agent;                              ///< Agent (for OBSERVANCE, AWARENESS).
    BeliefFormula m_observability_conditions;        ///< Observability condition (not grounded).
    BeliefFormula m_executability_conditions;        ///< Executability condition (not grounded).
};

/// \brief A list of Proposition objects.
using PropositionsList = std::list<Proposition>;