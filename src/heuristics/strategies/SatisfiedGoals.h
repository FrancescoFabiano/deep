/**
 * \class SatisfiedGoals
 * \brief Implements the heuristic "number of satisfied sub-goals".
 *
 * \details This class checks how many sub-goals (elements of \ref Domain::get_instance().get_goal_description())
 * each state satisfies and selects the states that satisfy the highest number of sub-goals for expansion.
 *
 * As an extension, complex belief formulas can be broken into multiple components for more sub-goals and accuracy.
 * For example, \ref C_FORMULA can be split into multiple components.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date November 28, 2019
 */

#pragma once

#include "utilities/Define.h"
#include "states/State.h"
#include "formulae/BeliefFormula.h"

class SatisfiedGoals
{
public:
    /// \name Singleton Access
    ///@{
    /**
     * \brief Provides access to the singleton instance of \ref SatisfiedGoals.
     * \warning The \ref set method must be called in the heuristics manager file only.
     * \return A reference to the singleton instance.
     */
    static SatisfiedGoals& get_instance();
    ///@}

    /// \name Setters
    ///@{
    /**
     * \brief Sets the goals and their count for this instance.
     * \param[in] goals The goal description, expanded by the heuristics' manager.
     */
    void set(const FormulaeList& goals);
    ///@}

    /// \name Heuristic Computation
    ///@{
    /**
     * \brief Computes the number of unsatisfied goals for a given state.
     * \tparam T The state representation type.
     * \param[in] eState The state to evaluate.
     * \return The number of unsatisfied goals in the given state.
     */
    template <StateRepresentation T>
    [[nodiscard]] unsigned short get_unsatisfied_goals(const State<T>& eState) const
    {
        unsigned short ret = m_goals_number;

        for (const auto& goal : m_goals)
        {
            if (eState.entails(goal))
            {
                --ret;
            }
        }
        return ret;
    }

    ///@}

    /// \name Getters
    ///@{
    /**
     * \brief Retrieves the list of expanded sub-goals.
     * \return A constant reference to the list of sub-goals.
     */
    [[nodiscard]] const FormulaeList& get_goals() const noexcept;

    /**
     * \brief Retrieves the number of expanded sub-goals.
     * \return The number of sub-goals.
     */
    [[nodiscard]] unsigned short get_goals_number() const noexcept;
    ///@}

    /// \name Deleted Methods
    ///@{
    /** \brief Copy constructor deleted since this is a singleton class. */
    SatisfiedGoals(const SatisfiedGoals&) = delete;

    /** \brief Copy assignment operator deleted since this is a singleton class. */
    SatisfiedGoals& operator=(const SatisfiedGoals&) = delete;
    ///@}

private:
    /// \name Private Constructor
    ///@{
    /** \brief Private constructor for the singleton class. */
    SatisfiedGoals() = default;
    ///@}

    /// \name Private Setters
    ///@{
    /**
     * \brief Sets the list of expanded sub-goals.
     * \param[in] to_set The list of sub-goals to set.
     */
    void set_goals(const FormulaeList& to_set);

    /**
     * \brief Sets the number of expanded sub-goals.
     * \param[in] to_set The number of sub-goals to set.
     */
    void set_goals_number(unsigned short to_set);
    ///@}

    /// \name Data Members
    ///@{
    /** \brief The number of sub-goals. */
    unsigned short m_goals_number = 0;

    /** \brief A local copy of the goals, possibly modified to include more sub-goals. */
    FormulaeList m_goals;
    ///@}
};
