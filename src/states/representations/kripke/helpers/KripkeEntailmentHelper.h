#pragma once

/**
 * \brief Helper class for entailment checks in Kripke structures.
 *
 * Provides static methods to check whether various formulae
 * (fluents, sets, formulas, belief formulas) are entailed in a given world, state, etc.
 *
 *
 * We use this class to alleviate the logic of KripkeState so we make everything private accessible only form KripkeState
 * \author Francesco Fabiano
 * \date May 17, 2025
 */

#include "utilities/Define.h"

class KripkeEntailmentHelper {
public:
    /**
     * \brief Default constructor.
     */
    KripkeEntailmentHelper() = default;

    /**
     * \brief Default destructor.
     */
    ~KripkeEntailmentHelper() = default;

    /**
     * \brief Copy constructor.
     */
    KripkeEntailmentHelper(const KripkeEntailmentHelper&) = default;

    /**
     * \brief Copy assignment operator.
     */
    KripkeEntailmentHelper& operator=(const KripkeEntailmentHelper&) = default;

    /**
     * \brief Move constructor.
     */
    KripkeEntailmentHelper(KripkeEntailmentHelper&&) = default;

    /**
     * \brief Move assignment operator.
     */
    KripkeEntailmentHelper& operator=(KripkeEntailmentHelper&&) = default;

    /// \name Entailment for KripkeWorld
    ///@{
    /**
     * \brief Check if a fluent is entailed by this world.
     * \param[in] to_check The fluent to check.
     * \param[in] world The KripkeWorld to check.
     * \return True if entailed, false otherwise.
     */
    [[nodiscard]] static bool entails(const Fluent& to_check, const KripkeWorld& world);

    /**
     * \brief Check if a conjunctive set of fluents is entailed by this world.
     * \param[in] to_check The set of fluents.
     * \param[in] world The KripkeWorld to check.
     * \return True if all are entailed, false otherwise.
     */
    [[nodiscard]] static bool entails(const FluentsSet& to_check, const KripkeWorld& world);

    /**
     * \brief Check if a DNF fluent formula is entailed by this world.
     * \param[in] to_check The formula to check.
     * \param[in] world The KripkeWorld to check.
     * \return True if entailed, false otherwise.
     */
    [[nodiscard]] static bool entails(const FluentFormula& to_check, const KripkeWorld& world);
    ///@}

    /// \name Entailment for KripkeWorldPointer
    ///@{
    /**
     * \brief Check if a fluent is entailed in a given Kripke world pointer.
     * \param[in] to_check The fluent to check.
     * \param[in] world The KripkeWorldPointer to check.
     * \return True if the fluent is entailed, false otherwise.
     */
    [[nodiscard]] static bool entails(const Fluent& to_check, const KripkeWorldPointer& world);

    /**
     * \brief Check if a set of fluents is entailed in a given Kripke world pointer.
     * \param[in] to_check The set of fluents to check.
     * \param[in] world The KripkeWorldPointer to check.
     * \return True if all fluents are entailed, false otherwise.
     */
    [[nodiscard]] static bool entails(const FluentsSet& to_check, const KripkeWorldPointer& world);

    /**
     * \brief Check if a fluent formula is entailed in a given Kripke world pointer.
     * \param[in] to_check The fluent formula to check.
     * \param[in] world The KripkeWorldPointer to check.
     * \return True if the formula is entailed, false otherwise.
     */
    [[nodiscard]] static bool entails(const FluentFormula& to_check, const KripkeWorldPointer& world);
    ///@}

    /// \name Entailment for KripkeState
    ///@{
    /**
     * \brief Check if a BeliefFormula is entailed in a set of Kripke world pointers.
     * \param[in] to_check The BeliefFormula to check.
     * \param[in] world The KripkeWorld pointer where to check entailment.
     * \param[in] kstate The KripkeState to use for reachability.
     * \return True if the formula is entailed, false otherwise.
     */
    [[nodiscard]] static bool entails(const BeliefFormula& to_check, const KripkeWorldPointer& world, const KripkeState& kstate);

    /**
     * \brief Check if a BeliefFormula is entailed in a given Kripke world, using a KripkeState.
     * \param[in] to_check The BeliefFormula to check.
     * \param[in] kstate The KripkeState to use for reachability. (also containing the pointed)
     * \return True if the formula is entailed, false otherwise.
     */
    [[nodiscard]] static bool entails(const BeliefFormula& to_check, const KripkeState& kstate);

    /**
     * \brief Check if a BeliefFormula is entailed in a set of Kripke worlds.
     * \param[in] to_check The BeliefFormula to check.
     * \param[in] worlds The set of KripkeWorld pointers to check.
    * \param[in] kstate The KripkeState to use for reachability.
     * \return True if the formula is entailed in all worlds, false otherwise.
     */
    [[nodiscard]] static bool entails(const BeliefFormula& to_check, const KripkeWorldPointersSet& worlds,const KripkeState & kstate);


    /**
 * \brief Check if a CNF of BeliefFormulae is entailed in a given Kripke world.
 * \param[in] to_check The list of BeliefFormulae to check (CNF).
 * \param[in] kstate The KripkeState to use for reachability (will give the pointed).
 * \return True if all formulae are entailed, false otherwise.
 */
    [[nodiscard]] static bool entails(const FormulaeList& to_check, const KripkeState& kstate);

    /** \brief Check epistemic properties after an action execution.
    *  \param[in] fully The set of fully observant agents.
    *  \param[in] partially The set of partially observant agents.
    *  \param[in] effects The effects of the action.
    *  \param[in] updated The updated KripkeState.
    *  \return True if all properties are respected, false otherwise.
    */
    [[nodiscard]] static bool check_properties(const AgentsSet &fully, const AgentsSet &partially, const FluentFormula &effects,
                          const KripkeState &updated);

    ///@}
    ///

    /*We use this class to alleviate the logic of KripkeState so we make everything private accessible only form KripkeState*/
    friend class KripkeState;
    friend class KripkeReachabilityHelper;
};
