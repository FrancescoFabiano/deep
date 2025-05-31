/**
 * \class StateLevel
 * \brief Class that implements a state layer of the epistemic planning graph data structure.
 *
 * \details In this implementation, the state layer contains complete e-state in order to have a complete planning graph.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano.
 * \date May 31, 2025
 */

#pragma once

#include <map>

#include "BeliefFormula.h"
#include "State.h"

/**
 * \brief Type alias for the map associating each grounded fluent to a score.
 */
using PG_FluentsScoreMap = std::map<Fluent, short>;

/**
 * \brief Type alias for the map associating each grounded BeliefFormula to a score.
 */
using PG_BeliefFormulaeMap = std::map<BeliefFormula, short>;

/**
 * \brief Represents a level in the epistemic planning graph containing state information.
 */
class StateLevel {
public:
    /// \name Constructors
    ///@{

    /**
     * \brief Default constructor. Sets the depth to 0 and initializes the maps.
     */
    StateLevel() = default;

    /**
     * \brief Copy constructor.
     * \param to_assign The StateLevel to copy.
     */
    StateLevel(const StateLevel& to_assign);

    /**
     * \brief Constructor that sets the depth and the maps.
     * \param f_map The map with the fluents' truth values.
     * \param bf_map The map with the belief formulas' truth values.
     * \param depth The value to assign to m_depth.
     */
    StateLevel(const PG_FluentsScoreMap& f_map, const PG_BeliefFormulaeMap& bf_map, unsigned short depth);

    ///@}

    /// \name Initialization
    ///@{

    /**
     * \brief Initializes the state level with the given goal formulae list.
     * \param goal The list of goal formulae.
     */
    void initialize(const FormulaeList& goal);

    /**
     * \brief Initializes the state level with the given goal formulae list and e-state.
     * \tparam T The state representation type.
     * \param goals The list of goal formulae.
     * \param eState The epistemic state.
     */
    template<StateRepresentation T>
    void initialize(const FormulaeList& goals, State<T>& eState)
    {
        build_init_f_map(eState);
        build_init_bf_map(goals, eState);
    }


    ///@}

    /// \name Assignment
    ///@{

    /**
     * \brief Assignment operator.
     * \param to_assign The object to copy into this.
     * \return Reference to this.
     */
    StateLevel& operator=(const StateLevel& to_assign);

    /**
     * \brief Sets this state level to another.
     * \param to_assign The StateLevel to assign.
     */
    void set_state_level(const StateLevel& to_assign);

    ///@}

    /// \name Setters
    ///@{

    /**
     * \brief Sets the depth of this state level.
     * \param to_set The value to assign to m_depth.
     */
    void set_depth(unsigned short to_set);

    /**
     * \brief Modifies a pair fluent, bool in the field m_pg_f_map.
     * \param key The key of the pair.
     * \param value The value of the pair.
     */
    void modify_fluent_value(const Fluent& key, short value);

    /**
     * \brief Modifies a pair BeliefFormula, bool in the field m_pg_bf_map.
     * \param key The key of the pair.
     * \param value The value of the pair.
     */
    void modify_bf_value(const BeliefFormula& key, short value);

    ///@}

    /// \name Getters
    ///@{

    /**
     * \brief Gets the map of fluents' truth values.
     * \return The value of m_pg_f_map.
     */
    [[nodiscard]] const PG_FluentsScoreMap& get_f_map() const;

    /**
     * \brief Gets the map of belief formulas' truth values.
     * \return The value of m_pg_bf_map.
     */
    [[nodiscard]] const PG_BeliefFormulaeMap& get_bf_map() const;

    /**
     * \brief Gets the depth of this state level.
     * \return The value of m_depth.
     */
    [[nodiscard]] unsigned short get_depth() const;

    /**
     * \brief Gets the score from the depth.
     * \return The score computed from the depth.
     */
    [[nodiscard]] short get_score_from_depth() const;

    ///@}

    /// \name Entailment and Executability
    ///@{

    /**
     * \brief Checks satisfaction of a fluent on this state level.
     * \param f The fluent to check for entailment.
     * \return true if the fluent is entailed, false otherwise.
     */
    [[nodiscard]] bool pg_entailment(const Fluent& f) const;

    /**
     * \brief Checks satisfaction of a BeliefFormula on this state level.
     * \param bf The BeliefFormula to check for entailment.
     * \return true if the formula is entailed, false otherwise.
     */
    [[nodiscard]] bool pg_entailment(const BeliefFormula& bf) const;

    /**
     * \brief Checks satisfaction of a CNF of BeliefFormula on this state level.
     * \param fl The CNF of BeliefFormula to check for entailment.
     * \return true if the formula is entailed, false otherwise.
     */
    [[nodiscard]] bool pg_entailment(const FormulaeList& fl) const;

    /**
     * \brief Checks if an action is executable on this state level.
     * \param act The action to check for executability.
     * \return true if the action's executability conditions are entailed, false otherwise.
     */
    [[nodiscard]] bool pg_executable(const Action& act) const;

    ///@}

    /// \name Successor Computation
    ///@{

    /**
     * \brief Computes the successor state level given an action and its predecessor.
     * \param act The action to apply.
     * \param predecessor The previous state level.
     * \param false_bf The set of formulas that become false.
     * \return true if the successor was computed successfully, false otherwise.
     */
    bool compute_successor(const Action& act, const StateLevel& predecessor, FormulaeSet& false_bf);

    ///@}

private:
    /// \brief The map that associates each grounded fluent to TRUE or FALSE.
    PG_FluentsScoreMap m_pg_f_map = {};

    /// \brief The map that associates each grounded BeliefFormula to TRUE or FALSE.
    PG_BeliefFormulaeMap m_pg_bf_map = {};

    /// \brief The depth of this state level.
    unsigned short m_depth = 0;

    // --- Internal Setters ---

    /**
     * \brief Sets the fluent map for this state level.
     * \param to_set The fluent map to assign.
     */
    void set_f_map(const PG_FluentsScoreMap& to_set);

    /**
     * \brief Sets the belief formula map for this state level.
     * \param to_set The belief formula map to assign.
     */
    void set_bf_map(const PG_BeliefFormulaeMap& to_set);

    // --- Internal Getters ---

    /**
     * \brief Gets the value associated with a fluent in the fluent map.
     * \param key The fluent whose value is to be retrieved.
     * \return The value associated with the fluent.
     */
    [[nodiscard]] short get_fluent_value(const Fluent& key) const;

    /**
     * \brief Gets the value associated with a belief formula in the belief formula map.
     * \param key The belief formula whose value is to be retrieved.
     * \return The value associated with the belief formula.
     */
    [[nodiscard]] short get_bf_value(const BeliefFormula& key) const;

    // --- Initialization helpers ---

    /**
     * \brief Builds the initial fluent map for this state level.
     */
    void build_init_f_map();

    /**
     * \brief Builds the initial belief formula map for this state level.
     * \param goals The list of goal formulae.
     */
    void build_init_bf_map(const FormulaeList& goals);

    /**
     * \brief Inserts all subformulas of a list of belief formulas into the belief formula map with a given value.
     * \param fl The list of belief formulas.
     * \param value The value to assign to each subformula.
     */
    void insert_subformula_bf(const FormulaeList& fl, short value);

    /**
     * \brief Inserts all subformulas of a belief formula into the belief formula map with a given value.
     * \param bf The belief formula.
     * \param value The value to assign to each subformula.
     */
    void insert_subformula_bf(const BeliefFormula& bf, short value);

    /**
     * \brief Inserts all subformulas of a list of belief formulas into the belief formula map using entailment from the given state.
     * \tparam T The state representation type.
     * \param fl The list of belief formulas.
     * \param eState The epistemic state.
     */
    template<StateRepresentation T>
    void insert_subformula_bf(const FormulaeList& fl, State<T>& eState)
    {
        for (const auto& formula : fl) {
            insert_subformula_bf(formula, eState);
        }
    }

    /**
     * \brief Inserts all subformulas of a belief formula into the belief formula map using entailment from the given state.
     * \tparam T The state representation type.
     * \param bf The belief formula.
     * \param eState The epistemic state.
     */
    template<StateRepresentation T>
    void insert_subformula_bf(const BeliefFormula& bf, State<T>& eState)
    {
        short value = eState.entails(bf) ? 0 : -1;

        switch (bf.get_formula_type()) {
            case BeliefFormulaType::BELIEF_FORMULA:
                if (m_pg_bf_map.emplace(bf, value).second) {
                    insert_subformula_bf(bf.get_bf1(), eState);
                }
                break;

            case BeliefFormulaType::PROPOSITIONAL_FORMULA:
                switch (bf.get_operator()) {
                    case BeliefFormulaOperator::BF_NOT:
                        if (m_pg_bf_map.emplace(bf, value).second) {
                            insert_subformula_bf(bf.get_bf1(), eState);
                        }
                        break;
                    case BeliefFormulaOperator::BF_OR:
                    case BeliefFormulaOperator::BF_AND:
                        if (m_pg_bf_map.emplace(bf, value).second) {
                            insert_subformula_bf(bf.get_bf1(), eState);
                            insert_subformula_bf(bf.get_bf2(), eState);
                        }
                        break;
                    case BeliefFormulaOperator::BF_FAIL:
                    default:
                        ExitHandler::exit_with_message(
                            ExitHandler::ExitCode::BeliefFormulaOperatorUnset,
                            "Error: Unexpected operator in PROPOSITIONAL_FORMULA while generating subformulas for the Planning Graph."
                        );
                }
                break;

            case BeliefFormulaType::C_FORMULA:
                if (m_pg_bf_map.emplace(bf, value).second) {
                    insert_subformula_bf(bf.get_bf1(), eState);
                }
                break;

            case BeliefFormulaType::FLUENT_FORMULA:
            case BeliefFormulaType::BF_EMPTY:
                break;

            case BeliefFormulaType::BF_TYPE_FAIL:
            default:
                ExitHandler::exit_with_message(
                    ExitHandler::ExitCode::BeliefFormulaTypeUnset,
                    "Error: Unexpected formula type in insert_subformula_bf while generating subformulas for the Planning Graph."
                );
        }
    }

    /**
     * \brief Extracts the base fluents from a belief formula.
     * \param bf The belief formula.
     * \param bf_base_fluents The set to store the base fluents.
     */
    static void get_base_fluents(const BeliefFormula& bf, FluentsSet& bf_base_fluents);

    // --- Action application helpers ---

    /**
     * \brief Executes the ontic part of an action on this state level.
     * \param act The action to apply.
     * \param predecessor The previous state level.
     * \param false_bf The set of formulas that become false.
     * \return true if the ontic effects were applied successfully, false otherwise.
     */
    bool exec_ontic(const Action& act, const StateLevel& predecessor, FormulaeSet& false_bf);

    /**
     * \brief Executes the epistemic part of an action on this state level.
     * \param act The action to apply.
     * \param predecessor The previous state level.
     * \param false_bf The set of formulas that become false.
     * \return true if the epistemic effects were applied successfully, false otherwise.
     */
    bool exec_epistemic(const Action& act, const StateLevel& predecessor, FormulaeSet& false_bf);

    /**
     * \brief Applies ontic effects to a belief formula.
     * \param bf The belief formula to modify.
     * \param fl The set of formulas to update.
     * \param fully The set of fully observant agents.
     * \param modified_pg Flag indicating if the planning graph was modified.
     * \return true if the ontic effects were applied successfully, false otherwise.
     */
    bool apply_ontic_effects(const BeliefFormula& bf, FormulaeSet& fl, const AgentsSet& fully, bool& modified_pg);

    /**
     * \brief Applies epistemic effects to a fluent and belief formula.
     * \param effect The fluent effect.
     * \param bf The belief formula to modify.
     * \param fl The set of formulas to update.
     * \param fully The set of fully observant agents.
     * \param partially The set of partially observant agents.
     * \param modified_pg Flag indicating if the planning graph was modified.
     * \param vis_cond The visibility condition.
     * \return true if the epistemic effects were applied successfully, false otherwise.
     */
    bool apply_epistemic_effects(const Fluent& effect, const BeliefFormula& bf, FormulaeSet& fl, const AgentsSet& fully, const AgentsSet& partially, bool& modified_pg, unsigned short vis_cond);
};
