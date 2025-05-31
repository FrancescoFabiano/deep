/**
 * \class PlanningGraph
 * \brief Implements the epistemic planning graph data structure for heuristic extrapolation.
 *
 * \details An e-Planning graph is a structure introduced in <https://aaai.org/ocs/index.php/ICAPS/ICAPS18/paper/view/17733>
 * that, like the planning graph from classical planning, is used to extrapolate qualitative values for states.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano.
 * \date May 31, 2025
 */

#pragma once

#include "ActionLevel.h"
#include "StateLevel.h"
#include "utilities/Define.h"
#include "actions/Action.h"
#include <vector>

/**
 * \brief The PlanningGraph class for epistemic planning heuristics.
 */
class PlanningGraph {
public:
    /// \name Constructors
    ///@{

    /**
     * \brief Default constructor. Sets the initial State level and the goal from the domain.
     *
     * This version should be used when a single planning BisGraph is created from the initial State.
     */
    PlanningGraph();

    /**
     * \brief Constructor that sets the initial State level from the domain and the goal as given.
     * \param[in] goal The formula_list that describes the given goals.
     */
    explicit PlanningGraph(const FormulaeList& goal);

    /**
     * \brief Copy constructor.
     * \param[in] pg The PlanningGraph to copy.
     */
    PlanningGraph(const PlanningGraph& pg);

    /**
     * \brief Constructor that sets the initial State level from a given eState and the goal from the domain.
     * \tparam T The state representation type.
     * \param[in] eState The initial eState from which to extract the first State level.
     */
    template<StateRepresentation T>
    explicit PlanningGraph(State<T>& eState) {
        auto goals = Domain::get_instance().get_goal_description();
        StateLevel pg_init;
        pg_init.initialize(goals, eState);
        init(goals, pg_init);
    }

    /**
     * \brief Constructor that sets the initial State level from a given eState and goal description.
     * \tparam T The state representation type.
     * \param[in] goal The formula_list that describes the given goals.
     * \param[in] eState The initial eState from which to extract the first State level.
     */
    template<StateRepresentation T>
    PlanningGraph(const FormulaeList& goal, State<T>& eState) {
        StateLevel pg_init;
        pg_init.initialize(goal, eState);
        init(goal, pg_init);
    }
    ///@}

    /**
     * \brief Initializes the PlanningGraph fields.
     * \param[in] goal The formula_list that describes the given goals.
     * \param[in] pg_init The initial State level.
     */
    void init(const FormulaeList& goal, const StateLevel& pg_init);

    /**
     * \brief Sets the length of the PlanningGraph.
     * \param[in] length The value to assign to m_pg_length.
     */
    void set_length(unsigned short length);

    /**
     * \brief Sets the sum of the depths of the levels that contain a subgoal.
     * \param[in] sum The value to assign to m_pg_sum.
     */
    void set_sum(unsigned short sum);

    /**
     * \brief Sets the goal formulae list.
     * \param[in] goal The value to assign to m_goal.
     */
    void set_goal(const FormulaeList& goal);

    /**
     * \brief Checks if the planning BisGraph can satisfy the domain.
     * \return True if a solution is found with the planning BisGraph, false otherwise.
     */
    [[nodiscard]] bool is_satisfiable() const;

    /**
     * \brief Gets the length of the PlanningGraph.
     * \return The value assigned to m_pg_length.
     */
    [[nodiscard]] unsigned short get_length() const;

    /**
     * \brief Gets the sum of the depths of the levels that contain a subgoal.
     * \return The value assigned to m_pg_sum.
     */
    [[nodiscard]] unsigned short get_sum() const;

    /**
     * \brief Gets the state levels of the PlanningGraph.
     * \return The vector of StateLevel objects.
     */
    [[nodiscard]] const std::vector<StateLevel>& get_state_levels() const;

    /**
     * \brief Gets the action levels of the PlanningGraph.
     * \return The vector of ActionLevel objects.
     */
    [[nodiscard]] const std::vector<ActionLevel>& get_action_levels() const;

    /**
     * \brief Gets the goal formulae list.
     * \return The goal FormulaeList.
     */
    [[nodiscard]] const FormulaeList& get_goal() const;

    /**
     * \brief Gets the set of actions never executed in the PlanningGraph.
     * \return The set of never executed actions.
     */
    [[nodiscard]] const ActionsSet& get_never_executed() const;

    /**
     * \brief Gets the set of belief formulas that are false.
     * \return The set of false belief formulas.
     */
    [[nodiscard]] const FormulaeSet& get_belief_formula_false() const;

    /**
     * \brief Sets the PlanningGraph to another instance.
     * \param[in] to_assign The PlanningGraph to assign from.
     */
    void set_pg(const PlanningGraph& to_assign);

    /**
     * \brief Assignment operator.
     * \param[in] to_assign The PlanningGraph to assign from.
     * \return Reference to this PlanningGraph.
     */
    PlanningGraph& operator=(const PlanningGraph& to_assign);

    /**
     * \brief Gets the fluent scores map.
     * \return The map of fluents to their scores.
     */
    [[nodiscard]] const PG_FluentsScoreMap& get_f_scores() const;

    /**
     * \brief Gets the belief formula scores map.
     * \return The map of belief formulas to their scores.
     */
    [[nodiscard]] const PG_BeliefFormulaeMap& get_bf_scores() const;

    /**
     * \brief Prints the PlanningGraph to std::cout.
     * \param os The output stream to print to.
     */
    void print(std::ostream &os) const;

private:
    /// \name Internal Data Members
    ///@{

    /** \brief The list of StateLevel objects representing the state levels of the PlanningGraph. */
    std::vector<StateLevel> m_state_levels;

    /** \brief The list of ActionLevel objects representing the action levels of the PlanningGraph. */
    std::vector<ActionLevel> m_action_levels;

    /** \brief The length of the PlanningGraph (used after the goal is reached). */
    unsigned short m_pg_length = 0;

    /** \brief The sum of the depths of the levels that contain a subgoal. */
    unsigned short m_pg_sum = 0;

    /** \brief True if the planning BisGraph can find a solution. */
    bool m_satisfiable = false;

    /** \brief The list of the subgoals (possibly enhanced by heuristics_manager). */
    FormulaeList m_goal;

    //pg_worlds_score m_worlds_score; // FOR FUTURE USE

    /** \brief A map that contains the first level of encounter (if any) of a belief formula calculated by list_bf_classical. */
    pg_bfs_score m_bfs_score;

    /** \brief The set of actions never executed in the PlanningGraph for optimization. */
    ActionsSet m_never_executed;

    /** \brief The set of belief formulas that are false in the PlanningGraph. */
    FormulaeSet m_belief_formula_false;

    /**
     * \brief Sets the satisfiable field.
     * \param[in] sat The value to assign to m_satisfiable.
     */
    void set_satisfiable(bool sat);

    /**
     * \brief Builds the PlanningGraph layer by layer until the goal is found or the PlanningGraph is saturated.
     */
    void pg_build();

    /**
     * \brief Adds the next (depth + 1) State layer to m_state_levels.
     * \param[in] s_level The level to add to m_state_levels.
     */
    void add_state_level(const StateLevel& s_level);

    /**
     * \brief Adds the next (depth + 1) action layer to m_action_levels.
     * \param[in] a_level The level to add to m_action_levels.
     */
    void add_action_level(const ActionLevel& a_level);

    ///@}
};
