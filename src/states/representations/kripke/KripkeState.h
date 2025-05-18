/**
 * \class KripkeState
 * \brief Class that represents a Kripke Structure for epistemic planning.
 *
 * \details  A Kripke Structure is the standard way of representing e-States in Epistemic Planning.
 *          See KripkeWorld and KripkeStorage for related structures.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano.
 * \date May 17, 2025
 */
#pragma once

#include "KripkeWorld.h"
#include "utilities/Define.h"
#include "actions/Action.h"
#include "../lib/bisimulation/bisimulation.h"

class KripkeState
{
public:
    // --- Constructors/Destructor ---
    KripkeState() = default;
    ~KripkeState() = default;
    // --- Setters ---
    /** \brief Set the set of worlds for this KripkeState.
     *  \param[in] to_set The set of KripkeWorld pointers to assign.
     */
    void set_worlds(const KripkeWorldPointersSet& to_set);
    /** \brief Set the pointed world for this KripkeState.
     *  \param[in] to_set The KripkeWorld pointer to assign as pointed.
     */
    void set_pointed(const KripkeWorldPointer& to_set);
    /** \brief Set the beliefs map for this KripkeState.
     *  \param[in] to_set The beliefs map to assign.
     */
    void set_beliefs(const KripkeWorldPointersTransitiveMap& to_set);
    /** \brief Set the maximum depth for this KripkeState.
     *  \param[in] to_set The unsigned int value to assign as max depth.
     */
    void set_max_depth(unsigned int to_set) noexcept;

    // --- Getters ---
    /** \brief Get the set of worlds in this KripkeState.
     *  \return The set of KripkeWorld pointers.
     */
    [[nodiscard]] const KripkeWorldPointersSet& get_worlds() const noexcept;
    /** \brief Get the pointed world in this KripkeState.
     *  \return The pointed KripkeWorld pointer.
     */
    [[nodiscard]] const KripkeWorldPointer& get_pointed() const noexcept;
    /** \brief Get the beliefs map in this KripkeState.
     *  \return The beliefs map.
     */
    [[nodiscard]] const KripkeWorldPointersTransitiveMap& get_beliefs() const noexcept;
    /** \brief Get the maximum depth of this KripkeState.
     *  \return The max depth value.
     */
    [[nodiscard]] unsigned int get_max_depth() const noexcept;

    // --- Entailment ---
    /** \brief Check if a fluent is entailed in the pointed world.
     *  \param[in] to_check The fluent to check.
     *  \return True if entailed, false otherwise.
     */
    [[nodiscard]] bool entails(const Fluent &to_check) const;
    /** \brief Check if a set of fluents is entailed in the pointed world.
     *  \param[in] to_check The set of fluents to check.
     *  \return True if entailed, false otherwise.
     */
    [[nodiscard]] bool entails(const FluentsSet& to_check) const;
    /** \brief Check if a fluent formula is entailed in the pointed world.
     *  \param[in] to_check The fluent formula to check.
     *  \return True if entailed, false otherwise.
     */
    [[nodiscard]] bool entails(const FluentFormula& to_check) const;
    /** \brief Check if a BeliefFormula is entailed in the pointed world.
     *  \param[in] to_check The BeliefFormula to check.
     *  \return True if entailed, false otherwise.
     */
    [[nodiscard]] bool entails(const BeliefFormula& to_check) const;

    // --- Structure Building ---
    /** \brief Build the initial Kripke structure (choose method internally). */
    void build_initial();
    /** \brief Build the initial Kripke structure using pruning. */
    void build_initial_prune();
    /** \brief Generate all possible permutations of the domain's fluents.
     *  \param[out] permutation The permutation in construction.
     *  \param[in] index The index of the fluent to add.
     *  \param[in] initially_known The set of initially known fluents.
     */
    void generate_initial_pworlds(FluentsSet& permutation, int index, const FluentsSet& initially_known);
    /** \brief Check if a KripkeWorld respects initial conditions and add it if so.
     *  \param[in] possible_add The KripkeWorld to check.
     */
    void add_initial_pworld(const KripkeWorld& possible_add);
    /** \brief Generate all initial edges for the KripkeState. */
    void generate_initial_pedges();
    /** \brief Remove an edge for an agent between two worlds.
     *  \param[in] from The KripkeWorld pointer to remove the edge from.
     *  \param[in] to The KripkeWorld to remove.
     *  \param[in] ag The agent.
     */
    void remove_edge(KripkeWorldPointer& from, const KripkeWorld& to, Agent ag);
    /** \brief Remove initial edges based on known fluent formula for an agent.
     *  \param[in] known_ff The fluent formula known by the agent.
     *  \param[in] ag The agent.
     */
    void remove_initial_pedge(const FluentFormula& known_ff, Agent ag);
    /** \brief Remove initial edges based on a BeliefFormula.
     *  \param[in] to_check The BeliefFormula to check.
     */
    void remove_initial_pedge_bf(const BeliefFormula& to_check);
    /** \brief Add beliefs to a world.
     *  \param[in] world The KripkeWorld pointer.
     *  \param[in] beliefs The beliefs map.
     */
    void add_pworld_beliefs(const KripkeWorldPointer& world, const KripkeWorldPointersMap& beliefs);
    /** \brief Copy worlds and beliefs of oblivious agents to another KripkeState.
     *  \param[in] ret The new KripkeState.
     *  \param[in] oblivious_obs_agents The set of oblivious agents.
     */
    void maintain_oblivious_believed_pworlds(KripkeState& ret, const AgentsSet& oblivious_obs_agents) const;

    // --- Transition/Execution ---
    /** \brief Recursively compute the result of an ontic action.
     *  \param[in] act The ontic action to apply.
     *  \param[in] ret The resulting KripkeState.
     *  \param[in] current_pw The world being currently calculated.
     *  \param[in] calculated Map tracking results of the transition function.
     *  \param[in] oblivious_obs_agents The set of oblivious agents.
     *  \return The resulting KripkeWorld pointer.
     */
    KripkeWorldPointer execute_ontic_helper(const Action& act, KripkeState& ret, const KripkeWorldPointer& current_pw, TransitionMap& calculated, AgentsSet& oblivious_obs_agents) const;
    /** \brief Recursively compute the result of a sensing/announcement action.
     *  \param[in] effects The effects of the action.
     *  \param[in] ret The resulting KripkeState.
     *  \param[in] current_pw The world being currently calculated.
     *  \param[in] calculated Map tracking results of the transition function.
     *  \param[in] partially_obs_agents The set of partially observant agents.
     *  \param[in] oblivious_obs_agents The set of oblivious agents.
     *  \param[in] previous_entailment The value of the coming state entailment.
     *  \return The resulting KripkeWorld pointer.
     */
    KripkeWorldPointer execute_sensing_announcement_helper(const FluentFormula& effects, KripkeState& ret, const KripkeWorldPointer& current_pw, TransitionMap& calculated, AgentsSet& partially_obs_agents, AgentsSet& oblivious_obs_agents, bool previous_entailment) const;
    /** \brief Apply an ontic action to this KripkeState.
     *  \param[in] act The ontic action to apply.
     *  \return The resulting KripkeState.
     */
    KripkeState execute_ontic(const Action& act) const;
    /** \brief Apply a sensing action to this KripkeState.
     *  \param[in] act The sensing action to apply.
     *  \return The resulting KripkeState.
     */
    KripkeState execute_sensing(const Action& act) const;
    /** \brief Apply an announcement action to this KripkeState.
     *  \param[in] act The announcement action to apply.
     *  \return The resulting KripkeState.
     */
    KripkeState execute_announcement(const Action& act) const;

    /** \brief Compute the successor state after applying an action.
     *  \param[in] act The action to apply.
     *  \return The resulting KripkeState.
     */
    KripkeState compute_succ(const Action& act) const;

    /** \brief Check epistemic properties after an action execution.
     *  \param[in] fully The set of fully observant agents.
     *  \param[in] partially The set of partially observant agents.
     *  \param[in] effects The effects of the action.
     *  \param[in] updated The updated KripkeState.
     *  \return True if all properties are respected, false otherwise.
     */
    [[nodiscard]] bool check_properties(const AgentsSet& fully, const AgentsSet& partially, const FluentFormula& effects, const KripkeState& updated) const;

    /** \brief Minimize this KripkeState to the minimum bisimilar structure. */
    void calc_min_bisimilar();

    // --- Operators ---
    /** \brief Copy Assignment operator.*/
    KripkeState & operator=(const KripkeState& to_copy);
    /** \brief Less-than operator for set operations.
     *  \param[in] to_compare The KripkeState to compare.
     *  \return True if this is less than to_compare, false otherwise.
     */
    [[nodiscard]] bool operator<(const KripkeState& to_compare) const;

    // --- Printing ---
    /** \brief Print this KripkeState to std::cout. */
    void print() const;
    /** \brief Print this KripkeState in ML dataset format.
     *  \param[in] graphviz The output stream.
     */
    void print_ML_dataset(std::ostream& graphviz) const;
    /** \brief Print this KripkeState in Graphviz format.
     *  \param[in] graphviz The output stream.
     */
    void print_graphviz(std::ostream& graphviz) const;

private:
    // --- Data members ---
    /** \brief Used to differentiate partial KripkeStates generated by partial observation. */
    unsigned int m_max_depth = 0;
    /** \brief Set of pointers to each world in the structure. */
    KripkeWorldPointersSet m_worlds;
    /** \brief Pointer to the pointed world. */
    KripkeWorldPointer m_pointed;
    /** \brief Beliefs of each agent in every world. */
    KripkeWorldPointersTransitiveMap m_beliefs;

    // --- Internal helpers ---
    /** \brief Add a world to the Kripke structure.
     *  \param[in] to_add The KripkeWorld to add.
     */
    void add_world(const KripkeWorld& to_add);
    /** \brief Add a belief edge for an agent between two worlds.
     *  \param[in] from The source world.
     *  \param[in] to The target world.
     *  \param[in] ag The agent.
     */
    void add_edge(const KripkeWorldPointer& from, const KripkeWorldPointer& to, Agent ag);

    /** \brief Add a world with repetition tracking.
     *  \param[in] to_add The KripkeWorld to add.
     *  \return Pointer to the newly inserted KripkeWorld.
     */
    KripkeWorldPointer add_rep_world(const KripkeWorld& to_add);
    /** \brief Add a world with old repetition tracking.
     *  \param[in] to_add The KripkeWorld to add.
     *  \param[in] old_repetition Used to distinguish from same level but different origins.
     *  \return Pointer to the newly inserted KripkeWorld.
     */
    KripkeWorldPointer add_rep_world(const KripkeWorld& to_add, unsigned short old_repetition);
    /** \brief Add a world with repetition and newness tracking.
     *  \param[in] to_add The KripkeWorld to add.
     *  \param[in] repetition Used to distinguish from other levels.
     *  \param[out] is_new Indicates if the world was already present.
     *  \return Pointer to the newly inserted KripkeWorld.
     */
    KripkeWorldPointer add_rep_world(const KripkeWorld& to_add, unsigned short repetition, bool& is_new);

    /** \brief Get all reachable worlds and edges from a given world.
     *  \param[in] pw The starting world.
     *  \param[out] reached_worlds The set of reached worlds.
     *  \param[out] reached_edges The map of reached edges.
     */
    void get_all_reachable_worlds(const KripkeWorldPointer& pw, KripkeWorldPointersSet& reached_worlds, KripkeWorldPointersTransitiveMap& reached_edges) const;
    /** \brief Remove unreachable worlds from the structure. */
    void clean_unreachable_pworlds();

    /** \brief Convert this KripkeState to an automaton.
     *  \param[out] pworld_vec The vector of KripkeWorld pointers.
     *  \param[in] agent_to_label The map from agent to bisimulation label.
     *  \return The automaton representation.
     */
    [[nodiscard]] automa pstate_to_automaton(std::vector<KripkeWorldPointer>& pworld_vec, const std::map<Agent, bis_label>& agent_to_label) const;
    /** \brief Convert an automaton to this KripkeState.
     *  \param[in] a The automaton.
     *  \param[in] pworld_vec The vector of KripkeWorld pointers.
     *  \param[in] label_to_agent The map from bisimulation label to agent.
     */
    void automaton_to_pstate(const automa& a, const std::vector<KripkeWorldPointer>& pworld_vec, const std::map<bis_label, Agent>& label_to_agent);
};
