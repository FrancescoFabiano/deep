/**
 * \class KripkeState
 * \brief Class that represents a Kripke Structure for epistemic planning.
 *
 * \details  A Kripke Structure is the standard way of representing e-States in
 * Epistemic Planning. See KripkeWorld and KripkeStorage for related structures.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano.
 * \date May 17, 2025
 */
#pragma once

#include "KripkeWorld.h"
#include "actions/Action.h"
#include "bisimulation/Bisimulation.h"
#include "utilities/Define.h"

class KripkeState {
public:
  // --- Constructors/Destructor ---
  KripkeState() = default;

  /**
   * \brief Copy constructor.
   * \param other The KripkeState to copy from.
   */
  KripkeState(const KripkeState &other);

  ~KripkeState() = default;

  // --- Setters ---
  /** \brief Set the set of worlds for this KripkeState.
   *  \param[in] to_set The set of KripkeWorld pointers to assign.
   */
  void set_worlds(const KripkeWorldPointersSet &to_set);

  /** \brief Set the pointed world for this KripkeState.
   *  \param[in] to_set The KripkeWorld pointer to assign as pointed.
   */
  void set_pointed(const KripkeWorldPointer &to_set);

  /** \brief Set the beliefs map for this KripkeState.
   *  \param[in] to_set The beliefs map to assign.
   */
  void set_beliefs(const KripkeWorldPointersTransitiveMap &to_set);

  /** \brief Clears the beliefs map for this KripkeState.
   * This is only usable by Bisimulation.
   */
  void clear_beliefs();

  /** \brief Set the maximum depth for this KripkeState.
   *  \param[in] to_set The unsigned int value to assign as max depth.
   */
  void set_max_depth(unsigned int to_set) noexcept;

  // --- Getters ---
  /** \brief Get the set of worlds in this KripkeState.
   *  \return The set of KripkeWorld pointers.
   */
  [[nodiscard]] const KripkeWorldPointersSet &get_worlds() const noexcept;

  /** \brief Get the pointed world in this KripkeState.
   *  \return The pointed KripkeWorld pointer.
   */
  [[nodiscard]] const KripkeWorldPointer &get_pointed() const noexcept;

  /** \brief Get the beliefs map in this KripkeState.
   *  \return The beliefs map.
   */
  [[nodiscard]] const KripkeWorldPointersTransitiveMap &
  get_beliefs() const noexcept;

  /** \brief Get the maximum depth of this KripkeState.
   *  \return The max depth value.
   */
  [[nodiscard]] unsigned int get_max_depth() const noexcept;

  /** \brief Compute the successor state after applying an action.
   *  \param[in] act The action to apply.
   *  \return The resulting KripkeState.
   */
  [[nodiscard]] KripkeState compute_successor(const Action &act) const;

  // --- Operators ---
  /** \brief Copy Assignment operator.*/
  KripkeState &operator=(const KripkeState &to_copy);

  /** \brief Less-than operator for set operations.
   *  \param[in] to_compare The KripkeState to compare.
   *  \return True if this is less than \ref to_compare, false otherwise.
   */
  [[nodiscard]] bool operator<(const KripkeState &to_compare) const;

  /// \name Needed for State<T>
  ///@{
  /** \brief Build the initial Kripke structure (choose method internally).
   */
  void build_initial();

  /** \brief Function that checks if *this* entails a \ref fluent.
   *
   *
   * @param to_check: the \ref fluent to check if is entailed by *this*.
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.
   */
  [[nodiscard]] bool entails(const Fluent &to_check) const;

  /** \brief Function that checks if *this* entails a conjunctive set of \ref
   * fluent.
   *
   *
   * @param to_check: the conjunctive set of \ref fluent to check if is entailed
   * by *this*.
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.*/
  [[nodiscard]] bool entails(const FluentsSet &to_check) const;

  /** \brief Function that checks if *this* entails a DNF \ref FluentFormula.
   *
   *
   * @param to_check: the DNF \ref FluentFormula to check if is entailed by
   * *this*.
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.*/
  [[nodiscard]] bool entails(const FluentFormula &to_check) const;

  /** \brief Function that checks if *this* entails a \ref BeliefFormula.
   *
   *
   * @param to_check: the \ref BeliefFormula to check if is entailed by *this*.
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.*/
  [[nodiscard]] bool entails(const BeliefFormula &to_check) const;

  /** \brief Function that checks if *this* entails a CNF \ref FormulaeList.
   *
   *
   *
   * @param to_check: the CNF \ref FormulaeList to check if is entailed by
   * *this*.
   *
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.*/
  [[nodiscard]] bool entails(const FormulaeList &to_check) const;

  /** \brief Function that applies bisimulation contraction to this*/
  void contract_with_bisimulation();

  ///}
  // --- Printing ---
  /** \brief Print this KripkeState.*/
  void print() const;

  /** \brief Print this KripkeState to a dot format in the file stream ofs.
   * Params: [in] ofs — The output file stream.*/
  void print_dot_format(std::ofstream &ofs) const;

  /** \brief Function that prints the information of *this* for the generation
   * of the dataset used to train the GNN. \param ofs The output stream to print
   * to. \param use_hash Flag to indicate whether to use hash IDs or adjusted
   * IDs (needed because sometime both is set) \param is_merged Flag to indicate
   * if each dataset entry is merged <goal,state> or not.
   */
  void print_dataset_format(std::ofstream &ofs, const bool use_hash,
                            const bool is_merged) const;

private:
  // --- Data members ---
  /** \brief Used to differentiate partial KripkeStates generated by partial
   * observation. */
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
  void add_world(const KripkeWorld &to_add);

  /** \brief Add a belief edge for an agent between two worlds.
   *  \param[in] from The source world.
   *  \param[in] to The target world.
   *  \param[in] ag The agent.
   */
  void add_edge(const KripkeWorldPointer &from, const KripkeWorldPointer &to,
                const Agent &ag);

  /** \brief Add a world with repetition tracking.
   *  \param[in] to_add The KripkeWorld to add.
   *  \return Pointer to the newly inserted KripkeWorld.
   */
  KripkeWorldPointer add_rep_world(const KripkeWorld &to_add);

  /** \brief Add a world with old repetition tracking.
   *  \param[in] to_add The KripkeWorld to add.
   *  \param[in] old_repetition Used to distinguish from same level but
   * different origins. \return Pointer to the newly inserted KripkeWorld.
   */
  KripkeWorldPointer add_rep_world(const KripkeWorld &to_add,
                                   unsigned short old_repetition);

  /** \brief Add a world with repetition and newness tracking.
   *  \param[in] to_add The KripkeWorld to add.
   *  \param[in] repetition Used to distinguish from other levels.
   *  \param[out] is_new Indicates if the world was already present.
   *  \return Pointer to the newly inserted KripkeWorld.
   */
  KripkeWorldPointer add_rep_world(const KripkeWorld &to_add,
                                   unsigned short repetition, bool &is_new);

  // --- Structure Building ---

  /** \brief Generate all possible permutations of the domain's fluents.
   *  \param[out] permutation The permutation in construction.
   *  \param[in] index The index of the fluent to add.
   *  \param[in] initially_known The set of initially known fluents.
   */
  void generate_initial_worlds(FluentsSet &permutation, unsigned int index,
                               const FluentsSet &initially_known);

  /** \brief Check if a KripkeWorld respects initial conditions and add it if
   * so. \param[in] possible_add The KripkeWorld to check.
   */
  void add_initial_world(const KripkeWorld &possible_add);

  /** \brief Generate all initial edges for the KripkeState. */
  void generate_initial_edges();

  /** \brief Remove an edge for an agent between two worlds.
   *  \param[in] from The KripkeWorld pointer to remove the edge from.
   *  \param[in] to The KripkeWorld to remove.
   *  \param[in] ag The agent.
   */
  void remove_edge(const KripkeWorldPointer &from, const KripkeWorldPointer &to,
                   const Agent &ag);

  /** \brief Remove initial edges based on known fluent formula for an agent.
   *  \param[in] known_ff The fluent formula known by the agent.
   *  \param[in] ag The agent.
   */
  void remove_initial_edge(const FluentFormula &known_ff, const Agent &ag);

  /** \brief Remove initial edges based on a BeliefFormula.
   *  \param[in] to_check The BeliefFormula to check.
   */
  void remove_initial_edge_bf(const BeliefFormula &to_check);

  /** \brief Add beliefs to a world.
   *  \param[in] world The KripkeWorld pointer.
   *  \param[in] beliefs The beliefs map.
   */
  void add_world_beliefs(const KripkeWorldPointer &world,
                         const KripkeWorldPointersMap &beliefs);

  /** \brief Copy worlds and beliefs of oblivious agents to another KripkeState.
   *  \param[in] ret The new KripkeState.
   *  \param[in] oblivious_obs_agents The set of oblivious agents.
   */
  void maintain_oblivious_believed_worlds(
      KripkeState &ret, const AgentsSet &oblivious_obs_agents) const;

  // --- Transition/Execution ---
  /** \brief Recursively compute the result of an ontic action.
   *  \param[in] act The ontic action to apply.
   *  \param[in] ret The resulting KripkeState.
   *  \param[in] current_pw The world being currently calculated.
   *  \param[in] calculated Map tracking results of the transition function.
   *  \param[in] oblivious_obs_agents The set of oblivious agents.
   *  \return The resulting KripkeWorld pointer.
   */
  KripkeWorldPointer execute_ontic_helper(
      const Action &act, KripkeState &ret, const KripkeWorldPointer &current_pw,
      TransitionMap &calculated, AgentsSet &oblivious_obs_agents) const;

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
  KripkeWorldPointer execute_sensing_announcement_helper(
      const FluentFormula &effects, KripkeState &ret,
      const KripkeWorldPointer &current_pw, TransitionMap &calculated,
      AgentsSet &partially_obs_agents, AgentsSet &oblivious_obs_agents,
      bool previous_entailment) const;

  /** \brief Apply an ontic action to this KripkeState.
   *  \param[in] act The ontic action to apply.
   *  \return The resulting KripkeState.
   */
  [[nodiscard]] KripkeState execute_ontic(const Action &act) const;

  /** \brief Apply a sensing action to this KripkeState.
   *  \param[in] act The sensing action to apply.
   *  \return The resulting KripkeState.
   */
  [[nodiscard]] KripkeState execute_sensing(const Action &act) const;

  /** \brief Apply an announcement action to this KripkeState.
   *  \param[in] act The announcement action to apply.
   *  \return The resulting KripkeState.
   */
  [[nodiscard]] KripkeState execute_announcement(const Action &act) const;

  /*This is to allow bisimulation to reduce the size of the object*/
  friend class Bisimulation;
};
