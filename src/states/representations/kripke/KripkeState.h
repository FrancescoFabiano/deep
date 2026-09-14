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
#include "neuralnets/GraphTensor.h"
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

  /** \brief Set the designated worlds for this KripkeState.
   *  \param[in] to_set The set of designated KripkeWorld pointers.
   */
  void set_designated_worlds(
      const KripkeWorldPointersSet &to_set);

  /** \brief Add a designated world to this KripkeState.
   *  \param[in] to_add The KripkeWorld pointer to designate.
   */
  void add_designated_world(
      const KripkeWorldPointer &to_add);

  /** \brief Set the beliefs map for this KripkeState.
   *  \param[in] to_set The beliefs map to assign.
   */
  void set_beliefs(const KripkeWorldPointersTransitiveMap &to_set);

  /** \brief Clears the beliefs map for this KripkeState.
   * This is only usable by Bisimulation.
   */
  void clear_beliefs();


  // --- Getters ---
  /** \brief Get the set of worlds in this KripkeState.
   *  \return The set of KripkeWorld pointers.
   */
  [[nodiscard]] const KripkeWorldPointersSet &get_worlds() const noexcept;

  /** \brief Get the vector of worlds in this KripkeState.
   *  \return The vector of KripkeWorld pointers.
   */
  [[nodiscard]] const KripkeWorldPointersVec &get_worlds_vec() const noexcept;

  /** \brief Get the designated worlds in this KripkeState.
   *  \return The set of designated KripkeWorld pointers.
   */
  [[nodiscard]] const KripkeWorldPointersSet &
  get_designated_worlds() const noexcept;

    [[nodiscard]] uint64_t get_hash() const noexcept;


  /** \brief Check whether a world is designated. */
  [[nodiscard]] bool
  is_designated(
      const KripkeWorldPointer &world) const noexcept;

  /** \brief Get the beliefs map in this KripkeState.
   *  \return The beliefs map.
   */
  [[nodiscard]] const KripkeWorldPointersTransitiveMap &
  get_beliefs() const noexcept;

  /** \brief Get the beliefs map in this KripkeState in vectorized form.
   *  \return The beliefs map.
   */
  [[nodiscard]] const KripkeWorldPointersTransitiveMapVec &
  get_beliefs_vec() const noexcept;


  /** \brief Compute the successor state after applying an action.
   *  \param[in] action The action to apply.
   *  \return The resulting KripkeState.
   */
  [[nodiscard]] KripkeState compute_successor(const Action &action) const;

  // --- Operators ---
  /** \brief Copy Assignment operator.*/
  KripkeState &operator=(const KripkeState &to_copy);

  /** \brief Less-than operator for set operations.
   *  \param[in] to_compare The KripkeState to compare.
   *  \return True if this is less than to_compare, false otherwise.
   */
  [[nodiscard]] bool operator<(const KripkeState &to_compare) const;

  /** \brief Equal operator.
   *  \param[in] to_compare The KripkeState to compare.
   *  \return True if this is less than to_compare, false otherwise.
   */
  bool operator==(const KripkeState &to_compare) const;

  /// \name Needed for State<T>
  ///@{
  /** \brief Build the initial Kripke structure (choose method internally).
   */
  void build_initial();

  /** \brief Function that checks if *this* entails a Fluent.
   *
   *
   * @param to_check: the Fluent to check if is entailed by *this*.
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.
   */
  [[nodiscard]] bool entails(const Fluent &to_check) const;

  /** \brief Function that checks if *this* entails a conjunctive set of Fluent.
   *
   *
   * @param to_check: the conjunctive set of \ref Fluent to check if is entailed
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

  [[nodiscard]] const GraphTensor &get_tensor_representation();

  ///}
  // --- Printing ---
  /** \brief Print this KripkeState.*/
  void print() const;

  /** \brief Print this KripkeState to a dot format in the file stream ofs.
   * Params: [in] ofs — The output file stream.*/
  void print_dot_format(std::ofstream &ofs) const;

  /** \brief Function that prints the information of *this* for the generation
   * of the dataset used to train the GNN. \param ofs The output stream to print
   * to.
   * if each dataset entry is merged <goal,state> or not.
   */
  void print_dataset_format(std::ofstream &ofs) const;

private:
  // --- Data members ---
  /** \brief Set of pointers to each world in the structure. */
  KripkeWorldPointersSet m_worlds;
  /** \brief Set of designated worlds. */
  KripkeWorldPointersSet m_designated_worlds;
  /** \brief Beliefs of each agent in every world. */
  KripkeWorldPointersTransitiveMap m_beliefs;

  /** \brief Set of pointers to each world in the structure -- empty otherwise.
   */
  KripkeWorldPointersVec m_worlds_vec;
  /** \brief Beliefs of each agent in every world in vector form for strong
   * equivalence check -- empty otherwise. */
  KripkeWorldPointersTransitiveMapVec m_beliefs_vec;


    uint64_t m_hash = 0;

  void set_worlds_vec();

  void set_beliefs_vec();

  /** \brief Tensor version of this for the various NN-based heuristics */
  GraphTensor m_tensor_representation;
  /** Guard to indicate whether the GraphTensor needs to be computed*/
  bool m_computed_tensor_representation = false;

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

  KripkeWorldPointer add_rep_world(const KripkeWorld &to_add);*/

  /** \brief Add a world with old repetition tracking.
   *  \param[in] to_add The KripkeWorld to add.
   *  \param[in] old_repetition Used to distinguish from same level but
   * different origins. \return Pointer to the newly inserted KripkeWorld.*/
  KripkeWorldPointer add_rep_world(const KripkeWorld &to_add,
                                   unsigned short old_repetition);

  /** \brief Add a world with repetition and newness tracking.
   *  \param[in] to_add The KripkeWorld to add.
   *  \param[in] repetition Used to distinguish from other levels.
   *  \param[out] is_new Indicates if the world was already present.
   *  \return Pointer to the newly inserted KripkeWorld.

  KripkeWorldPointer add_rep_world(const KripkeWorld &to_add,
                                   unsigned short repetition, bool &is_new); */

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


    void recompute_hash();


  /* This is to allow bisimulation to reduce the size of the object*/
  friend class Bisimulation;
};
