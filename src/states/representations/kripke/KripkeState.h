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


    [[nodiscard]] const KripkeWorldPointersVec &
get_designated_worlds_vec() const noexcept {
        return m_designated_worlds_vec;
    }


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

  void set_designated_worlds_vec();


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



    [[nodiscard]]
bool is_executable(const Action &action) const;

private:
  // --- Data members ---
  /** \brief Set of pointers to each world in the structure. */
  KripkeWorldPointersSet m_worlds;
  /** \brief Set of designated worlds. */
  KripkeWorldPointersSet m_designated_worlds;
  /** \brief Beliefs of each agent in every world. */
  KripkeWorldPointersTransitiveMap m_beliefs;


    KripkeWorldPointersVec m_designated_worlds_vec;

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

  /** \brief Add a world with old repetition tracking.
   *  \param[in] to_add The KripkeWorld to add.
   *  \param[in] repetition Used to distinguish from same level but
   * different origins. \return Pointer to the newly inserted KripkeWorld.*/
    KripkeWorldPointer add_rep_world(
        const KripkeWorld &to_add,
        unsigned short repetition);



    void recompute_hash();


    // === DEL Product Update Helpers ===
    using ProductWorld =
        std::pair<KripkeWorldPointer, EventId>;

    using ProductWorldMap =
        std::map<ProductWorld, KripkeWorldPointer>;

    using ProductWorldQueue =
        std::queue<ProductWorld>;

    using ResolvedObservability =
        std::map<Agent, ObservabilityType>;

    using ApplicabilityCache =
    std::map<ProductWorld, bool>;

    [[nodiscard]]
    bool is_event_applicable(
        const Event &event,
        const KripkeWorldPointer &world) const;

    [[nodiscard]]
bool is_event_applicable_cached(
    const Event &event,
    const KripkeWorldPointer &world,
    ApplicabilityCache &cache) const;

    [[nodiscard]]
    FluentsSet apply_event_postconditions(
        const Event &event,
        const KripkeWorldPointer &world) const;

    [[nodiscard]]
    ResolvedObservability resolve_observability_types(
        const Action &action) const;

    KripkeWorldPointer get_or_create_product_world(
        const KripkeWorldPointer &source_world,
        const Event &event,
        KripkeState &successor,
        ProductWorldMap &product_worlds,
        ProductWorldQueue &pending,
        unsigned short &next_repetition) const;

    void create_designated_product_worlds(
        const Action &action,
        KripkeState &successor,
        ProductWorldMap &product_worlds,
        ProductWorldQueue &pending,
        ApplicabilityCache &applicability_cache,
        unsigned short &next_repetition) const;

    void expand_product_relations(
        const Action &action,
        const ResolvedObservability &observability,
        KripkeState &successor,
        ProductWorldMap &product_worlds,
        ProductWorldQueue &pending,
        ApplicabilityCache &applicability_cache,
        unsigned short &next_repetition) const;



  /* This is to allow bisimulation to reduce the size of the object*/
  friend class Bisimulation;
};
