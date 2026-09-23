/**
 * \class KripkeState
 * \brief Kripke-structure representation of an epistemic search state.
 *
 * \details A Kripke state stores:
 *   - the set of worlds currently present in the epistemic model;
 *   - the designated worlds that represent the current uncertainty set;
 *   - one accessibility relation per agent over those worlds.
 *
 * Successor generation follows DEL product update against a grounded
 * multi-pointed action model. Designated successor worlds are seeded from
 * designated `(world, event)` pairs, then the reachable product structure is
 * expanded according to the event relation selected for each agent by the
 * action's observability conditions.
 *
 * See \ref KripkeWorld and \ref KripkeStorage for the related world-level
 * structures.
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
  void set_designated_worlds(const KripkeWorldPointersSet &to_set);

  /** \brief Add a designated world to this KripkeState.
   *  \param[in] to_add The KripkeWorld pointer to designate.
   */
  void add_designated_world(const KripkeWorldPointer &to_add);

  /** \brief Set the beliefs map for this KripkeState.
   *  \param[in] to_set The beliefs map to assign.
   */
  void set_beliefs(const KripkeWorldPointersTransitiveMap &to_set);

  /** \brief Clears the accessibility relation of this Kripke state.
   * This is only used internally by bisimulation reduction.
   */
  void clear_beliefs();

  /** \brief Get the set of worlds in this KripkeState.
   *  \return The set of KripkeWorld pointers.
   */
  [[nodiscard]] const KripkeWorldPointersSet &get_worlds() const noexcept;

  /** \brief Get the designated worlds in this KripkeState.
   *  \return The set of designated KripkeWorld pointers.
   */
  [[nodiscard]] const KripkeWorldPointersSet &
  get_designated_worlds() const noexcept;

  /**
   * \brief Get the cached structural hash of this state.
   * \return The current hash value.
   */
  [[nodiscard]] uint64_t get_hash() const noexcept;

  /** \brief Check whether a world is designated. */
  [[nodiscard]] bool
  is_designated(const KripkeWorldPointer &world) const noexcept;

  /** \brief Get the agent accessibility relation in this KripkeState.
   *  \return The map world -> agent -> reachable worlds.
   */
  [[nodiscard]] const KripkeWorldPointersTransitiveMap &
  get_beliefs() const noexcept;

  /** \brief Compute the DEL product-update successor of this state.
   *
   * The action's designated events determine which product worlds become
   * designated, and each agent's observability type is resolved once on the
   * source state before expanding the reachable product model.
   *
   *  \param[in] action The grounded action model to apply.
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

  /** \brief Equality operator.
   *  \param[in] to_compare The KripkeState to compare.
   *  \return True if both states are considered equal, false otherwise.
   */
  bool operator==(const KripkeState &to_compare) const;

  /// \name Needed for State<T>
  ///@{
  /** \brief Build the initial Kripke structure imported from the grounded task.
   */
  void build_initial();

  /** \brief Check whether all designated worlds entail a fluent.
   *
   * @param to_check: the Fluent to check if is entailed by *this*.
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.
   */
  [[nodiscard]] bool entails(const Fluent &to_check) const;

  /** \brief Check whether all designated worlds entail a conjunctive fluent
   * set.
   *
   * @param to_check: the conjunctive set of \ref Fluent to check if is entailed
   * by *this*.
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.*/
  [[nodiscard]] bool entails(const FluentsSet &to_check) const;

  /** \brief Check whether all designated worlds entail a fluent formula.
   *
   * @param to_check: the DNF \ref FluentFormula to check if is entailed by
   * *this*.
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.*/
  [[nodiscard]] bool entails(const FluentFormula &to_check) const;

  /** \brief Check whether this epistemic state entails a belief formula.
   *
   * @param to_check: the \ref BeliefFormula to check if is entailed by *this*.
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.*/
  [[nodiscard]] bool entails(const BeliefFormula &to_check) const;

  /** \brief Check whether this epistemic state entails a conjunction of belief
   * formulas.
   *
   * @param to_check: the CNF \ref FormulaeList to check if is entailed by
   * *this*.
   *
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.*/
  [[nodiscard]] bool entails(const FormulaeList &to_check) const;

  /** \brief Apply bisimulation contraction after pruning unreachable worlds. */
  void contract_with_bisimulation();

  /** \brief Compute or retrieve the graph-tensor view used by learned
   * heuristics. \return The cached tensor representation of this state.
   */
  [[nodiscard]] const GraphTensor &get_tensor_representation();

  ///}
  // --- Printing ---
  /** \brief Print this KripkeState.*/
  void print() const;

  /** \brief Print this KripkeState in DOT format.
   * \param ofs The output file stream.
   */
  void print_dot_format(std::ofstream &ofs) const;

  /** \brief Print this state in the dataset format used for GNN/RL training.
   * \param ofs The output stream to print to.
   */
  void print_dataset_format(std::ofstream &ofs) const;

  /**
   * \brief Check whether a grounded action is executable in this state.
   *
   * An action is executable iff every designated world satisfies the
   * precondition of at least one designated event of the action model.
   *
   * \param action The grounded action model to test.
   * \return True if the action is executable, false otherwise.
   */
  [[nodiscard]] bool is_executable(const Action &action) const;

private:
  // --- Data members ---
  /** \brief Set of pointers to each world in the structure. */
  KripkeWorldPointersSet m_worlds;
  /** \brief Set of designated worlds. */
  KripkeWorldPointersSet m_designated_worlds;
  /** \brief Accessibility relation: source world -> agent -> reachable worlds.
   */
  KripkeWorldPointersTransitiveMap m_beliefs;

  /** \brief Cached structural hash used by fast state-comparison modes. */
  uint64_t m_hash = 0;

  /** \brief Tensor version of this for the various NN-based heuristics. */
  GraphTensor m_tensor_representation;
  /** \brief True once \ref m_tensor_representation has been computed. */
  bool m_computed_tensor_representation = false;

  // --- Internal helpers ---
  /** \brief Add a world to the Kripke structure.
   *  \param[in] to_add The KripkeWorld to add.
   */
  void add_world(const KripkeWorld &to_add);
  /** \brief Move-based overload of \ref add_world(const KripkeWorld &). */
  void add_world(KripkeWorld &&to_add);

  /** \brief Add a belief edge for an agent between two worlds.
   *  \param[in] from The source world.
   *  \param[in] to The target world.
   *  \param[in] ag The agent.
   */
  void add_edge(const KripkeWorldPointer &from, const KripkeWorldPointer &to,
                const Agent &ag);

  /** \brief Add a world together with its product-update repetition tag.
   *  \param[in] to_add The KripkeWorld to add.
   *  \param[in] repetition Used to distinguish equal valuations created from
   * different product-world origins.
   *  \return Pointer to the newly inserted KripkeWorld.
   */
  KripkeWorldPointer add_rep_world(const KripkeWorld &to_add,
                                   unsigned short repetition);

  /** \brief Move-based overload of \ref add_rep_world(const KripkeWorld &,
   * unsigned short). */
  KripkeWorldPointer add_rep_world(KripkeWorld &&to_add,
                                   unsigned short repetition);

  /** \brief Recompute the cached structural hash after the state changes. */
  void recompute_hash();

  // === DEL Product Update Helpers ===
  /// \brief Product-world identifier `(source world, event id)`.
  using ProductWorld = std::pair<KripkeWorldPointer, EventId>;

  /// \brief Mapping from product-world identifier to created successor world.
  using ProductWorldMap = std::map<ProductWorld, KripkeWorldPointer>;

  /// \brief Work queue of product worlds whose outgoing relations remain to
  /// expand.
  using ProductWorldQueue = std::queue<ProductWorld>;

  /// \brief Cache of event applicability for `(world,event)` pairs.
  using ApplicabilityCache = std::map<ProductWorld, bool>;

  /// \brief Observability type selected for each agent on the source state.
  using ResolvedObservability = std::map<Agent, ObservabilityType>;

  /** \brief Check whether one event precondition holds in one source world. */
  [[nodiscard]]
  bool is_event_applicable(const Event &event,
                           const KripkeWorldPointer &world) const;

  /**
   * \brief Cached version of \ref is_event_applicable.
   * \param event The event to test.
   * \param world The source world where the precondition is evaluated.
   * \param cache Applicability cache shared during successor construction.
   * \return True if the event is applicable in the world.
   */
  [[nodiscard]] bool
  is_event_applicable_cached(const Event &event,
                             const KripkeWorldPointer &world,
                             ApplicabilityCache &cache) const;

  /**
   * \brief Apply one event's postconditions to one source world valuation.
   * \param event The event whose postconditions are evaluated.
   * \param world The source world providing the valuation and context.
   * \return The valuation assigned to the created product world.
   */
  [[nodiscard]]
  FluentsSet apply_event_postconditions(const Event &event,
                                        const KripkeWorldPointer &world) const;

  /**
   * \brief Resolve one observability type per agent on the source state.
   * \param action The action whose observability cases are evaluated.
   * \return Mapping from each agent to the selected observability type.
   */
  [[nodiscard]]
  ResolvedObservability resolve_observability_types(const Action &action) const;

  /**
   * \brief Return the successor world for one product pair, creating it if
   * needed. \param source_world Source world from the predecessor state. \param
   * event Event applied at \p source_world. \param successor Successor state
   * under construction. \param product_worlds Cache of already-created product
   * worlds. \param pending Queue of product worlds still to expand. \param
   * next_repetition Repetition counter for duplicate valuations. \return The
   * created or reused successor-world pointer.
   */
  KripkeWorldPointer get_or_create_product_world(
      const KripkeWorldPointer &source_world, const Event &event,
      KripkeState &successor, ProductWorldMap &product_worlds,
      ProductWorldQueue &pending, unsigned short &next_repetition) const;

  /**
   * \brief Seed the designated part of the reachable product model.
   * \param action Action being applied.
   * \param successor Successor state under construction.
   * \param product_worlds Cache of already-created product worlds.
   * \param pending Queue of product worlds still to expand.
   * \param applicability_cache Cache of `(world,event)` executability tests.
   * \param next_repetition Repetition counter for duplicate valuations.
   */
  void create_designated_product_worlds(const Action &action,
                                        KripkeState &successor,
                                        ProductWorldMap &product_worlds,
                                        ProductWorldQueue &pending,
                                        ApplicabilityCache &applicability_cache,
                                        unsigned short &next_repetition) const;

  /**
   * \brief Expand the reachable product relations induced by world and event
   * accessibility. \param action Action being applied. \param observability
   * Observability type selected for each agent. \param successor Successor
   * state under construction. \param product_worlds Cache of already-created
   * product worlds. \param pending Queue of product worlds still to expand.
   * \param applicability_cache Cache of `(world,event)` executability tests.
   * \param next_repetition Repetition counter for duplicate valuations.
   */
  void expand_product_relations(const Action &action,
                                const ResolvedObservability &observability,
                                KripkeState &successor,
                                ProductWorldMap &product_worlds,
                                ProductWorldQueue &pending,
                                ApplicabilityCache &applicability_cache,
                                unsigned short &next_repetition) const;

  /* This is to allow bisimulation to reduce the size of the object*/
  friend class Bisimulation;
};
