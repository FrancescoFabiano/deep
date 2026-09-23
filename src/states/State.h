/**
 * \class State
 * \brief Generic planner state wrapper around one epistemic-state
 * representation.
 *
 * \details This template is the search-layer wrapper used by the planner. It
 * delegates semantic operations such as entailment, executability, successor
 * generation, and optional bisimulation contraction to the underlying
 * representation type \p StateRepr.
 *
 * Template and not virtual to keep the pointer and, since the type of search is
 * decided at compile-time virtual overhead is not necessary.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 20, 2025
 */
#pragma once
#include <concepts>

#include "actions/Action.h"
#include "neuralnets/GraphTensor.h"
#include "utilities/Define.h"

/**
 * @brief Concept that enforces the required interface for a state
 * representation type `StateRepr`.
 *
 * This concept defines the contract that a type `StateRepr` must fulfill to be
 * used with the `State<StateRepr>` class. It ensures that `StateRepr` provides
 * entailment checks, executability conditions, printing, and comparison.
 *
 * @tparam StateRepr The type to be checked against the required interface.
 */
template <typename StateRepr>
concept StateRepresentation =
    requires(StateRepr rep, const Fluent &f, const FluentsSet &fs,
             const FluentFormula &ff, const BeliefFormula &bf,
             const FormulaeList &fl, const Action &act, std::ofstream &ofs,
             const StateRepr &other) {
      /**
       * @name Semantic Operations
       */
      ///@{
      { std::as_const(rep).is_executable(act) }-> std::same_as<bool>;
      { std::as_const(rep).entails(f) } -> std::same_as<bool>;
      { std::as_const(rep).entails(fs) } -> std::same_as<bool>;
      { std::as_const(rep).entails(ff) } -> std::same_as<bool>;
      { std::as_const(rep).entails(bf) } -> std::same_as<bool>;
      { std::as_const(rep).entails(fl) } -> std::same_as<bool>;
      ///@}

      /**
       * @brief Constructs the initial epistemic state for the loaded task.
       */
      { rep.build_initial() };

      /**
       * @brief Reduces the state using bisimulation contraction when enabled.
       */
      { rep.contract_with_bisimulation() };

      /**
       * @brief Compute the tensor view used by learned heuristics.
       */
      { rep.get_tensor_representation() } -> std::same_as<const GraphTensor &>;

      /**
       * @brief Compute the epistemic successor reached by executing an action.
       */
      { std::as_const(rep).compute_successor(act) } -> std::same_as<StateRepr>;
      /**
       * @name Output Methods
       * Required methods for formatted output.
       */
      ///@{
      { std::as_const(rep).print() };
      { std::as_const(rep).print_dot_format(ofs) };
      { std::as_const(rep).print_dataset_format(ofs) };
      ///@}

      /**
       * @name Operators
       * Required comparison and assignment operators.
       */
      ///@{
      { rep.operator=(other) } -> std::same_as<StateRepr &>;
      { std::as_const(rep).operator<(other) } -> std::same_as<bool>;
      ///@}
      ///
      /**
       * @brief Copy constructor requirement.
       */
      { StateRepr(other) } -> std::same_as<StateRepr>;
    };

/**
 * @tparam StateRepr The state representation class satisfying
 * StateRepresentation
 */
template <StateRepresentation StateRepr> class State {
public:
  /** \brief Constructor without parameters.
   *
   * It creates \ref m_representation calling its **StateRepr** constructor.*/
  State() = default;

  /** \brief Copy constructor.
   *
   * @param other The State to copy from.
   */
  State(const State &other);

  /** \brief Constructor with that set *this* as successor of the given one.
   *
   * @param prev_state The predecessor state.
   * @param executed_action The action applied to \p prev_state.
   */
  State(const State &prev_state, const Action &executed_action);

  /** \brief Compute the next planner state produced by applying an action.
   *
   *  @param executed_action The action to execute from the current state.
   *  @return The successor planner state.
   */
  [[nodiscard]] State compute_successor(const Action &executed_action);

  /** \brief Getter of \ref m_executed_actions_id.
   *
   * @return the \ref ActionIdsList that represents all the executed Action
   * before to obtain *this*.*/
  [[nodiscard]] const ActionIdsList &get_executed_actions() const;
  /** \brief Getter of plan length.
   *
   * @return the length of the plan up to *this*.*/
  [[nodiscard]] unsigned short get_plan_length() const;
  /** \brief Setter for the field \ref m_heuristic_value.
   *
   * @param[in] heuristic_value: the int to copy in \ref m_heuristic_value.*/
  void set_heuristic_value(short heuristic_value);

  /** \brief Setter for the field \ref m_old_heuristic_value.
   *
   * @param[in] old_heuristic_value: the int to copy in \ref
   * m_old_heuristic_value.*/
  void set_old_heuristic_value(const short old_heuristic_value);

  /** \brief Setter for the field \ref m_heuristics_evaluation_times.
   *
   * @param[in] heuristics_evaluation_times: the int to copy in \ref
   * m_heuristics_evaluation_times.*/
  void set_heuristics_evaluation_times(const short heuristics_evaluation_times);

  /** \brief Increase the field \ref m_heuristics_evaluation_times by one.*/
  void increase_heuristics_evaluation_times();

  /** \brief Getter of \ref m_heuristic_value.
   *
   * @return the heuristic value of *this*.*/
  [[nodiscard]] short get_heuristic_value() const;

  /** \brief Getter of \ref m_old_heuristic_value.
   *
   * @return the old heuristic value of  *this*.*/
  [[nodiscard]] float get_old_heuristic_value() const;

  /** \brief Getter of \ref m_heuristics_evaluation_times.
   *
   * @return the number of times the heuristic value for *this* has been
   * evaluated.*/
  [[nodiscard]] float get_heuristics_evaluation_times() const;

  /** \brief Getter of \ref m_representation.
   *
   * @return the m_representation of *this*.*/
  [[nodiscard]] const StateRepr &get_representation() const;

  /** \brief Getter of \ref m_tensor_representation.
   *
   * @return the m_tensor_representation of *this*.*/
  [[nodiscard]] const GraphTensor &get_tensor_representation();

  /** \brief Append the id of one executed action to the stored plan prefix.
   *
   * @param[in] to_add The action whose id is appended.
   */
  void add_executed_action(const Action &to_add);

  /** \brief Replace the underlying state representation.
   *
   * @param[in] to_set The new representation to store.
   */
  void set_representation(const StateRepr &to_set);

  /** \brief Check whether all designated worlds in this state entail a fluent.
   *
   * The actual entailment test is delegated to the specific state
   * representation (\ref m_representation).
   *
   * @param to_check: the Fluent to check if is entailed by *this*.
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.
   */
  [[nodiscard]] bool entails(const Fluent &to_check) const;

  /** \brief Check whether all designated worlds entail a conjunctive fluent set.
   *
   * The actual entailment test is delegated to the specific state
   * representation (\ref m_representation).
   *
   * @param to_check: the conjunctive set of Fluent to check if is entailed
   * by *this*.
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.*/
  [[nodiscard]] bool entails(const FluentsSet &to_check) const;
  /** \brief Check whether all designated worlds entail a fluent formula.
   *
   * The actual entailment test is delegated to the specific state
   * representation (\ref m_representation).
   *
   * @param to_check: the DNF \ref FluentFormula to check if is entailed by
   * *this*.
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.*/
  [[nodiscard]] bool entails(const FluentFormula &to_check) const;

  /** \brief Check whether the epistemic state entails a belief formula.
   *
   * The actual entailment test is delegated to the specific state
   * representation (\ref m_representation).
   *
   * @param to_check: the \ref BeliefFormula to check if is entailed by *this*.
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.*/
  [[nodiscard]] bool entails(const BeliefFormula &to_check) const;

  /** \brief Check whether the epistemic state entails a conjunction of belief formulas.
   *
   * The actual entailment test is delegated to the specific state
   * representation (\ref m_representation).
   *
   *
   * @param to_check: the CNF \ref FormulaeList to check if is entailed by
   * *this*.
   *
   *
   * @return true if \p to_check is entailed by *this*.
   * @return false if \p -to_check is entailed by *this*.*/
  [[nodiscard]] bool entails(const FormulaeList &to_check) const;

  /** \brief Build the initial planner state for the currently loaded task.
   *
   * The actual construction of the State is left to the specific
   * State-representation (\ref m_representation).
   */
  void build_initial();

  /** \brief Check whether an action is executable in this state.
   *
   * @param[in] act: The action to be checked on *this*.
   * @return true: \p act is executable in *this*.
   * @return false: \p act is not executable in *this*.*/
  [[nodiscard]] bool is_executable(const Action &act) const;

  /** \brief Function that checks if *this* is a goal state.
   *
   * @return true: if *this* is a goal state.
   * @return false: otherwise.*/
  [[nodiscard]] bool is_goal() const;

  /** \brief Contract the underlying epistemic state using bisimulation.
   *
   * The exact contraction depends on the state representation type.
   */
  void contract_with_bisimulation();

  /** \brief The copy operator.
   *
   * @param [in] to_assign: the State to assign to *this*.
   * @return This with the copied assigned values.*/
  State &operator=(const State<StateRepr> &to_assign);

  /** \brief The < operator for set operations.
   *
   * The result is left to the representations.
   *
   * @param [in] to_compare: the State to to_compare to *this*.
   * @return true: if *this* is smaller than to_compare.
   * @return false: otherwise.*/
  bool operator<(const State<StateRepr> &to_compare) const;

  /**
   * \brief Equality operator.
   *
   * Two states are equal if neither is less than the other.
   *
   * \param to_compare The state to compare with.
   * \return true if the states are equal, false otherwise.
   */
  bool operator==(const State<StateRepr> &to_compare) const;

  /**
   * \brief Inequality operator.
   *
   * Two states are not equal if one is less than the other.
   *
   * \param to_compare The state to compare with.
   * \return true if the states are not equal, false otherwise.
   */
  bool operator!=(const State<StateRepr> &to_compare) const;

  /** \brief Function that prints the information of *this*.
   */
  void print() const;

  /** \brief Function that prints the information of *this* in dot format.
   * \param ofs The output file stream to print to.
   */
  void print_dot_format(std::ofstream &ofs) const;

  /** \brief Print this state in the dataset format used by learned heuristics.
   * \param ofs The output stream to print to.
   */
  void print_dataset_format(std::ofstream &ofs) const;

private:
  /** \brief The type of state m_representation.
   *
   * One of the Possible representation for a State */
  StateRepr m_representation;

  /** \brief The list of executed Action to get from the initial state to
   * *this*.
   *
   * Is a std::vector because we can repeat the same action.
   * @see action and action::m_id.*/
  ActionIdsList m_executed_actions_id;

  /** \brief The heuristic value of the *this*.
   *
   * This value is given by the chosen implementation of Heuristics.*/
  int m_heuristic_value = 0;

  /** \brief The old heuristic value of the *this*, used to compute the RL
   * heuristic value*/
  float m_old_heuristic_value = 0;
  /** \brief Number of time *this* has been evaluated, used to compute the RL
   * heuristic value*/
  float m_heuristics_evaluation_times = 0;

  /** \brief Setter for the field \ref m_executed_actions_id.
   *
   * @param[in] to_set: the list of ActionId object to copy in \ref
   * m_executed_actions_id.*/
  void set_executed_actions(const ActionIdsList &to_set);
};

/**Implementation of the template class State<StateRepr>*/
#include "State.tpp"
