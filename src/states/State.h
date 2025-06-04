/**
* \class State
 * \brief Templatic Class that encodes a state of Planner.h.
 *
 * \details  This is the *TEMPLATE* and will be used as black box from planner.h:
 * its implementation will depend on the initial choices.
 *
 * This class should be used to check entailment and to produce successors.
 *
 * Template and not virtual to keep the pointer and, since the type of search is decided
 * at compile-time virtual overhead is not necessary.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 20, 2025
 */
#pragma once
#include <concepts>

#include "utilities/Define.h"
#include "actions/Action.h"


/**
 * @brief Concept that enforces the required interface for a state representation type `StateRepr`.
 *
 * This concept defines the contract that a type `StateRepr` must fulfill to be used with the `State<StateRepr>` class.
 * It ensures that `StateRepr` provides entailment checks, executability conditions, printing, and comparison.
 *
 * @tparam StateRepr The type to be checked against the required interface.
 */
template <typename StateRepr>
concept StateRepresentation = requires(StateRepr rep, const Fluent& f, const FluentsSet& fs,
                                       const FluentFormula& ff, const BeliefFormula& bf,
                                       const FormulaeList& fl, const Action& act,
                                       std::ofstream& ofs, const StateRepr& other)
{
    /**
     * @name Entailment Methods for logical entailment evaluation
     */
    ///@{
    { std::as_const(rep).entails(f) } -> std::same_as<bool>;
    { std::as_const(rep).entails(fs) } -> std::same_as<bool>;
    { std::as_const(rep).entails(ff) } -> std::same_as<bool>;
    { std::as_const(rep).entails(bf) } -> std::same_as<bool>;
    { std::as_const(rep).entails(fl) } -> std::same_as<bool>;
    ///@}

    /**
     * @brief Constructs the initial state.
     */
    { rep.build_initial() };

    /**
     * @brief Reduces the state using bisimulation contraction.
     */
    { rep.contract_with_bisimulation() };


    /**
     * @brief Successor computation method.
     * \warning compute_successor is not working if set to const, no idea why
     */
    { std::as_const(rep).compute_successor(act) } -> std::same_as<StateRepr>;
    //{ rep.compute_successor(act) } -> std::same_as<StateRepr>;
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
    { rep.operator=(other) } -> std::same_as<StateRepr&>;
    { std::as_const(rep).operator<(other) } -> std::same_as<bool>;
    ///@}
    ///
    /**
     * @brief Copy constructor requirement.
     */
    { StateRepr(other) } -> std::same_as<StateRepr>;
};


/**
 * @tparam StateRepr The state representation class satisfying StateRepresentation
 */
template <StateRepresentation StateRepr>
class State
{
public:
    /** \brief Constructor without parameters.
     *
     * It creates \ref m_representation calling its **StateRepr** constructor.*/
    State() = default;

    /** \brief Copy constructor.
     *
     * @param other The State to copy from.
     */
    State(const State& other);

    /** \brief Constructor with that set *this* as successor of the given one.
     *
     * @param prev_state: the \ref State that is the predecessor of *this*.
     *  @param executed_action: the \ref action applied to \p prev_state.*/
    State(const State& prev_state, const Action& executed_action);

    /** \brief Function that compute the next state applying an action to this.
    *
    *  @param executed_action: the \ref action applied to \p prev_state.*/
    [[nodiscard]] State compute_successor(const Action& executed_action);

    /** \brief Getter of \ref m_executed_actions_id.
     *
     * @return the \ref ActionIdsList that represents all the executed \ref action before to obtain *this*.*/
    [[nodiscard]] const ActionIdsList& get_executed_actions() const;
    /** \brief Getter of \ref m_plan_length.
     *
     * @return the length of the plan up to *this*.*/
    [[nodiscard]] unsigned short get_plan_length() const;
    /** \brief Setter for the field \ref m_heuristic_value.
     *
     * @param[in] heuristic_value: the int to copy in \ref m_heuristic_value.*/
    void set_heuristic_value(short heuristic_value);
    /** \brief Getter of \ref m_heuristic_value.
     *
     * @return the heuristic value *this*.*/
    [[nodiscard]] short get_heuristic_value() const;
    /** \brief Getter of \ref m_representation.
     *
     * @return the m_representation of *this*.*/
    [[nodiscard]] const StateRepr& get_representation() const;

    /** \brief Function that add and \ref action_id to \ref m_executed_actions_id.
        *
        * @param[in] to_add: the \ref action_id to add to \ref m_executed_actions_id.*/
    void add_executed_action(const Action& to_add);

    /** \brief Setter of \ref m_representation.
     *
     * @param[in] to_set: the m_representation to assign to \ref m_representation.*/
    void set_representation(const StateRepr& to_set);

    /** \brief Function that checks if *this* entails a \ref fluent.
     *
     * The actual entailment is left to the specific State-representation (\ref m_representation).
     *
     * @param to_check: the \ref fluent to check if is entailed by *this*.
     *
     * @return true if \p to_check is entailed by *this*.
     * @return false if \p -to_check is entailed by *this*.
     */
    [[nodiscard]] bool entails(const Fluent& to_check) const;

    /** \brief Function that checks if *this* entails a conjunctive set of \ref fluent.
     *
     * The actual entailment is left to the specific State-representation (\ref m_representation).
     *
     * @param to_check: the conjunctive set of \ref fluent to check if is entailed by *this*.
     *
     * @return true if \p to_check is entailed by *this*.
     * @return false if \p -to_check is entailed by *this*.*/
    [[nodiscard]] bool entails(const FluentsSet& to_check) const;
    /** \brief Function that checks if *this* entails a DNF \ref FluentFormula.
     *
     * The actual entailment is left to the specific State-representation (\ref m_representation).
     *
     * @param to_check: the DNF \ref FluentFormula to check if is entailed by *this*.
     *
     * @return true if \p to_check is entailed by *this*.
     * @return false if \p -to_check is entailed by *this*.*/
    [[nodiscard]] bool entails(const FluentFormula& to_check) const;

    /** \brief Function that checks if *this* entails a \ref BeliefFormula.
     *
     * The actual entailment is left to the specific State-representation (\ref m_representation).
     *
     * @param to_check: the \ref BeliefFormula to check if is entailed by *this*.
     *
     * @return true if \p to_check is entailed by *this*.
     * @return false if \p -to_check is entailed by *this*.*/
    [[nodiscard]] bool entails(const BeliefFormula& to_check) const;

    /** \brief Function that checks if *this* entails a CNF \ref FormulaeList.
     *
     * The actual entailment is left to the specific State-representation (\ref m_representation).
     *
     *
     * @param to_check: the CNF \ref FormulaeList to check if is entailed by *this*.
     *
     *
     * @return true if \p to_check is entailed by *this*.
     * @return false if \p -to_check is entailed by *this*.*/
    [[nodiscard]] bool entails(const FormulaeList& to_check) const;

    /** \brief Function that builds the initial \ref state and set *this* with it.
     *
     * The actual construction of the \ref state is left to the specific State-representation (\ref m_representation).
     *
     * @see initially*/
    void build_initial();

    /** \brief Function that checks if a given action is executable in *this*.
     *
     * @see action.
     *
     * @param[in] act: The action to be checked on *this*.
     * @return true: \p act is executable in *this*.
     * @return false: \p act is not executable in *this*.*/
    [[nodiscard]] bool is_executable(const Action& act) const;

    /** \brief Function that checks if *this* is a goal state.
     *
     * @return true: if *this* is a goal state.
     * @return false: otherwise.*/
    [[nodiscard]] bool is_goal() const;

    /** \brief Function that determines the minimum \ref e-state that is bisimilar to the current one.
     *
     * The function depends on the type of e-State.
     *
     * @return the minimum bisimilar e-state to *this*.*/
    void contract_with_bisimulation();

    /** \brief The copy operator.
     *
     * @param [in] to_assign: the \ref state to assign to *this*.
     * @return This with the copied assigned values.*/
    State& operator=(const State<StateRepr>& to_assign);

    /** \brief The < operator for set operations.
     *
     * The result is left to the representations.
     *
     * @param [in] to_compare: the \ref state to to_compare to *this*.
     * @return true: if *this* is smaller than to_compare.
     * @return false: otherwise.*/
    bool operator<(const State<StateRepr>& to_compare) const;


    /** \brief Function that prints the information of *this*.
     */
    void print() const;

    /** \brief Function that prints the information of *this* in dot format.
     * \param ofs The output file stream to print to.
     */
    void print_dot_format(std::ofstream& ofs) const;

    /** \brief Function that prints the information of *this* for the generation of the dataset used to train the GNN.
    * \param ofs The output stream to print to.
    */
    void print_dataset_format(std::ofstream& ofs) const;

private:
    /** \brief The type of state m_representation.
     *
     * One of the Possible representation for a State */
    StateRepr m_representation;

    /** \brief The list of executed \ref action to get from the initial state to *this*.
     *
     * Is a std::vector because we can repeat the same action.
     * @see action and action::m_id.*/
    ActionIdsList m_executed_actions_id;


    /** \brief The heuristic value of the *this*.
     *
     * This value is given by the chosen implementation of \ref heuristic.*/
    short m_heuristic_value = 0;

    /** \brief Setter for the field \ref m_executed_actions_id.
     *
     * @param[in] to_set: the list of \ref action_id object to copy in \ref m_executed_actions_id.*/
    void set_executed_actions(const ActionIdsList& to_set);
};

/**Implementation of the template class State<StateRepr>*/
#include "State.tpp"
