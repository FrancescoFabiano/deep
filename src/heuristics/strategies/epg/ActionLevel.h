/**
 * \class ActionLevel
 * \brief Class that implements an action level of the planning graph.
 *
 * An ActionLevel represents a set of executable actions at a specific depth in
 * the planning graph.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano.
 * \date May 31, 2025
 */

#pragma once

#include "Action.h"

/**
 * \brief Represents a level in the planning graph containing executable
 * actions.
 */
class ActionLevel {
public:
  /// \name Constructors
  ///@{

  /**
   * \brief Default constructor. Initializes the action level with depth 0 and
   * an empty action set.
   */
  ActionLevel() = default;

  /**
   * \brief Constructs an ActionLevel with a given set of actions and depth 0.
   * \param[in] actions The set of actions to assign to this level.
   */
  explicit ActionLevel(const ActionsSet &actions);

  /**
   * \brief Constructs an ActionLevel with a given set of actions and a specific
   * depth. \param[in] actions The set of actions to assign to this level.
   * \param[in] depth The depth to assign to this level.
   */
  ActionLevel(const ActionsSet &actions, unsigned short depth);

  ///@}

  /// \name Setters
  ///@{

  /**
   * \brief Sets the depth of this action level.
   * \param[in] depth The value to assign to m_depth.
   */
  void set_depth(unsigned short depth) noexcept;

  /**
   * \brief Sets the set of actions for this level.
   * \param[in] actions The set of actions to assign.
   */
  void set_actions(const ActionsSet &actions);

  /**
   * \brief Adds a single action to this level if not already present.
   * \param[in] act The action to add.
   */
  void add_action(const Action &act);

  ///@}

  /// \name Getters
  ///@{

  /**
   * \brief Gets the depth of this action level.
   * \return The depth value.
   */
  [[nodiscard]] unsigned short get_depth() const noexcept;

  /**
   * \brief Gets the set of actions for this level.
   * \return A const reference to the set of actions.
   */
  [[nodiscard]] const ActionsSet &get_actions() const noexcept;

  ///@}

  /**
   * \brief Assignment operator.
   * \param[in] to_assign The ActionLevel to copy into this.
   * \return True if assignment was successful.
   */
  ActionLevel &operator=(const ActionLevel &to_assign);

  /**
   * \brief Prints this ActionLevel to the given output stream.
   */
  void print() const;

private:
  /// \brief The set of executable actions at this level.
  ActionsSet m_actions = {};

  /// \brief The depth of the action level.
  unsigned short m_depth = 0;
};
