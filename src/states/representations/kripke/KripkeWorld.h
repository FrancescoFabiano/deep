/**
 * \class KripkeWorld
 * \brief Represents a possible interpretation of the world and agents' beliefs.
 *
 * \details A KripkeWorld is a consistent set of Fluent (a
 * FluentsSet), with an information set representing the possibilities agents
 * consider true. \see KripkeState, KripkeStorage \copyright GNU Public License.
 * \author Francesco Fabiano.
 * \date May 17, 2025
 */
#pragma once

#include <memory>
#include <set>

#include "utilities/Define.h"

class KripkeWorld {
  friend class KripkeWorldPointer;

public:
  /// \name Constructors & Destructor
  ///@{
  /** \brief Default constructor. */
  KripkeWorld() = default;

  /** \brief Construct from a set of fluents.
   *  \param[in] description The set of fluents to initialize this world.
   */
  explicit KripkeWorld(const FluentsSet &description);

  /** \brief Copy constructor.
   *  \param[in] world The KripkeWorld to copy.
   */
  KripkeWorld(const KripkeWorld &world);

  /** \brief Destructor. */
  ~KripkeWorld() = default;

  /** \brief Copy assignment operator.
   *  \param[in] to_assign The KripkeWorld to assign from.
   *  \return Reference to this.
   */
  KripkeWorld &operator=(const KripkeWorld &to_assign);
  ///@}

  /// \name Getters
  ///@{
  /** \brief Get the set of fluents describing this world.
   *  \return Reference to the set of fluents.
   */
  [[nodiscard]] const FluentsSet &get_fluent_set() const noexcept;

  /** \brief Get the unique id of this world.
   *  \return The unique id.
   */
  [[nodiscard]] KripkeWorldId get_id() const noexcept;
  ///@}

  /// \name Comparison Operators
  ///@{
  /** \brief Less-than operator based on unique id.
   *  \param[in] to_compare The KripkeWorld to compare.
   *  \return True if this < to_compare.
   */
  [[nodiscard]] bool operator<(const KripkeWorld &to_compare) const noexcept;

  /** \brief Greater-than operator based on unique id.
   *  \param[in] to_compare The KripkeWorld to compare.
   *  \return True if this > to_compare.
   */
  [[nodiscard]] bool operator>(const KripkeWorld &to_compare) const noexcept;

  /** \brief Equality operator based on unique id.
   *  \param[in] to_compare The KripkeWorld to compare.
   *  \return True if equal.
   */
  [[nodiscard]] bool operator==(const KripkeWorld &to_compare) const noexcept;
  ///@}

  /// \name Utility
  ///@{
  /** \brief Print all information about this world.
   */
  void print() const;
  ///@}

private:
  /// \name Data Members
  ///@{
  /** \brief The set of fluents describing this world. */
  FluentsSet m_fluent_set;
  /** \brief The unique id of this world. */
  KripkeWorldId m_id = 0;
  ///@}

  /// \name Internal Methods
  ///@{
  /** \brief Hash this world's fluents into a unique id.
   *  \return The unique id.
   */
  [[nodiscard]] KripkeWorldId hash_fluents_into_id() const;

  /** \brief Set the fluent set, ensuring consistency.
   *  \param[in] description The set of fluents.
   */
  void set_fluent_set(const FluentsSet &description);

  /** \brief Set the unique id for this world. */
  void set_id();
  ///@}
};

/**
 * \brief Calculates the maximum number of decimal digits representable by
 * KripkeWorldId. \details This function computes how many digits are needed to
 * represent the maximum value of KripkeWorldId. \return The number of decimal
 * digits.
 */
constexpr unsigned short calculate_max_digits() {
  KripkeWorldId max_rep = std::numeric_limits<KripkeWorldId>::max();
  unsigned short digits = 0;
  while (max_rep > 0) {
    max_rep /= 10;
    ++digits;
  }
  return digits;
}

/**
 * \brief The maximum number of decimal digits for KripkeWorldId.
 */
constexpr unsigned short max_digits = calculate_max_digits();

/**
 * \class KripkeWorldPointer
 * \brief Wrapper for std::shared_ptr<const KripkeWorld> for use in
 * KripkeStorage.
 *
 * \details Enables set operations and pointer comparison by world content.
 * \copyright GNU Public License.
 * \author Francesco Fabiano.
 * \date May 17, 2025
 */
class KripkeWorldPointer {
public:
  /// \name Constructors & Destructor
  ///@{
  /** \brief Default constructor. */
  KripkeWorldPointer() = default;

  /** \brief Construct from shared_ptr (copy).
   *  \param[in] ptr The pointer to assign.
   *  \param[in] repetition The repetition count (default 0).
   */
  explicit KripkeWorldPointer(const std::shared_ptr<const KripkeWorld> &ptr,
                              unsigned short repetition = 0);

  /** \brief Construct from shared_ptr (move).
   *  \param[in] ptr The pointer to move.
   *  \param[in] repetition The repetition count (default 0).
   */
  explicit KripkeWorldPointer(std::shared_ptr<const KripkeWorld> &&ptr,
                              unsigned short repetition = 0);

  /** \brief Construct from KripkeWorld by value.
   *  \param[in] world The world to point to.
   *  \param[in] repetition The repetition count (default 0).
   */
  explicit KripkeWorldPointer(const KripkeWorld &world,
                              unsigned short repetition = 0);

  /**
   * \brief Copy constructor.
   * \param other The KripkeWorldPointer to copy from.
   */
  KripkeWorldPointer(const KripkeWorldPointer &other);

  /** \brief Destructor. */
  ~KripkeWorldPointer() = default;

  /** \brief Copy assignment operator.
   *  \param[in] to_copy The pointer to assign from.
   *  \return Reference to this.
   */
  KripkeWorldPointer &operator=(const KripkeWorldPointer &to_copy);
  ///@}

  /// \name Getters & Setters
  ///@{
  /** \brief Get the underlying pointer.
   *  \return Copy of the shared pointer.
   */
  [[nodiscard]] std::shared_ptr<const KripkeWorld> get_ptr() const noexcept;

  /** \brief Set the underlying pointer (copy).
   *  \param[in] ptr The pointer to assign.
   */
  void set_ptr(const std::shared_ptr<const KripkeWorld> &ptr);

  /** \brief Set the underlying pointer (move).
   *  \param[in] ptr The pointer to move.
   */
  void set_ptr(std::shared_ptr<const KripkeWorld> &&ptr);

  /** \brief Set the repetition count.
   *  \param[in] repetition The value to set.
   */
  void set_repetition(unsigned short repetition) noexcept;

  /** \brief Increase the repetition count.
   *  \param[in] increase The value to add.
   */
  void increase_repetition(unsigned short increase) noexcept;

  /** \brief Get the repetition count.
   *  \return The repetition count.
   */
  [[nodiscard]] unsigned short get_repetition() const noexcept;
  ///@}

  /// \name World Info Access
  ///@{
  /** \brief Get the fluent set of the pointed world.
   *  \return Reference to the fluent set.
   */
  [[nodiscard]] const FluentsSet &get_fluent_set() const;

  /** \brief Get the id of the pointed world plus repetition.
   *  \return The id.
   */
  [[nodiscard]] KripkeWorldId get_id() const noexcept;

  /** \brief Set the id of the pointed world based on its content.
   *  \details This method computes a unique id based on the fluent set and
   * repetition (using hashing).
   */
  void set_id() noexcept;

  /** \brief Get the numerical id of the pointed world.
   *  \return The id.
   */
  [[nodiscard]] KripkeWorldId get_internal_world_id() const noexcept;

  /** \brief Get the fluent-based id of the pointed world.
   *  \return The id.
   */
  [[nodiscard]] KripkeWorldId get_fluent_based_id() const noexcept;
  ///@}

  /// \name Comparison Operators
  ///@{
  /** \brief Less-than operator for set operations.
   *  \param[in] to_compare The pointer to compare.
   *  \return True if this < to_compare.
   */
  [[nodiscard]] bool
  operator<(const KripkeWorldPointer &to_compare) const noexcept;

  /** \brief Greater-than operator for set operations.
   *  \param[in] to_compare The pointer to compare.
   *  \return True if this > to_compare.
   */
  [[nodiscard]] bool
  operator>(const KripkeWorldPointer &to_compare) const noexcept;

  /** \brief Equality operator.
   *  \param[in] to_compare The pointer to compare.
   *  \return True if equal.
   */
  [[nodiscard]] bool
  operator==(const KripkeWorldPointer &to_compare) const noexcept;
  ///@}

private:
  /// \name Data Members
  ///@{
  /** \brief The wrapped pointer. */
  std::shared_ptr<const KripkeWorld> m_ptr;
  KripkeWorldId m_id = 0;
  /** \brief The repetition count for oblivious observations. */
  unsigned short m_repetition = 0;
  ///@}
};
