/**
 * \class KripkeWorld
 * \brief Represents a possible interpretation of the world and agents' beliefs.
 *
 * \details A KripkeWorld is a consistent set of \ref fluent (a \ref FluentsSet), with an information set
 *          representing the possibilities agents consider true.
 * \see KripkeState, KripkeStorage
 * \copyright GNU Public License.
 * \author Francesco Fabiano.
 * \date May 17, 2025
 */
#pragma once

#include <iostream>
#include <set>
#include <memory>

#include "utilities/Define.h"

class KripkeWorld
{
    friend class KripkeWorldPointer;

public:
    /// \name Constructors & Destructor
    ///@{
    /** \brief Default constructor. */
    KripkeWorld() = default;

    /** \brief Construct from a set of fluents.
     *  \param[in] description The set of fluents to initialize this world.
     */
    explicit KripkeWorld(const FluentsSet& description);

    /** \brief Copy constructor.
     *  \param[in] world The KripkeWorld to copy.
     */
    KripkeWorld(const KripkeWorld& world);

    /** \brief Destructor. */
    ~KripkeWorld() = default;

    /** \brief Copy assignment operator.
     *  \param[in] to_assign The KripkeWorld to assign from.
     *  \return Reference to this.
     */
    KripkeWorld& operator=(const KripkeWorld& to_assign);
    ///@}

    /// \name Getters
    ///@{
    /** \brief Get the set of fluents describing this world.
     *  \return Reference to the set of fluents.
     */
    [[nodiscard]] const FluentsSet& get_fluent_set() const noexcept;

    /** \brief Get the unique id of this world.
     *  \return The unique id.
     */
    [[nodiscard]] KripkeWorldId get_id() const noexcept;
    ///@}

    /// \name Entailment
    ///@{
    /** \brief Check if a fluent is entailed by this world.
     *  \param[in] to_check The fluent to check.
     *  \return True if entailed, false otherwise.
     */
    [[nodiscard]] bool entails(const Fluent &to_check) const;

    /** \brief Check if a conjunctive set of fluents is entailed.
     *  \param[in] to_check The set of fluents.
     *  \return True if all are entailed, false otherwise.
     */
    [[nodiscard]] bool entails(const FluentsSet& to_check) const;

    /** \brief Check if a DNF fluent formula is entailed.
     *  \param[in] to_check The formula to check.
     *  \return True if entailed, false otherwise.
     */
    [[nodiscard]] bool entails(const FluentFormula& to_check) const;
    ///@}

    /// \name Comparison Operators
    ///@{
    /** \brief Less-than operator based on unique id.
     *  \param[in] to_compare The KripkeWorld to compare.
     *  \return True if this < to_compare.
     */
    [[nodiscard]] bool operator<(const KripkeWorld& to_compare) const noexcept;

    /** \brief Greater-than operator based on unique id.
     *  \param[in] to_compare The KripkeWorld to compare.
     *  \return True if this > to_compare.
     */
    [[nodiscard]] bool operator>(const KripkeWorld& to_compare) const noexcept;

    /** \brief Equality operator based on unique id.
     *  \param[in] to_compare The KripkeWorld to compare.
     *  \return True if equal.
     */
    [[nodiscard]] bool operator==(const KripkeWorld& to_compare) const noexcept;
    ///@}

    /// \name Utility
    ///@{
    /** \brief Print all information about this world.
     * \param os The output stream to print to (default: std::cout).
     */
    void print(std::ostream& os = std::cout) const;
    ///@}

private:
    /// \name Data Members
    ///@{
    /** \brief The set of fluents describing this world. */
    FluentsSet m_fluent_set;
    /** \brief The unique id of this world. */
    KripkeWorldId m_id;
    ///@}

    /// \name Internal Methods
    ///@{
    /** \brief Hash this world's fluents into a unique id.
     *  \return The unique id.
     */
    KripkeWorldId hash_fluents_into_id() const;

    /** \brief Set the fluent set, ensuring consistency.
     *  \param[in] description The set of fluents.
     */
    void set_fluent_set(const FluentsSet& description);

    /** \brief Set the unique id for this world. */
    void set_id();
    ///@}
};

/**
 * \class KripkeWorldPointer
 * \brief Wrapper for std::shared_ptr<const KripkeWorld> for use in KripkeStorage.
 *
 * \details Enables set operations and pointer comparison by world content.
 * \copyright GNU Public License.
 * \author Francesco Fabiano.
 * \date May 17, 2025
 */
class KripkeWorldPointer
{
public:
    /// \name Constructors & Destructor
    ///@{
    /** \brief Default constructor. */
    KripkeWorldPointer() = default;

    /** \brief Construct from shared_ptr (copy).
     *  \param[in] ptr The pointer to assign.
     *  \param[in] repetition The repetition count (default 0).
     */
    KripkeWorldPointer(const std::shared_ptr<const KripkeWorld>& ptr, unsigned short repetition = 0);

    /** \brief Construct from shared_ptr (move).
     *  \param[in] ptr The pointer to move.
     *  \param[in] repetition The repetition count (default 0).
     */
    KripkeWorldPointer(std::shared_ptr<const KripkeWorld>&& ptr, unsigned short repetition = 0);

    /** \brief Construct from KripkeWorld by value.
     *  \param[in] world The world to point to.
     *  \param[in] repetition The repetition count (default 0).
     */
    explicit KripkeWorldPointer(const KripkeWorld& world, unsigned short repetition = 0);

    /** \brief Destructor. */
    ~KripkeWorldPointer() = default;

    /** \brief Copy assignment operator.
     *  \param[in] to_copy The pointer to assign from.
     *  \return Reference to this.
     */
    KripkeWorldPointer& operator=(const KripkeWorldPointer& to_copy);
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
    void set_ptr(const std::shared_ptr<const KripkeWorld>& ptr);

    /** \brief Set the underlying pointer (move).
     *  \param[in] ptr The pointer to move.
     */
    void set_ptr(std::shared_ptr<const KripkeWorld>&& ptr);

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
    [[nodiscard]] const FluentsSet& get_fluent_set() const;

    /** \brief Get the id of the pointed world plus repetition.
     *  \return The id.
     */
    [[nodiscard]] KripkeWorldId get_id() const noexcept;

    /** \brief Get the numerical id of the pointed world.
     *  \return The id.
     */
    [[nodiscard]] KripkeWorldId get_internal_world_id() const noexcept;

    /** \brief Get the fluent-based id of the pointed world.
     *  \return The id.
     */
    [[nodiscard]] KripkeWorldId get_fluent_based_id() const noexcept;
    ///@}

    /// \name Entailment
    ///@{
    /** \brief Check if a fluent is entailed by the pointed world.
     *  \param[in] to_check The fluent to check.
     *  \return True if entailed.
     */
    [[nodiscard]] bool entails(const Fluent &to_check) const;

    /** \brief Check if a set of fluents is entailed by the pointed world.
     *  \param[in] to_check The set to check.
     *  \return True if all are entailed.
     */
    [[nodiscard]] bool entails(const FluentsSet& to_check) const;

    /** \brief Check if a DNF formula is entailed by the pointed world.
     *  \param[in] to_check The formula to check.
     *  \return True if entailed.
     */
    [[nodiscard]] bool entails(const FluentFormula& to_check) const;
    ///@}

    /// \name Comparison Operators
    ///@{
    /** \brief Less-than operator for set operations.
     *  \param[in] to_compare The pointer to compare.
     *  \return True if this < to_compare.
     */
    [[nodiscard]] bool operator<(const KripkeWorldPointer& to_compare) const noexcept;

    /** \brief Greater-than operator for set operations.
     *  \param[in] to_compare The pointer to compare.
     *  \return True if this > to_compare.
     */
    [[nodiscard]] bool operator>(const KripkeWorldPointer& to_compare) const noexcept;

    /** \brief Equality operator.
     *  \param[in] to_compare The pointer to compare.
     *  \return True if equal.
     */
    [[nodiscard]] bool operator==(const KripkeWorldPointer& to_compare) const noexcept;
    ///@}

private:
    /// \name Data Members
    ///@{
    /** \brief The wrapped pointer. */
    std::shared_ptr<const KripkeWorld> m_ptr;
    /** \brief The repetition count for oblivious observations. */
    unsigned short m_repetition = 0;
    ///@}
};
