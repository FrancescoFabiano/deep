/**
 * \class KripkeStorage
 * \brief Singleton class that stores the unique copy of each \ref KripkeWorld created.
 *
 * \details All other classes only store pointers to \ref KripkeWorld, and this class manages creation and uniqueness.
 *          This is done for efficiency, as \ref KripkeWorld objects are used in many places. Follows the Dynamic Programming principle.
 *
 * \see KripkeWorld
 * \copyright GNU Public License.
 * \author Francesco Fabiano.
 * \date May 17, 2025
 */
#pragma once

#include <set>
#include "KripkeWorld.h"

/// \brief Alias for a set of KripkeWorlds, used to store all created worlds.
using KripkeWorldsSet = std::set<KripkeWorld>;

class KripkeStorage
{
public:
    /// \name Singleton Access
    ///@{
    /**
     * \brief Get the singleton instance of KripkeStorage.
     * \return Reference to the singleton instance.
     */
    static KripkeStorage& get_instance() noexcept;
    ///@}

    /// \name World Management
    ///@{
    /**
     * \brief Add a KripkeWorld to the storage and return its pointer.
     *
     * If the world does not exist, it is inserted and a pointer is returned.
     * If it already exists, a pointer to the existing world is returned.
     *
     * \param[in] to_add The KripkeWorld to add.
     * \return A KripkeWorldPointer to the stored world.
     */
    KripkeWorldPointer add_world(const KripkeWorld& to_add);
    ///@}

    /// \name Deleted Special Members
    ///@{
    KripkeStorage(const KripkeStorage&) = delete;
    KripkeStorage& operator=(const KripkeStorage&) = delete;
    ///@}

private:
    /**
     * \brief Private constructor for singleton pattern.
     */
    KripkeStorage() = default;

    /**
     * \brief Set of all created KripkeWorlds. All other classes only have pointers to elements of this set.
     */
    KripkeWorldsSet m_created_worlds;
};
