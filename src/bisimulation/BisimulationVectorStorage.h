// BisimulationVectorStorage.h
#pragma once

#include "VectorBisWrapper.h"
#include "utilities/Define.h"
#include <memory>

/**
 * \class BisimulationVectorStorage
 * \brief Thread-local singleton that stores all vector fields of Bisimulation.
 *
 * Each thread gets its own instance, ensuring the vectors are created and
 * destroyed only once per thread.
 */
class BisimulationVectorStorage {
public:
  /// \brief Returns the thread-local singleton instance.
  static BisimulationVectorStorage &get_instance() {
    thread_local BisimulationVectorStorage instance;
    return instance;
  }

  // Vector fields (copied from Bisimulation)
  VectorBisWrapper<BisGraph> G{BisPreAllocatedIndex};
  VectorBisWrapper<Bis_qPartition> Q{BisPreAllocatedIndex};
  VectorBisWrapper<Bis_xPartition> X{BisPreAllocatedIndex};
  VectorBisWrapper<BisIndexType> B1{BisPreAllocatedIndex};
  VectorBisWrapper<BisIndexType> B_1{BisPreAllocatedIndex};
  VectorBisWrapper<BisIndexType> splitD{BisPreAllocatedIndex};
  VectorBisWrapper<std::shared_ptr<BisAdjList_1>> borderEdges{};

  // Deleted copy/move
  BisimulationVectorStorage(const BisimulationVectorStorage &) = delete;
  BisimulationVectorStorage &
  operator=(const BisimulationVectorStorage &) = delete;
  BisimulationVectorStorage(BisimulationVectorStorage &&) = delete;
  BisimulationVectorStorage &operator=(BisimulationVectorStorage &&) = delete;

private:
  BisimulationVectorStorage() = default;
  ~BisimulationVectorStorage() = default;
};