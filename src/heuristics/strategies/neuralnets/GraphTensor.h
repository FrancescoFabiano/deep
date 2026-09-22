// GraphTensor.h
#pragma once
#include <cstdint>
#include <vector>

/**
 * \struct GraphTensor
 * \brief Represents a graph in tensor format for input to a Graph Neural
 * Network (GNN) using ONNX.
 *
 * This structure encapsulates the graph as a set of arrays:
 * - edge_src: 1D array of symbolic source node IDs for each edge.
 * - edge_dst: 1D array of symbolic destination node IDs for each edge.
 * - edge_attrs: 1D array of edge attributes or labels, aligned with edges.
 * - real_node_ids: 1D array mapping symbolic node IDs to their corresponding
 * - real_node_ids_bitmask: flatten multiDim array mapping symbolic node IDs to
 * their corresponding BITMASK IDs.
 *
 * All arrays are designed for compatibility with ONNX Runtime and GNN models
 * exported to ONNX format.
 */
struct GraphTensor {


    std::vector<int64_t> pointed_ids;
    /// Encoded node IDs of the designated/pointed worlds.
    /// These IDs use the same state-node ID representation as real_node_ids.

    std::vector<int64_t> edge_src;
  ///< [1, num_edges] -- First dimension.
  ///< Symbolic source node ID for each edge.
  std::vector<int64_t>
      edge_dst; ///< [1, num_edges] -- Second dimension. Symbolic destination
  ///< node ID for each edge.

  /// edge_src and edge_dest are used to create edge_index -> list <edge_source,
  /// edge_target> -> [2, num_edges]

  std::vector<int64_t>
      edge_attrs; ///< [1, num_edges] Edge attributes or labels,

  ///< aligned with edge_ids.
  std::vector<int64_t> real_node_ids;
  ///< [num_nodes, 1] Mapping from symbolic
  ///< node IDs to real/hashed node IDs.
  ///< aligned with edge_ids.

  std::vector<uint8_t> real_node_ids_bitmask;
  ///< Special Case: BITMASK nodes have BITMASKS as real IDs (lists of 0-1)
  ///< flattened in a single vector (use uint for easier conversion)
};
