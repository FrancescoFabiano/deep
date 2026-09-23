// GraphTensor.h
#pragma once
#include <cstdint>
#include <vector>

/**
 * \struct GraphTensor
 * \brief Tensorized graph view of a Kripke state for ONNX-based learned
 * heuristics.
 *
 * \details This structure stores the minimal tensor components extracted from a
 * planner state:
 * - designated world ids (`pointed_ids`);
 * - graph edges (`edge_src`, `edge_dst`);
 * - edge labels (`edge_attrs`);
 * - per-node symbolic ids (`real_node_ids`);
 * - optional flattened bitmask encodings (`real_node_ids_bitmask`).
 *
 * All arrays are designed for compatibility with ONNX Runtime and GNN models
 * exported to ONNX format.
 */
struct GraphTensor {


    /// Encoded node ids of the designated worlds in the represented state.
    /// These ids use the same symbolic node-id space as \ref real_node_ids.
    std::vector<int64_t> pointed_ids;

    std::vector<int64_t> edge_src;
  ///< [num_edges] Symbolic source node id for each edge.
  std::vector<int64_t>
      edge_dst; ///< [num_edges] Symbolic destination node id for each edge.

  /// `edge_src` and `edge_dst` together form the usual `[2, num_edges]`
  /// `edge_index` representation expected by graph-learning pipelines.

  std::vector<int64_t>
      edge_attrs; ///< [num_edges] Edge attributes or labels aligned with edges.
  std::vector<int64_t> real_node_ids;
  ///< [num_nodes] Mapping from symbolic node ids to planner-side node ids.

  std::vector<uint8_t> real_node_ids_bitmask;
  ///< Flattened bitmask encoding used by BITMASK datasets when requested.
};
