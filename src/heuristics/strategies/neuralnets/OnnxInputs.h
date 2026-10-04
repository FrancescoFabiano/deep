// OnnxInputs.h
#pragma once
#include "ExitHandler.h"
#include "neuralnets/GraphTensor.h"
#include <deque>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

/**
 * \struct PackedGraph
 * \brief One or more state graphs laid out as the ONNX models expect them.
 *
 * \details Several graphs are packed as one disjoint graph: node, edge and
 * pointed ids of each graph are shifted by the number of nodes of the graphs
 * before it, and \ref batch records which graph each node belongs to.
 */
struct PackedGraph {
  size_t num_nodes = 0;          ///< Total number of nodes.
  std::vector<int64_t> node_ids; ///< [num_nodes] (non-BITMASK only).
  std::vector<uint8_t>
      node_bits; ///< [num_nodes * bitmask_size], flattened (BITMASK only).
  std::vector<int64_t>
      edge_index; ///< [2 * num_edges]: all sources, then all destinations.
  std::vector<int64_t> edge_attrs; ///< [num_edges] Edge labels.
  std::vector<int64_t>
      batch; ///< [num_nodes] Index of the graph each node belongs to.
  std::vector<int64_t> pointed_ids; ///< Node indices of the designated worlds.
};

/**
 * \brief ONNX Runtime setup shared by GraphNN and FringeEvalRL.
 */
namespace onnx_runtime {
/**
 * \brief The process-wide ONNX Runtime environment.
 *
 * \details One environment for every session, created on first use. Its
 * logger prints errors, and, with --onnx_placement, the execution provider
 * each model node was placed on (ORT reports it when a session is created).
 */
Ort::Env &env();

/**
 * \brief Apply the planner's ONNX options to \p options.
 *
 * \details Graph optimisation, --onnx_threads, logging and the device:
 * --onnx_device cpu never uses CUDA; cuda requires it (built with use_gpu and
 * a CUDA-capable ONNX Runtime) and fails otherwise; auto (default) uses CUDA
 * when the build has it and falls back to the CPU with a warning.
 * Prints the device it settled on.
 */
void configure_session(Ort::SessionOptions &options);

/**
 * \brief Run options for inference: per-run logging stays at warnings even
 * when the session logs verbosely for --onnx_placement.
 */
Ort::RunOptions &run_options();
} // namespace onnx_runtime

/**
 * \class OnnxInputs
 * \brief Builds the ONNX input tensors shared by GraphNN and FringeEvalRL and
 * runs the model.
 *
 * \details ORT does not copy input data, so every buffer wrapped by a tensor
 * must outlive Session::Run. Packed graphs are referenced (they must outlive
 * this object); every other buffer is owned here. std::deque keeps the owned
 * buffers at stable addresses while new ones are added.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date September 28, 2026
 */
class OnnxInputs {
public:
  explicit OnnxInputs(const Ort::MemoryInfo &memory_info)
      : m_memory_info(memory_info) {}

  /**
   * \brief Packs a single graph (batch all zeros).
   * \param graph The graph to pack.
   * \param bitmask Whether nodes are encoded as bitmasks.
   * \param bitmask_size Number of bits per node (only used if \p bitmask).
   */
  [[nodiscard]] static PackedGraph pack(const GraphTensor &graph, bool bitmask,
                                        size_t bitmask_size);

  /**
   * \brief Packs several graphs into one disjoint graph.
   * \param graphs The graphs to pack; graph i gets batch index i.
   * \param bitmask Whether nodes are encoded as bitmasks.
   * \param bitmask_size Number of bits per node (only used if \p bitmask).
   */
  [[nodiscard]] static PackedGraph
  pack(const std::vector<const GraphTensor *> &graphs, bool bitmask,
       size_t bitmask_size);

  /**
   * \brief Adds the inputs of a packed graph, in the planner's order: nodes,
   * edge_index [2,E], edge_attr, batch, and optionally pointed_ids.
   * \param graph The packed graph; it must outlive this object.
   * \param bitmask Nodes as uint8 [N, bitmask_size] instead of int64 [N].
   * \param bitmask_size Number of bits per node (only used if \p bitmask).
   * \param edge_attr_2d edge_attr as [E, 1] (GNN) instead of [E] (RL).
   * \param with_pointed Whether to add pointed_ids (not used for the goal).
   */
  void add_graph(const PackedGraph &graph, bool bitmask, size_t bitmask_size,
                 bool edge_attr_2d, bool with_pointed);

  /// \brief Adds a 1-D uint8 input, taking ownership of \p data.
  void add_owned(std::vector<uint8_t> data);

  /**
   * \brief Checks the input count against the model and runs it.
   * \param session The ONNX session.
   * \param input_names The model's input names.
   * \param output_names The model's output names.
   * \param mismatch_code Exit code used when the input count does not match.
   * \return The model outputs.
   */
  [[nodiscard]] std::vector<Ort::Value>
  run(Ort::Session &session, const std::vector<std::string> &input_names,
      const std::vector<std::string> &output_names,
      ExitHandler::ExitCode mismatch_code);

private:
  /// \name Wrap \p data without copying it (ORT never writes to inputs).
  ///@{
  void add_view(const std::vector<int64_t> &data, std::vector<int64_t> shape);
  void add_view(const std::vector<uint8_t> &data, std::vector<int64_t> shape);
  ///@}

  const Ort::MemoryInfo &m_memory_info; ///< CPU memory info for the tensors.
  std::deque<std::vector<uint8_t>> m_owned_uint8; ///< Owned uint8 buffers.
  std::deque<std::vector<int64_t>> m_shapes;      ///< Tensor shapes.
  std::vector<Ort::Value> m_values;               ///< The input tensors.
};
