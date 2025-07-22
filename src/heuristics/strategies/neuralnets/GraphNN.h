#pragma once
#include "KripkeState.h"
#include "State.h"
#include "neuralnets/GraphNN.h"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>

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
 * real or hashed node IDs.
 *
 * All arrays are designed for compatibility with ONNX Runtime and GNN models
 * exported to ONNX format.
 */
struct GraphTensor {
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
  std::vector<uint64_t> real_node_ids;
  ///< [num_nodes, 1] Mapping from symbolic
  ///< node IDs to real/hashed node IDs.
};

/**
 * \class GraphNN
 * \brief Singleton class for Graph Neural Network-based heuristic evaluation.
 *
 * This class provides an interface for evaluating states using a neural
 * network-based heuristic. It is implemented as a singleton, ensuring only one
 * instance exists during the application's lifetime.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date June 2, 2025
 */
template <StateRepresentation StateRepr> class GraphNN {
public:
  /**
   * \brief Get the singleton instance of GraphNN.
   * \return Reference to the singleton instance.
   */
  static GraphNN &get_instance();

  /**
   * \brief Create the singleton instance of GraphNN.
   */

  static void create_instance();

  /**
   * \brief Get the score for a given state using the neural network heuristic
   * using native C++ code \tparam StateRepr The state representation type.
   * \param state The state to evaluate.
   * \return The heuristic score for the state.
   */
  [[nodiscard]] int get_score(const State<StateRepr> &state);

  /**
   * \brief Get the score for a given state using the neural network heuristic
   * thorugh the python code \tparam StateRepr The state representation type.
   * \param state The state to evaluate.
   * \return The heuristic score for the state.
   */
  [[nodiscard]] short get_score_python(const State<StateRepr> &state);

  /** \brief Deleted copy constructor (singleton pattern). */
  GraphNN(const GraphNN &) = delete;

  /** \brief Deleted copy assignment operator (singleton pattern). */
  GraphNN &operator=(const GraphNN &) = delete;

  /** \brief Deleted move constructor (singleton pattern). */
  GraphNN(GraphNN &&) = delete;

  /** \brief Deleted move assignment operator (singleton pattern). */
  GraphNN &operator=(GraphNN &&) = delete;

private:
  /**
   * \brief Private constructor for singleton pattern.
   */
  GraphNN();

  static GraphNN *instance; ///< Singleton instance pointer

  std::string
      m_checking_file_path;     ///< Path to the file where the state is printed
  std::string m_goal_file_path; ///< Path to the file where the goal is stored
  std::string m_agents_number = std::to_string(
      Domain::get_instance()
          .get_agent_number()); ///< Number of agents in the domain
  std::string m_model_path =
      Configuration::get_instance()
          .get_GNN_model_path(); ///< Path to the GNN model

  GraphTensor m_goal_graph_tensor;
  ///< This is the goal tensor, computed only once for
  ///< efficiency. If merged is active also the
  ///< additional structural nodes are added

  size_t m_symbolic_id = 0;
  ///< Current symbolic ID counter (will be
  ///< incremented if a new ID is assigned)
  std::unordered_map<size_t, size_t>
      m_node_to_symbolic; ///<  Map from real node IDs to symbolic IDs.
  std::vector<size_t> m_real_node_ids;
  ///< Vector storing real node IDs in symbolic order.
  ///< (Assume that the position is meaningful)
  std::vector<int64_t> m_edge_src;
  ///< Source node IDs for each edge. (Assume
  ///< that the position is meaningful)
  std::vector<int64_t> m_edge_dst;
  ///< Destination node IDs for each edge.
  ///< (Assume that the position is meaningful)
  std::vector<int64_t> m_edge_labels;
  ///< Labels or attributes for each edge. (Assume that the
  ///< position is meaningful)

  size_t m_edges_initial_size =
      0; ///< Initial size for the edges vector (to remove the new inserted
         ///< nodes while processing the heuristics)
  size_t m_node_ids_initial_size =
      0; ///< Initial size for the node IDs vector (to remove the new inserted
         ///< nodes while processing the heuristics)
  size_t m_starting_symbolic_id =
      0; ///< Initial symbolic ID (to remove the new inserted nodes while
         ///< processing the heuristics)

  ///// --- ONNX Runtime inference components ---
  Ort::Env m_env{ORT_LOGGING_LEVEL_WARNING,
                 "GraphNNEnv"}; ///< ONNX Runtime environment for GNN inference.
  Ort::SessionOptions m_session_options; ///< ONNX Runtime session options.
  std::unique_ptr<Ort::Session>
      m_session; ///< Pointer to the ONNX Runtime session.
  std::unique_ptr<Ort::AllocatorWithDefaultOptions>
      m_allocator; ///< Allocator for ONNX Runtime memory management.
  std::unique_ptr<Ort::MemoryInfo>
      m_memory_info; ///< Memory info for ONNX Runtime tensors.

  std::vector<std::string>
      m_input_names; ///< Names of the input nodes for the ONNX model.
  std::vector<std::string>
      m_output_names; ///< Names of the output nodes for the ONNX model.

  bool m_model_loaded =
      false; ///< Indicates whether the ONNX model has been loaded.

  float m_normalization_slope =
      1.0f; ///< Normalization slope for the GNN model output.

    float m_normalization_intercept =
    0.0f; ///< Normalization intercept for the GNN model output.

  /**
   * \brief Converts a KripkeState to a minimal GraphTensor representation.
   *
   * This function transforms the given KripkeState into a GraphTensor,
   * extracting only the essential information required for GNN input.
   *
   * \param kstate The KripkeState to convert.
   * \return A GraphTensor containing the minimal tensor representation of the
   * graph.
   */
  [[nodiscard]] GraphTensor state_to_tensor_minimal(const KripkeState &kstate);

  /**
   * \brief Checks the consistency between a GraphTensor and the original state
   * by comparing their DOT representations.
   *
   * This function generates a DOT file from the given GraphTensor and another
   * DOT file from the original state, then compares the two files to verify
   * that the tensor representation matches the original graph structure.
   *
   * \param state_tensor The tensor representation of the graph.
   * \param state The original state to compare against.
   * \return True if the DOT files are equivalent, false otherwise.
   */
  [[nodiscard]] bool
  check_tensor_against_dot(const GraphTensor &state_tensor,
                           const State<StateRepr> &state) const;

  /**
   * \brief Writes a GraphTensor to a DOT file for comparison.
   *
   * This function generates a DOT file representing the graph structure encoded
   * in the given GraphTensor. The DOT file is then compared with original state
   * graph for consistency check.
   *
   *
   * \param origin_filename The path to the original states that DOT file.
   * \param state_tensor The GraphTensor containing the graph data to be
   * written.
   */
  [[nodiscard]] static bool
  write_and_compare_tensor_to_dot(const std::string &origin_filename,
                                  const GraphTensor &state_tensor);

  /**
   * \brief Populates the given GraphTensor with the current graph data.
   *
   * This function fills the provided GraphTensor reference with the current
   * edge IDs, edge attributes, and real node IDs, converting the internal
   * graph representation into tensor format suitable for GNN input.
   *
   * \param tensor Reference to the GraphTensor to be populated.
   */
  void fill_graph_tensor(GraphTensor &tensor) const;

  /**
   * \brief Converts the goal graph into the info that will then be added to a
   * Tensor. If the tensor are generated with the goal (merged) this info will
   * be directly embedded in the states. Otherwise, it will populate
   * m_goal_graph_tensor to be passed as argument.
   */
  void populate_with_goal();

  /**
   * \brief Initializes the ONNX Runtime model for GNN inference.
   *
   * Sets up the ONNX Runtime environment, session options, loads the GNN model,
   * and prepares input/output names and memory information required for
   * inference. This function should be called before performing any inference
   * with the model.
   */
  void initialize_onnx_model();

  /**
   * \brief Runs ONNX Runtime inference on the provided GraphTensor.
   *
   * This method takes a GraphTensor representing the input graph, prepares the
   * necessary ONNX Runtime tensors, and performs inference using the loaded GNN
   * model. It returns the resulting score or output from the neural network.
   *
   * \param tensor The GraphTensor containing the graph data for inference.
   * \return The output score from the GNN model as a float.
   */
  float run_inference(const GraphTensor &tensor) const;

  /**
   * \brief Compares the inference results of the C++ ONNX model with those
   * obtained from Python (both ONNX and PyTorch models) for correctness
   * checking.
   *
   * This function is used to validate the consistency between the C++ ONNX
   * Runtime inference and the Python-based predictors. It takes the given state
   * and the C++ computed score, then calls the Python predictor to perform
   * inference using both ONNX and PyTorch models. The results are compared to
   * ensure correctness and identify any discrepancies between the
   * implementations.
   *
   * \param state The state to evaluate.
   * \param c_score The score computed by the C++ ONNX model.
   */
  void compare_predictions(const State<StateRepr> &state, float c_score);

  /**
   * \brief Returns the symbolic ID for a node, assigning a new one if it does
   * not exist.
   *
   * If the given node is already present in the m_node_to_symbolic map, returns
   * its symbolic ID. Otherwise, assigns the next available symbolic ID to the
   * node, updates the map, appends the real node ID to the m_real_node_ids
   * vector, and increments the m_symbolic_id counter.
   *
   * \param node The real node ID to assign or retrieve a symbolic ID for.
   * \return The symbolic ID corresponding to the node.
   */
  [[nodiscard]] size_t get_symbolic_id(size_t node);

  /**
   * \brief Adds an edge to the graph representation while also adding the
   * symbolic ids of the two vertices.
   *
   * Appends a new edge to the internal edge lists, specifying the source node,
   * destination node, and edge label. The order of insertion is meaningful and
   * should be consistent with the symbolic node IDs.
   *
   * To get the symbolic ids it uses the function \ref get_symbolic_id.
   *
   *
   * \param src The source node ID (real).
   * \param dst The destination node ID (real).
   * \param label The label or attribute associated with the edge.
   */
  void add_edge(int64_t src, int64_t dst, int64_t label);

  /**
   * \brief Parses the constant used for normalization in prediction results.
   *
   * This function reads and sets the normalization constant, which is used to
   * multiply the float value returned by the neural network prediction to scale
   * the output appropriately.
   */
  void parse_constant_for_normalization();
};

#include "GraphNN.tpp"
