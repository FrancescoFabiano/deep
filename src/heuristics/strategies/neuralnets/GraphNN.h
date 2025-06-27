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
 * Network (GNN).
 *
 * This structure encapsulates the graph as a set of tensors:
 * - \ref edge_ids: A 2 x num_edges tensor of symbolic node IDs representing the
 * source and destination of each edge.
 * - \ref edge_attrs: A num_edges x 1 tensor of edge attributes or labels,
 * aligned with \ref edge_ids.
 * - \ref real_node_ids: A num_nodes x 1 tensor mapping symbolic node IDs to
 * their corresponding real or hashed node IDs.
 *
 * All tensors are designed for compatibility with PyTorch and GNN frameworks.
 */
struct GraphTensor {
  torch::Tensor edge_ids;   ///< [2, num_edges] Symbolic source and destination
                            ///< node IDs for each edge (torch::kInt64).
  torch::Tensor edge_attrs; ///< [num_edges, 1] Edge attributes or labels,
                            ///< aligned with edge_ids (torch::kInt64).
  torch::Tensor real_node_ids; ///< [num_nodes, 1] Mapping from symbolic node
                               ///< IDs to real/hashed node IDs (torch::kUInt64
                               ///< for hashed cases).
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
  [[nodiscard]] short get_score(const State<StateRepr> &state);

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

  GraphTensor
      m_goal_graph_tensor; ///< This is the goal tensor, computed only once for
                           ///< efficiency. If merged is active also the
                           ///< additional structural nodes are added

  size_t m_symbolic_id = 0; ///< Current symbolic ID counter (will be
                            ///< incremented if a new ID is assigned)
  std::unordered_map<size_t, size_t>
      m_node_to_symbolic; ///<  Map from real node IDs to symbolic IDs.
  std::vector<size_t>
      m_real_node_ids; ///< Vector storing real node IDs in symbolic order.
                       ///< (Assume that the position is meaningful)
  std::vector<int64_t> m_edge_src; ///< Source node IDs for each edge. (Assume
                                   ///< that the position is meaningful)
  std::vector<int64_t> m_edge_dst; ///< Destination node IDs for each edge.
                                   ///< (Assume that the position is meaningful)
  std::vector<int64_t>
      m_edge_labels; ///< Labels or attributes for each edge. (Assume that the
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

  const torch::TensorOptions m_options = torch::TensorOptions().dtype(
      torch::kInt64); ///< Tensor options for edge and attribute tensors (int64)
  const torch::TensorOptions m_options_node_ids = torch::TensorOptions().dtype(
      torch::kUInt64); ///< Tensor options for node ID tensors (kUInt64)

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
  void fill_graph_tensor(GraphTensor &tensor);

  /**
   * \brief Converts the goal graph into the info that will then be added to a
   * Tensor. If the tensor are generated with the goal (merged) this info will
   * be directly embedded in the states. Otherwise, it will populate
   * m_goal_graph_tensor to be passed as argument.
   */
  void populate_with_goal();

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
};

#include "GraphNN.tpp"
