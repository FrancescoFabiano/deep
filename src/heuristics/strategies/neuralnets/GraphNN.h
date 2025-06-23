#pragma once
#include "State.h"
#include <string>
#include "neuralnets/GraphNN.h"
#include <torch/torch.h>
#include "KripkeState.h"
#include <unordered_map>


/**
 * \struct GraphTensor
 * \brief Structure representing a graph in tensor format for GNN input.
 *
 * Contains edge indices, edge attributes, and node IDs.
 */
struct GraphTensor {
    torch::Tensor edge_index;   // [2, num_edges]
    torch::Tensor edge_attr;    // [num_edges, 1]
    std::vector<size_t> node_ids; // Optional: useful for mapping back
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
   * \brief Get the score for a given state using the neural network heuristic.
   * \tparam StateRepr The state representation type.
   * \param state The state to evaluate.
   * \return The heuristic score for the state.
   */
  [[nodiscard]] short get_score(const State<StateRepr> &state);

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

    /**
         * \brief Converts a KripkeState to a minimal GraphTensor representation.
         *
         * This function transforms the given KripkeState into a GraphTensor,
         * extracting only the essential information required for GNN input.
         *
         * \param kstate The KripkeState to convert.
         * \return A GraphTensor containing the minimal tensor representation of the graph.
         */
    GraphTensor kripke_to_tensor_minimal(const KripkeState& kstate);

};

#include "GraphNN.tpp"
