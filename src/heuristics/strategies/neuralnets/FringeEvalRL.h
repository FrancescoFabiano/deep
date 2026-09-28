#pragma once
#include "KripkeState.h"
#include "State.h"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>

#include "GraphNN.h"

/**
 * \class FringeEvalRL
 * \brief Singleton class for RL-based evaluation.
 *
 * This class provides an interface for evaluating a fringe of states using RL.
 * It is implemented as a singleton, ensuring only one
 * instance exists during the application's lifetime.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date April 9, 2026
 */
template <StateRepresentation StateRepr> class FringeEvalRL {
public:
  /**
   * \brief Get the singleton instance of FringeEvalRL.
   * \return Reference to the singleton instance.
   */
  static FringeEvalRL &get_instance();

  /**
   * \brief Create the singleton instance of FringeEvalRL.
   */

  /// static void create_instance();

  /**
   * \brief Get the scores for a set of states (fringe) using RL
   * using native C++ code \tparam StateRepr The state representation type.
   * \param states The set of states to evaluate.
   * \param raw_scores Optional out-parameter receiving the raw model logits
   * (higher = better) before rank conversion; used by the adaptive schedule
   * to measure model confidence.
   * \return The relative score for the states in the fringe.
   * \note \p states is non-const so that each state caches its own tensor.
   */
  [[nodiscard]] std::vector<float>
  get_score(std::vector<State<StateRepr>> &states,
            std::vector<float> *raw_scores = nullptr);

  /** \brief Deleted copy constructor (singleton pattern). */
  FringeEvalRL(const FringeEvalRL &) = delete;

  /** \brief Deleted copy assignment operator (singleton pattern). */
  FringeEvalRL &operator=(const FringeEvalRL &) = delete;

  /** \brief Deleted move constructor (singleton pattern). */
  FringeEvalRL(FringeEvalRL &&) = delete;

  /** \brief Deleted move assignment operator (singleton pattern). */
  FringeEvalRL &operator=(FringeEvalRL &&) = delete;

private:
  /**
   * \brief Private constructor for singleton pattern.
   */
  FringeEvalRL();

  // static FringeEvalRL *instance; ///< Singleton instance pointer

  std::string m_model_path = ArgumentParser::get_instance()
                                 .get_RL_model_path(); ///< Path to the RL model

  ///// --- ONNX Runtime inference components ---
  Ort::Env m_env{
      ORT_LOGGING_LEVEL_ERROR,
      "FringeEvalRLEnv"}; ///< ONNX Runtime environment for Fringe inference.
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

  /**
   * \brief Converts a set of KripkeState (Fringe) to a minimal GraphTensor
   * representation.
   *
   * This function packs the tensors of the given Fringe into one disjoint
   * graph, extracting only the essential information required for RL input.
   *
   * \param states The set of States to convert.
   * \return The packed graph; its batch field is the per-node membership.
   */
  [[nodiscard]] PackedGraph
  fringe_to_tensor_minimal(std::vector<State<StateRepr>> &states);

  /**
   * \brief Initializes the ONNX Runtime model for RL inference.
   *
   * Sets up the ONNX Runtime environment, session options, loads the RL model,
   * and prepares input/output names and memory information required for
   * inference. This function should be called before performing any inference
   * with the model.
   */
  void initialize_onnx_model();

  /** \brief Function that return the position of each state in the fringe wrt
   * to their score
   *
   * \param scores The Array of scores ordered as in input
   * \param n
   * \return THe array that associates to each position the rank of the states
   */
  std::vector<float> rankScores(const float *scores, size_t n) const;
};

#include "FringeEvalRL.tpp"
