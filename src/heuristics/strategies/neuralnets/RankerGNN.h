#pragma once
/**
 * \file RankerGNN.h
 * \brief Learned sibling ranker used as the GNN heuristic (--ranker_model).
 *
 * The network scores an epistemic state; a higher score means closer to the
 * goal. The state is read as a graph built directly from the Kripke structure,
 * with the rules of the training code (lib/gnn_epddl, encoding.py): the goal
 * tree (its operator nodes typed by operator when the model was trained with
 * them), an edge to every designated world, the belief edges (typed by agent
 * rank), an edge from every world to each positive fluent true in it, and
 * edges from every fluent to its arguments. Node types come from the sidecar
 * <model>.vocab ("key id" lines, plus "#anon", "#drop_holds", "#ops" and
 * "#scale"). Model inputs: x [N] int64, edge_index [2,E] int64, edge_attr [E]
 * int64, pmask [N] float; output: score. The heuristic value is
 * h = round(scale * (1000 - score)), clamped to >= 0 because HFS drops
 * negative values; the map keeps the model's order.
 *
 * With --ranker_model_gpu the same network runs in two sessions: states with
 * at least --ranker_gpu_edges edges on CUDA (the standard export), smaller
 * ones on the CPU (the export for the CPU). Both exports compute the same
 * scores up to floating-point rounding.
 */
#include "ArgumentParser.h"
#include "Domain.h"
#include "FormulaHelper.h"
#include "HelperPrint.h"
#include "State.h"
#include "neuralnets/OnnxInputs.h"
#include "neuralnets/TrainingDataset.h"
#include "utilities/ExitHandler.h"
#include <cstdint>
#include <map>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>
#include <vector>

template <StateRepresentation StateRepr> class RankerGNN {
public:
  /** \brief True when a ranker model was given (--ranker_model). */
  static bool enabled() {
    return !ArgumentParser::get_instance().get_ranker_model().empty();
  }

  static RankerGNN &get_instance() {
    static RankerGNN instance;
    return instance;
  }

  /**
   * \brief Reads the per-problem parts of the graph from the initial state:
   * fluent names, agent ranks (agents in the order of their ids, among those
   * with a belief edge), the goal tree and its operators.
   */
  void set_root(const State<StateRepr> &root);

  /** \brief Heuristic value of \p state (lower is better). */
  int get_score(const State<StateRepr> &state);

private:
  struct Edge {
    std::string u, v, l;
  };
  struct Graph {
    std::vector<int64_t> x, src, dst, typ;
    std::vector<float> pmask;
  };

  RankerGNN();

  static std::string unquote(std::string s);

  /// World nodes are the hashed world ids: abs(id) > 10^6.
  static bool is_world(const std::string &t);

  /// Edges "u -> v [label=l]" of a DOT text (used for the goal tree).
  static std::vector<Edge> parse_edges(const std::string &dot);

  void read_sidecar(const std::string &path);

  /// Node-type id; an unseen type gets the next id (untrained embedding).
  int64_t vid(const std::string &key);

  static std::vector<std::string> split_us(const std::string &s);

  /// The state's edges in the order of the dataset DOT writer
  /// (HelperPrint::print_dataset_format with --ranker_encoding): goal tree,
  /// designated worlds, belief edges, holds edges (positive fluents).
  std::vector<Edge> edges(const State<StateRepr> &state);

  /// The model's input from an edge list (encoding.py, line for line).
  Graph build_edges(const std::vector<Edge> &E);

  /// Scores \p g with the GPU session when it has at least
  /// --ranker_gpu_edges edges and that session exists, else on the CPU one.
  float infer(Graph &g);

  std::map<std::string, int64_t> m_vocab;
  int64_t m_next_id = 0;
  bool m_anon = false, m_drop_holds = false, m_ops = false, m_root_set = false;
  double m_scale = 1000.0;
  std::unordered_map<std::string, std::string> m_names;
  std::vector<std::string> m_name_order;
  std::unordered_map<std::string, int> m_agents;
  std::unordered_map<std::string, std::string> m_goal_ops; ///< node -> op
  std::vector<Edge> m_static_edges;
  Ort::SessionOptions m_options, m_gpu_options;
  std::unique_ptr<Ort::Session> m_session, m_gpu_session;
  std::unique_ptr<Ort::MemoryInfo> m_memory;
  int64_t m_gpu_edges = 0;
};

#include "RankerGNN.tpp"
