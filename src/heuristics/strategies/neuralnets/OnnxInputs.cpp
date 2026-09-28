#include "OnnxInputs.h"

PackedGraph OnnxInputs::pack(const GraphTensor &graph, const bool bitmask,
                             const size_t bitmask_size) {
  return pack(std::vector<const GraphTensor *>{&graph}, bitmask, bitmask_size);
}

PackedGraph OnnxInputs::pack(const std::vector<const GraphTensor *> &graphs,
                             const bool bitmask, const size_t bitmask_size) {
  PackedGraph packed;

  size_t num_edges = 0;
  for (const auto *graph : graphs) {
    num_edges += graph->edge_src.size();
  }
  std::vector<int64_t> edge_dst;
  packed.edge_index.reserve(2 * num_edges);
  edge_dst.reserve(num_edges);
  packed.edge_attrs.reserve(num_edges);

  int64_t graph_index = 0;
  for (const auto *graph : graphs) {
    const auto offset = static_cast<int64_t>(packed.num_nodes);

    size_t graph_nodes;
    if (bitmask) {
      graph_nodes = graph->real_node_ids_bitmask.size() / bitmask_size;
      packed.node_bits.insert(packed.node_bits.end(),
                              graph->real_node_ids_bitmask.begin(),
                              graph->real_node_ids_bitmask.end());
    } else {
      graph_nodes = graph->real_node_ids.size();
      packed.node_ids.insert(packed.node_ids.end(),
                             graph->real_node_ids.begin(),
                             graph->real_node_ids.end());
    }

    for (size_t e = 0; e < graph->edge_src.size(); ++e) {
      packed.edge_index.push_back(graph->edge_src[e] + offset);
      edge_dst.push_back(graph->edge_dst[e] + offset);
    }
    packed.edge_attrs.insert(packed.edge_attrs.end(), graph->edge_attrs.begin(),
                             graph->edge_attrs.end());

    for (const auto pointed_id : graph->pointed_ids) {
      packed.pointed_ids.push_back(pointed_id + offset);
    }

    packed.batch.insert(packed.batch.end(), graph_nodes, graph_index);
    packed.num_nodes += graph_nodes;
    ++graph_index;
  }

  packed.edge_index.insert(packed.edge_index.end(), edge_dst.begin(),
                           edge_dst.end());
  return packed;
}

void OnnxInputs::add_graph(const PackedGraph &graph, const bool bitmask,
                           const size_t bitmask_size, const bool edge_attr_2d,
                           const bool with_pointed) {
  const auto num_nodes = static_cast<int64_t>(graph.num_nodes);
  const auto num_edges = static_cast<int64_t>(graph.edge_attrs.size());

  if (bitmask) {
    add_view(graph.node_bits, {num_nodes, static_cast<int64_t>(bitmask_size)});
  } else {
    add_view(graph.node_ids, {num_nodes});
  }
  add_view(graph.edge_index, {2, num_edges});
  if (edge_attr_2d) {
    add_view(graph.edge_attrs, {num_edges, 1});
  } else {
    add_view(graph.edge_attrs, {num_edges});
  }
  add_view(graph.batch, {num_nodes});
  if (with_pointed) {
    add_view(graph.pointed_ids,
             {static_cast<int64_t>(graph.pointed_ids.size())});
  }
}

void OnnxInputs::add_owned(std::vector<uint8_t> data) {
  const auto size = static_cast<int64_t>(data.size());
  add_view(m_owned_uint8.emplace_back(std::move(data)), {size});
}

std::vector<Ort::Value>
OnnxInputs::run(Ort::Session &session,
                const std::vector<std::string> &input_names,
                const std::vector<std::string> &output_names,
                const ExitHandler::ExitCode mismatch_code) {
  if (m_values.size() != input_names.size()) {
    ExitHandler::exit_with_message(
        mismatch_code,
        "ONNX input count mismatch: model expects " +
            std::to_string(input_names.size()) +
            " input tensors but C++ prepared " +
            std::to_string(m_values.size()) +
            " (check that the model was exported for this dataset "
            "type/mode and takes pointed_ids; see "
            "docs/python_followups.md).");
  }

  std::vector<const char *> input_names_cstr;
  input_names_cstr.reserve(input_names.size());
  for (const auto &name : input_names) {
    input_names_cstr.push_back(name.c_str());
  }
  std::vector<const char *> output_names_cstr;
  output_names_cstr.reserve(output_names.size());
  for (const auto &name : output_names) {
    output_names_cstr.push_back(name.c_str());
  }

  return session.Run(Ort::RunOptions{nullptr}, input_names_cstr.data(),
                     m_values.data(), m_values.size(), output_names_cstr.data(),
                     output_names_cstr.size());
}

void OnnxInputs::add_view(const std::vector<int64_t> &data,
                          std::vector<int64_t> shape) {
  const auto &stored_shape = m_shapes.emplace_back(std::move(shape));
  m_values.emplace_back(Ort::Value::CreateTensor<int64_t>(
      m_memory_info, const_cast<int64_t *>(data.data()), data.size(),
      stored_shape.data(), stored_shape.size()));
}

void OnnxInputs::add_view(const std::vector<uint8_t> &data,
                          std::vector<int64_t> shape) {
  const auto &stored_shape = m_shapes.emplace_back(std::move(shape));
  m_values.emplace_back(Ort::Value::CreateTensor<uint8_t>(
      m_memory_info, const_cast<uint8_t *>(data.data()), data.size(),
      stored_shape.data(), stored_shape.size()));
}
