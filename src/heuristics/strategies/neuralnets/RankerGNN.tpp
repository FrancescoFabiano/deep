/**
 * \file RankerGNN.tpp
 * \brief Implementation of RankerGNN (see RankerGNN.h).
 */
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

template <StateRepresentation StateRepr> RankerGNN<StateRepr>::RankerGNN() {
  const auto &parser = ArgumentParser::get_instance();
  const std::string &model = parser.get_ranker_model();
  const std::string &gpu_model = parser.get_ranker_model_gpu();
  read_sidecar(model + ".vocab");
  // With a GPU model the CPU session stays on the CPU whatever --onnx_device
  // says: small states are cheaper there than a GPU call.
  onnx_runtime::configure_session(m_options, gpu_model.empty() ? "" : "cpu");
  m_session = std::make_unique<Ort::Session>(onnx_runtime::env(), model.c_str(),
                                             m_options);
  if (!gpu_model.empty()) {
    onnx_runtime::configure_session(m_gpu_options, "cuda");
    m_gpu_session = std::make_unique<Ort::Session>(
        onnx_runtime::env(), gpu_model.c_str(), m_gpu_options);
    m_gpu_edges = parser.get_ranker_gpu_edges();
  }
  m_memory = std::make_unique<Ort::MemoryInfo>(
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
}

template <StateRepresentation StateRepr>
void RankerGNN<StateRepr>::set_root(const State<StateRepr> &root) {
  auto &td = TrainingDataset<StateRepr>::get_instance();
  for (const auto &fl : Domain::get_instance().get_positive_fluents()) {
    const std::string id = std::to_string(td.ranker_fluent_id(fl));
    if (!m_names.contains(id)) {
      m_name_order.push_back(id);
    }
    m_names[id] =
        HelperPrint::get_instance().get_grounder().deground_fluent(fl);
  }
  std::set<long long> labels;
  for (const auto &[from_pw, from_map] :
       root.get_representation().get_beliefs()) {
    for (const auto &[ag, to_set] : from_map) {
      if (!to_set.empty()) {
        labels.insert(static_cast<long long>(td.get_unique_a_id_from_map(ag)));
      }
    }
  }
  int rank = 0;
  for (const auto l : labels) {
    m_agents[std::to_string(l)] = rank++;
  }
  std::ostringstream st;
  st << "  " << td.get_epsilon_node_id_string() << " -> "
     << td.get_goal_parent_id_string() << " [label=\""
     << td.get_to_goal_edge_id_string() << "\"];\n"
     << td.get_goal_string();
  m_static_edges = parse_edges(st.str());
  // Goal operators are recorded while the goal string is written; a goal
  // without operator nodes (only fluents) leaves the map empty.
  if (m_ops) {
    for (const auto &[node, op] : td.get_goal_ops()) {
      m_goal_ops[node] = op;
    }
  }
  m_root_set = true;
}

template <StateRepresentation StateRepr>
int RankerGNN<StateRepr>::get_score(const State<StateRepr> &state) {
  if (!m_root_set) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::GNNInstanceError,
        "[RankerGNN] root state not set before scoring.");
  }
  Graph g = build_edges(edges(state));
  const float score = infer(g);
  const double h = (1000.0 - static_cast<double>(score)) * m_scale;
  const double lim = static_cast<double>(std::numeric_limits<int>::max() / 2);
  return static_cast<int>(std::lround(std::clamp(h, 0.0, lim)));
}

template <StateRepresentation StateRepr>
std::string RankerGNN<StateRepr>::unquote(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), '"'), s.end());
  return s;
}

template <StateRepresentation StateRepr>
bool RankerGNN<StateRepr>::is_world(const std::string &t) {
  std::string d = t[0] == '-' ? t.substr(1) : t;
  if (d.empty() || !std::all_of(d.begin(), d.end(), ::isdigit)) {
    return false;
  }
  d.erase(0, std::min(d.find_first_not_of('0'), d.size() - 1));
  return d.size() > 7 || (d.size() == 7 && d > "1000000");
}

template <StateRepresentation StateRepr>
std::vector<typename RankerGNN<StateRepr>::Edge>
RankerGNN<StateRepr>::parse_edges(const std::string &dot) {
  std::vector<Edge> out;
  std::istringstream in(dot);
  std::string line;
  while (std::getline(in, line)) {
    const auto arrow = line.find("->");
    const auto lab = line.find("label");
    if (arrow == std::string::npos || lab == std::string::npos) {
      continue;
    }
    std::istringstream lu(line.substr(0, arrow)),
        lv(line.substr(arrow + 2, line.find('[') - arrow - 2));
    std::string u, v;
    lu >> u;
    lv >> v;
    const auto eq = line.find('=', lab);
    std::string l;
    for (auto i = eq + 1; i < line.size(); ++i) {
      if (line[i] == '"' || line[i] == ' ') {
        if (!l.empty() && line[i] == '"') {
          break;
        }
        continue;
      }
      if (line[i] == ']' || line[i] == ';') {
        break;
      }
      l += line[i];
    }
    out.push_back({unquote(u), unquote(v), l});
  }
  return out;
}

template <StateRepresentation StateRepr>
void RankerGNN<StateRepr>::read_sidecar(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::GNNFileError,
                                   "[RankerGNN] missing " + path);
  }
  std::string k;
  long long v;
  while (in >> k >> v) {
    if (k == "#anon") {
      m_anon = v != 0;
    } else if (k == "#drop_holds") {
      m_drop_holds = v != 0;
    } else if (k == "#ops") {
      m_ops = v != 0;
    } else if (k == "#scale") {
      m_scale = static_cast<double>(v);
    } else if (k[0] == '#') {
      if (v != 0) {
        ExitHandler::exit_with_message(ExitHandler::ExitCode::GNNFileError,
                                       "[RankerGNN] " + path +
                                           ": unsupported model option " + k);
      }
    } else {
      m_vocab[k] = v;
    }
  }
  m_next_id = 0;
  for (const auto &[key, id] : m_vocab) {
    m_next_id = std::max<int64_t>(m_next_id, id + 1);
  }
}

template <StateRepresentation StateRepr>
int64_t RankerGNN<StateRepr>::vid(const std::string &key) {
  if (m_anon && key.rfind("pred:", 0) == 0) {
    return vid("fluent");
  }
  if (auto it = m_vocab.find(key); it != m_vocab.end()) {
    return it->second;
  }
  return m_vocab[key] = m_next_id++;
}

template <StateRepresentation StateRepr>
std::vector<std::string> RankerGNN<StateRepr>::split_us(const std::string &s) {
  std::vector<std::string> out;
  std::string cur;
  for (char ch : s) {
    if (ch == '_') {
      out.push_back(cur);
      cur.clear();
    } else {
      cur += ch;
    }
  }
  out.push_back(cur);
  return out;
}

template <StateRepresentation StateRepr>
std::vector<typename RankerGNN<StateRepr>::Edge>
RankerGNN<StateRepr>::edges(const State<StateRepr> &state) {
  const auto &ks = state.get_representation();
  auto &td = TrainingDataset<StateRepr>::get_instance();
  std::vector<Edge> E = m_static_edges;
  auto str = [](const auto &x) {
    std::ostringstream o;
    o << x;
    return o.str();
  };
  const std::string eps = str(td.get_epsilon_node_id_string()),
                    ts = str(td.get_to_state_edge_id_string());
  for (const auto &dw : ks.get_designated_worlds()) {
    E.push_back({eps, std::to_string(dw.get_id_casted()), ts});
  }
  for (const auto &[from_pw, from_map] : ks.get_beliefs()) {
    const std::string f = std::to_string(from_pw.get_id_casted());
    for (const auto &[ag, to_set] : from_map) {
      const std::string l = str(td.get_unique_a_id_from_map(ag));
      for (const auto &to_pw : to_set) {
        E.push_back({f, std::to_string(to_pw.get_id_casted()), l});
      }
    }
  }
  for (const auto &pw : ks.get_worlds()) {
    const std::string w = std::to_string(pw.get_id_casted());
    for (const auto &fl : pw.get_fluent_set()) {
      if (!FormulaHelper::is_negated(fl)) {
        E.push_back({w, str(td.ranker_fluent_id(fl)), "4"});
      }
    }
  }
  return E;
}

template <StateRepresentation StateRepr>
typename RankerGNN<StateRepr>::Graph
RankerGNN<StateRepr>::build_edges(const std::vector<Edge> &E) {
  static const std::vector<std::string> ET = {
      "B0",     "B1",      "B2",    "B3",   "B4", "B5", "B6", "B7",
      "togoal", "tostate", "holds", "goal", "A0", "A1", "A2", "A3"};
  std::set<std::string> pointed;
  for (const auto &e : E) {
    if (e.l == "3") {
      pointed.insert(e.v);
    }
  }
  std::vector<std::string> nodes;
  std::unordered_map<std::string, int64_t> ix;
  auto push = [&](const std::string &n) {
    if (!ix.contains(n)) {
      ix[n] = static_cast<int64_t>(nodes.size());
      nodes.push_back(n);
    }
  };
  for (const auto &e : E) {
    push(e.u);
    push(e.v);
  }
  for (const auto &id : m_name_order) {
    push(id);
  }
  std::set<std::string> objs;
  for (const auto &id : m_name_order) {
    auto p = split_us(m_names[id]);
    for (size_t i = 1; i < p.size(); ++i) {
      objs.insert(p[i]);
    }
  }
  for (const auto &o : objs) {
    push("obj:" + o);
  }
  Graph g;
  for (const auto &n : nodes) {
    std::string t;
    if (n.rfind("obj:", 0) == 0) {
      t = "obj";
    } else if (is_world(n)) {
      t = pointed.contains(n) ? "P" : "W";
    } else if (n == "0") {
      t = "eps";
    } else if (n == "1") {
      t = "root";
    } else if (m_names.contains(n)) {
      t = "pred:" + split_us(m_names[n])[0];
    } else if (m_agents.contains(n)) {
      t = "agent";
    } else if (const auto o = m_goal_ops.find(n); o != m_goal_ops.end()) {
      t = "gop:" + o->second; // only with #ops 1
    } else {
      t = "gop";
    }
    g.x.push_back(vid(t));
    g.pmask.push_back(pointed.contains(n) ? 1.0f : 0.0f);
  }
  auto et = [&](const std::string &t) {
    return static_cast<int64_t>(std::find(ET.begin(), ET.end(), t) -
                                ET.begin());
  };
  auto add = [&](const std::string &u, const std::string &v,
                 const std::string &t) {
    const int64_t k = et(t);
    if (m_drop_holds && t == "holds") {
      return;
    }
    g.src.push_back(ix[u]);
    g.src.push_back(ix[v]);
    g.dst.push_back(ix[v]);
    g.dst.push_back(ix[u]);
    g.typ.push_back(2 * k);
    g.typ.push_back(2 * k + 1);
  };
  for (const auto &e : E) {
    if (is_world(e.u) && is_world(e.v)) {
      const auto a = m_agents.find(e.l);
      add(e.u, e.v,
          "B" +
              std::to_string(std::min(a == m_agents.end() ? 7 : a->second, 7)));
    } else if (e.l == "2") {
      add(e.u, e.v, "togoal");
    } else if (e.l == "3") {
      add(e.u, e.v, "tostate");
    } else if (e.l == "4") {
      add(e.u, e.v, "holds");
    } else {
      add(e.u, e.v, "goal");
    }
  }
  for (const auto &id : m_name_order) {
    auto p = split_us(m_names[id]);
    for (size_t i = 1; i < p.size(); ++i) {
      add(id, "obj:" + p[i], "A" + std::to_string(std::min<size_t>(i - 1, 3)));
    }
  }
  return g;
}

template <StateRepresentation StateRepr>
float RankerGNN<StateRepr>::infer(Graph &g) {
  const int64_t N = static_cast<int64_t>(g.x.size()),
                Ecount = static_cast<int64_t>(g.src.size());
  std::vector<int64_t> ei(g.src);
  ei.insert(ei.end(), g.dst.begin(), g.dst.end());
  const int64_t sx[] = {N}, se[] = {2, Ecount}, sa[] = {Ecount}, sp[] = {N};
  std::vector<Ort::Value> in;
  in.push_back(Ort::Value::CreateTensor<int64_t>(*m_memory, g.x.data(),
                                                 g.x.size(), sx, 1));
  in.push_back(Ort::Value::CreateTensor<int64_t>(*m_memory, ei.data(),
                                                 ei.size(), se, 2));
  in.push_back(Ort::Value::CreateTensor<int64_t>(*m_memory, g.typ.data(),
                                                 g.typ.size(), sa, 1));
  in.push_back(Ort::Value::CreateTensor<float>(*m_memory, g.pmask.data(),
                                               g.pmask.size(), sp, 1));
  const char *names_in[] = {"x", "edge_index", "edge_attr", "pmask"};
  const char *names_out[] = {"score"};
  auto &session =
      m_gpu_session && Ecount >= m_gpu_edges ? m_gpu_session : m_session;
  auto out = session->Run(Ort::RunOptions{nullptr}, names_in, in.data(),
                          in.size(), names_out, 1);
  return out[0].GetTensorData<float>()[0];
}
