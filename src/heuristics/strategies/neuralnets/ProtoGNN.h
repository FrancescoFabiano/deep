#pragma once
/*
 * PROTOTYPE (not for commit): heuristic of the new-representation models (epistemic-work/newm train3/train4).
 * Enabled with -u GNN and DEEP_PROTO_MODEL=<model.onnx> (plus DEEP_PROTO_REPR=holds,names, which makes
 * print_dataset_format write the world-to-fluent edges and fluent_names.csv).
 * The graph is built from the state's dataset DOT text with the same rules as newm/build2.py graph()
 * (one writer for training and inference). Sidecar <model.onnx>.vocab: "key id" lines (node types) and
 * "#anon 0|1", "#drop_holds 0|1", "#scale <s>". Model inputs: x [N] int64, edge_index [2,E] int64,
 * edge_attr [E] int64, pmask [N] float; output: score (higher = closer to the goal); h = round(scale * (1000 - score)), clamped to >= 0.
 * DEEP_PROTO_DUMP=<dir>: the first 200 scored states are written as <k>.dot and <k>.txt (tensors and score).
 * DEEP_PROTO_DIRECT=1: build the edge list directly from the Kripke structure instead of writing and parsing the DOT
 * text (same edges, same identifiers, same order; the goal part is taken once from the writer's goal string).
 * DEEP_PROTO_DIRECT_CHECK=1: compute both paths for every state, score with the text path, and report the largest
 * score difference and any edge-list mismatch every 25 states (for verification only).
 */
#include "State.h"
#include "FormulaHelper.h"
#include "neuralnets/TrainingDataset.h"
#include "neuralnets/OnnxInputs.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <onnxruntime_cxx_api.h>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <unistd.h>

template <StateRepresentation StateRepr> class ProtoGNN {
public:
  static bool enabled() { return std::getenv("DEEP_PROTO_MODEL") != nullptr; }
  static ProtoGNN &get_instance() {
    static ProtoGNN instance;
    return instance;
  }

  /// Agent ranks and fluent names come from the initial state, as in the Python scorer (search/run.py).
  void set_root(const State<StateRepr> &root) {
    // fluent names: written by print_dataset_format (DEEP_PROTO_REPR=...names) to a per-process file, because
    // parallel runs (e.g. the IEPC runner) share one working directory
    const std::string names = (std::filesystem::temp_directory_path() / ("proto_names_" + std::to_string(::getpid()) + ".csv")).string();
    ::setenv("DEEP_PROTO_NAMES_FILE", names.c_str(), 1);
    const std::string root_dot = to_dot(root);
    read_names(names);
    if (m_dump_dir.size()) std::filesystem::copy_file(names, m_dump_dir + "/fluent_names.csv", std::filesystem::copy_options::overwrite_existing);
    std::filesystem::remove(names);
    std::set<long long> labels;
    for (const auto &e : parse_edges(root_dot)) {
      if (is_world(e.u) && is_world(e.v)) labels.insert(std::stoll(e.l));
    }
    {   // static part of every state's edge list: eps -> goal root, then the goal tree
      auto &td = TrainingDataset<StateRepr>::get_instance();
      std::ostringstream st;
      st << "  " << td.get_epsilon_node_id_string() << " -> " << td.get_goal_parent_id_string() << " [label=\""
         << td.get_to_goal_edge_id_string() << "\"];\n" << td.get_goal_string();
      m_static_edges = parse_edges(st.str());
    }
    int rank = 0;
    for (const auto l : labels) m_agents[std::to_string(l)] = rank++;
    m_root_set = true;
  }

  int get_score(const State<StateRepr> &state) {
    if (!m_root_set) {
      ExitHandler::exit_with_message(ExitHandler::ExitCode::GNNInstanceError,
                                     "[ProtoGNN] root state not set before scoring.");
    }
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    std::string dot;
    Graph g;
    if (m_direct && !m_direct_check) {
      g = build_edges(direct_edges(state));
    } else {
      dot = to_dot(state);
      const auto parsed = parse_edges(dot);
      g = build_edges(parsed);
      if (m_direct_check) check_direct(state, parsed, g);
    }
    const auto t1 = clk::now();
    const auto t2 = clk::now();
    const float score = infer(g);
    const auto t3 = clk::now();
    m_t_dot += std::chrono::duration<double>(t1 - t0).count(); m_t_build += std::chrono::duration<double>(t2 - t1).count();   // dot = text or direct edge list
    m_t_infer += std::chrono::duration<double>(t3 - t2).count(); ++m_scored;
    if (m_log && m_scored % 25 == 0) {
      std::cerr << "[ProtoGNN] scored " << m_scored << " | last N=" << g.x.size() << " E=" << g.typ.size()
                << " | total s: dot " << m_t_dot << " build " << m_t_build << " infer " << m_t_infer << std::endl;
    }
    if (m_dump_dir.size() && m_dumped < 200 && !dot.empty()) dump(dot, g, score);
    // HeuristicFirst drops successors with h < 0, so the order-preserving map is shifted: h = scale * (1000 - score) >= 0
    const double h = (1000.0 - static_cast<double>(score)) * m_scale;
    const double lim = static_cast<double>(std::numeric_limits<int>::max() / 2);
    return static_cast<int>(std::lround(std::clamp(h, 0.0, lim)));
  }

private:
  struct Edge { std::string u, v, l; };
  struct Graph { std::vector<int64_t> x, src, dst, typ; std::vector<float> pmask; std::vector<int64_t> pointed; };

  ProtoGNN() {
    const std::string model = std::getenv("DEEP_PROTO_MODEL");
    m_tmp = (std::filesystem::temp_directory_path() / ("proto_state_" + std::to_string(::getpid()) + ".dot")).string();
    read_sidecar(model + ".vocab");
    m_log = std::getenv("DEEP_PROTO_LOG") != nullptr;
    m_direct = std::getenv("DEEP_PROTO_DIRECT") != nullptr;
    m_direct_check = std::getenv("DEEP_PROTO_DIRECT_CHECK") != nullptr;
    if (const char *d = std::getenv("DEEP_PROTO_DUMP")) {
      m_dump_dir = d;
      std::filesystem::create_directories(m_dump_dir);
    }
    onnx_runtime::configure_session(m_options);
    m_session = std::make_unique<Ort::Session>(onnx_runtime::env(), model.c_str(), m_options);
    m_memory = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
  }

  /// print_dataset_format writes to an ofstream only: a per-process temp file, read back.
  std::string to_dot(const State<StateRepr> &state) {
    {
      std::ofstream ofs(m_tmp);
      state.print_dataset_format(ofs);
    }
    std::stringstream ss;
    {
      std::ifstream in(m_tmp);
      ss << in.rdbuf();
    }
    std::filesystem::remove(m_tmp);   // no leftovers in the temp directory (runs killed at a limit included)
    return ss.str();
  }

  static std::string unquote(std::string s) {
    s.erase(std::remove(s.begin(), s.end(), '"'), s.end());
    return s;
  }
  /// build2.py world(): abs(int(t)) > 10**6
  static bool is_world(const std::string &t) {
    std::string d = t[0] == '-' ? t.substr(1) : t;
    if (d.empty() || !std::all_of(d.begin(), d.end(), ::isdigit)) return false;
    d.erase(0, std::min(d.find_first_not_of('0'), d.size() - 1));
    return d.size() > 7 || (d.size() == 7 && d > "1000000");
  }
  /// build2.py EDGE regex: u -> v [label="l"]
  static std::vector<Edge> parse_edges(const std::string &dot) {
    std::vector<Edge> out;
    std::istringstream in(dot);
    std::string line;
    while (std::getline(in, line)) {
      const auto arrow = line.find("->");
      const auto lab = line.find("label");
      if (arrow == std::string::npos || lab == std::string::npos) continue;
      std::istringstream lu(line.substr(0, arrow)), lv(line.substr(arrow + 2, line.find('[') - arrow - 2));
      std::string u, v;
      lu >> u; lv >> v;
      const auto eq = line.find('=', lab);
      std::string l;
      for (auto i = eq + 1; i < line.size(); ++i) {
        if (line[i] == '"' || line[i] == ' ') { if (!l.empty() && line[i] == '"') break; continue; }
        if (line[i] == ']' || line[i] == ';') break;
        l += line[i];
      }
      out.push_back({unquote(u), unquote(v), l});
    }
    return out;
  }

  void read_names(const std::string &path) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);   // header
    while (std::getline(in, line)) {
      const auto c = line.find(',');
      if (c == std::string::npos) continue;
      const std::string id = line.substr(0, c), name = unquote(line.substr(c + 1));
      if (!m_names.count(id)) m_name_order.push_back(id);
      m_names[id] = name;
    }
  }

  void read_sidecar(const std::string &path) {
    std::ifstream in(path);
    if (!in) ExitHandler::exit_with_message(ExitHandler::ExitCode::GNNFileError, "[ProtoGNN] missing " + path);
    std::string k;
    long long v;
    while (in >> k >> v) {
      if (k == "#anon") m_anon = v != 0;
      else if (k == "#drop_holds") m_drop_holds = v != 0;
      else if (k == "#scale") m_scale = static_cast<double>(v);
      else m_vocab[k] = v;
    }
    m_next_id = 0;
    for (const auto &[key, id] : m_vocab) m_next_id = std::max<int64_t>(m_next_id, id + 1);
  }

  int64_t vid(const std::string &key) {
    if (m_anon && key.rfind("pred:", 0) == 0) return vid("fluent");
    if (auto it = m_vocab.find(key); it != m_vocab.end()) return it->second;
    return m_vocab[key] = m_next_id++;   // build2.vid: unseen key -> next id (untrained embedding)
  }

  static std::vector<std::string> split_us(const std::string &s) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) { if (ch == '_') { out.push_back(cur); cur.clear(); } else cur += ch; }
    out.push_back(cur);
    return out;
  }

  /// The same edge list print_dataset_format writes for a merged HASHED state, without the text round trip:
  /// static goal part, eps -> designated worlds, belief edges, holds edges (positive fluents), in the writer's order.
  std::vector<Edge> direct_edges(const State<StateRepr> &state) {
    const auto &ks = state.get_representation();
    auto &td = TrainingDataset<StateRepr>::get_instance();
    std::vector<Edge> E = m_static_edges;
    auto str = [](const auto &x) { std::ostringstream o; o << x; return o.str(); };
    const std::string eps = str(td.get_epsilon_node_id_string()), ts = str(td.get_to_state_edge_id_string());
    for (const auto &dw : ks.get_designated_worlds()) E.push_back({eps, std::to_string(dw.get_id_casted()), ts});
    for (const auto &[from_pw, from_map] : ks.get_beliefs()) {
      const std::string f = std::to_string(from_pw.get_id_casted());
      for (const auto &[ag, to_set] : from_map) {
        const std::string l = str(td.get_unique_a_id_from_map(ag));
        for (const auto &to_pw : to_set) E.push_back({f, std::to_string(to_pw.get_id_casted()), l});
      }
    }
    for (const auto &pw : ks.get_worlds()) {
      const std::string w = std::to_string(pw.get_id_casted());
      for (const auto &fl : pw.get_fluent_set())
        if (!FormulaHelper::is_negated(fl)) E.push_back({w, str(td.proto_fluent_id(fl)), "4"});
    }
    return E;
  }

  void check_direct(const State<StateRepr> &state, const std::vector<Edge> &parsed, Graph &g_text) {
    const auto E = direct_edges(state);
    bool same = E.size() == parsed.size();
    for (size_t i = 0; same && i < E.size(); ++i) same = E[i].u == parsed[i].u && E[i].v == parsed[i].v && E[i].l == parsed[i].l;
    Graph g_dir = build_edges(E);
    const double diff = std::abs(static_cast<double>(infer(g_dir)) - static_cast<double>(infer(g_text)));
    m_check_max = std::max(m_check_max, diff); m_check_mismatch += !same; ++m_check_n;
    if (m_check_n % 25 == 0)
      std::cerr << "[ProtoGNN direct check] states " << m_check_n << ", edge-list mismatches " << m_check_mismatch
                << ", max |score direct - text| " << m_check_max << std::endl;
  }

  Graph build(const std::string &dot) { return build_edges(parse_edges(dot)); }

  /// newm/build2.py graph(), line for line, from an edge list.
  Graph build_edges(const std::vector<Edge> &E) {
    static const std::vector<std::string> ET = {"B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "togoal", "tostate", "holds", "goal", "A0", "A1", "A2", "A3"};
    std::set<std::string> pointed;
    for (const auto &e : E) if (e.l == "3") pointed.insert(e.v);
    std::vector<std::string> nodes;
    std::unordered_map<std::string, int64_t> ix;
    auto push = [&](const std::string &n) { if (!ix.count(n)) { ix[n] = static_cast<int64_t>(nodes.size()); nodes.push_back(n); } };
    for (const auto &e : E) { push(e.u); push(e.v); }
    for (const auto &id : m_name_order) push(id);
    std::set<std::string> objs;
    for (const auto &id : m_name_order) { auto p = split_us(m_names[id]); for (size_t i = 1; i < p.size(); ++i) objs.insert(p[i]); }
    for (const auto &o : objs) push("obj:" + o);
    Graph g;
    for (const auto &n : nodes) {
      std::string t;
      if (n.rfind("obj:", 0) == 0) t = "obj";
      else if (is_world(n)) t = pointed.count(n) ? "P" : "W";
      else if (n == "0") t = "eps";
      else if (n == "1") t = "root";
      else if (m_names.count(n)) t = "pred:" + split_us(m_names[n])[0];
      else if (m_agents.count(n)) t = "agent";
      else t = "gop";
      g.x.push_back(vid(t));
      g.pmask.push_back(pointed.count(n) ? 1.0f : 0.0f);
    }
    auto et = [&](const std::string &t) { return static_cast<int64_t>(std::find(ET.begin(), ET.end(), t) - ET.begin()); };
    auto add = [&](const std::string &u, const std::string &v, const std::string &t) {
      const int64_t k = et(t);
      if (m_drop_holds && t == "holds") return;
      g.src.push_back(ix[u]); g.src.push_back(ix[v]);
      g.dst.push_back(ix[v]); g.dst.push_back(ix[u]);
      g.typ.push_back(2 * k); g.typ.push_back(2 * k + 1);
    };
    for (const auto &e : E) {
      if (is_world(e.u) && is_world(e.v)) {
        const auto a = m_agents.find(e.l);
        add(e.u, e.v, "B" + std::to_string(std::min(a == m_agents.end() ? 7 : a->second, 7)));
      } else if (e.l == "2") add(e.u, e.v, "togoal");
      else if (e.l == "3") add(e.u, e.v, "tostate");
      else if (e.l == "4") add(e.u, e.v, "holds");
      else add(e.u, e.v, "goal");
    }
    for (const auto &id : m_name_order) {
      auto p = split_us(m_names[id]);
      for (size_t i = 1; i < p.size(); ++i) add(id, "obj:" + p[i], "A" + std::to_string(std::min<size_t>(i - 1, 3)));
    }
    for (const auto &p : pointed) g.pointed.push_back(ix[p]);
    return g;
  }

  float infer(Graph &g) {
    const int64_t N = static_cast<int64_t>(g.x.size()), Ecount = static_cast<int64_t>(g.src.size());
    std::vector<int64_t> ei(g.src);
    ei.insert(ei.end(), g.dst.begin(), g.dst.end());
    const int64_t sx[] = {N}, se[] = {2, Ecount}, sa[] = {Ecount}, sp[] = {N};
    std::vector<Ort::Value> in;
    in.push_back(Ort::Value::CreateTensor<int64_t>(*m_memory, g.x.data(), g.x.size(), sx, 1));
    in.push_back(Ort::Value::CreateTensor<int64_t>(*m_memory, ei.data(), ei.size(), se, 2));
    in.push_back(Ort::Value::CreateTensor<int64_t>(*m_memory, g.typ.data(), g.typ.size(), sa, 1));
    in.push_back(Ort::Value::CreateTensor<float>(*m_memory, g.pmask.data(), g.pmask.size(), sp, 1));
    const char *names_in[] = {"x", "edge_index", "edge_attr", "pmask"};
    const char *names_out[] = {"score"};
    auto out = m_session->Run(Ort::RunOptions{nullptr}, names_in, in.data(), in.size(), names_out, 1);
    return out[0].GetTensorData<float>()[0];
  }

  void dump(const std::string &dot, const Graph &g, float score) {
    const std::string base = m_dump_dir + "/" + std::to_string(m_dumped++);
    std::ofstream(base + ".dot") << dot;
    std::ofstream t(base + ".txt");
    auto row = [&](const char *name, const auto &v) { t << name; for (auto x : v) t << ' ' << x; t << '\n'; };
    row("x", g.x); row("src", g.src); row("dst", g.dst); row("typ", g.typ); row("pointed", g.pointed);
    t << "score " << score << '\n';
    t << "agents"; for (const auto &[l, r] : m_agents) t << ' ' << l << ':' << r; t << '\n';
  }

  std::map<std::string, int64_t> m_vocab;
  int64_t m_next_id = 0;
  bool m_anon = false, m_drop_holds = false, m_root_set = false;
  double m_scale = 1000.0;
  std::unordered_map<std::string, std::string> m_names;
  std::vector<std::string> m_name_order;
  std::unordered_map<std::string, int> m_agents;
  std::string m_dump_dir, m_tmp;
  int m_dumped = 0;
  long m_scored = 0;
  bool m_log = false, m_direct = false, m_direct_check = false;
  std::vector<Edge> m_static_edges;
  long m_check_n = 0, m_check_mismatch = 0;
  double m_check_max = 0;
  double m_t_dot = 0, m_t_build = 0, m_t_infer = 0;
  Ort::SessionOptions m_options;
  std::unique_ptr<Ort::Session> m_session;
  std::unique_ptr<Ort::MemoryInfo> m_memory;
};
