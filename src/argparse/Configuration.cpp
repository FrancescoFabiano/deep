/**
 * \brief Implementation of \ref Configuration.h.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 30, 2025
 */

#include "Configuration.h"
#include <algorithm>

#include "ArgumentParser.h"
#include "ExitHandler.h"

// Helper to trim trailing and leading spaces
static std::string trim(const std::string &s) {
  const auto start = s.find_first_not_of(" \t\r\n");
  const auto end = s.find_last_not_of(" \t\r\n");
  if (start == std::string::npos)
    return "";
  return s.substr(start, end - start + 1);
}

// Helper to convert string to bool
static bool str_to_bool(const std::string &s) {
  std::string val = trim(s);
  std::ranges::transform(val, val.begin(), ::tolower);
  return (val == "1" || val == "true" || val == "yes" || val == "on");
}

// Remove this:
// thread_local Configuration* Configuration::instance = nullptr;

thread_local bool Configuration::m_initialized = false;

void Configuration::create_instance() {
  // This will initialize the thread-local singleton instance on first call per
  // thread.
  (void)get_instance();
}

Configuration &Configuration::get_instance() {
  thread_local Configuration instance;
  if (!m_initialized) {
    // Copy values from ArgumentParser singleton using setters
    const ArgumentParser &parser = ArgumentParser::get_instance();
    instance.set_bisimulation(parser.get_bisimulation());
    instance.set_bisimulation_type(parser.get_bisimulation_type());
    instance.set_check_visited(parser.get_check_visited());
    instance.set_search_strategy(parser.get_search_strategy());
    instance.set_heuristic_opt(parser.get_heuristic());
    instance.set_GNN_model_path(parser.get_GNN_model_path());
    m_initialized = true;
  }
  return instance;
}

Configuration::Configuration() = default;

// Getters and setters

bool Configuration::get_bisimulation() const noexcept { return m_bisimulation; }
void Configuration::set_bisimulation(const std::string &val) {
  m_bisimulation = str_to_bool(val);
}
void Configuration::set_bisimulation(const bool val) { m_bisimulation = val; }

const std::string &Configuration::get_bisimulation_type() const noexcept {
  return m_bisimulation_type;
}

void Configuration::set_bisimulation_type(const std::string &val) {
  m_bisimulation_type = trim(val);
  set_bisimulation_type_bool();
}

bool Configuration::get_bisimulation_type_bool() const noexcept {
  return m_bisimulation_type_bool;
}
void Configuration::set_bisimulation_type_bool() {
  m_bisimulation_type_bool = get_bisimulation_type() != "PT";
}

bool Configuration::get_check_visited() const noexcept {
  return m_check_visited;
}
void Configuration::set_check_visited(const std::string &val) {
  m_check_visited = str_to_bool(val);
}
void Configuration::set_check_visited(const bool val) { m_check_visited = val; }

const SearchType &Configuration::get_search_strategy() const noexcept {
  return m_search_strategy_enum;
}

void Configuration::set_search_strategy(const std::string &val) {
  m_search_strategy = val;
  set_search_strategy_enum();
}

const Heuristics &Configuration::get_heuristic_opt() const noexcept {
  return m_heuristic_enum;
}

void Configuration::set_heuristic_opt(const std::string &val) {
  m_heuristic_opt = val;
  set_heuristic_enum();
}

void Configuration::set_GNN_model_path(const std::string &val) {
  m_GNN_model_path = val;
}

const std::string &Configuration::get_GNN_model_path() const noexcept {
  return m_GNN_model_path;
}

// In Configuration.cpp
void Configuration::set_field_by_name(const std::string &field,
                                      const std::string &value) {
  if (field == "bisimulation" || field == "b")
    set_bisimulation(value);
  else if (field == "bisimulation_type")
    set_bisimulation_type(value);
  else if (field == "check_visited" || field == "c")
    set_check_visited(value);
  else if (field == "search" || field == "s")
    set_search_strategy(value);
  else if (field == "heuristics" || field == "u")
    set_heuristic_opt(value);
  else if (field == "GNN_model")
    set_GNN_model_path(value);
  else {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::PortfolioConfigFieldError,
        "[PortfolioSearch] Error while reading config file; the field: " +
            field + " is not recognized." +
            "Please check the Line Arguments for the possible names of the "
            "fields. (Search related without the - or -- prefix)");
  }
  // Optionally handle unknown fields
}

void Configuration::set_search_strategy_enum() {
  if (m_search_strategy == "BFS") {
    m_search_strategy_enum = SearchType::BFS;
  } else if (m_search_strategy == "DFS") {
    m_search_strategy_enum = SearchType::DFS;
  } else if (m_search_strategy == "IDFS") {
    m_search_strategy_enum = SearchType::IDFS;
  } else if (m_search_strategy == "HFS") {
    m_search_strategy_enum = SearchType::HFS;
  } else {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::ArgParseError,
                                   "Invalid search strategy specified: " +
                                       m_search_strategy);
  }
}

void Configuration::add_bisimulation_failure() {
  int bisimulation_fails_threshold = 10;
  if (bisimulation_failures++ > bisimulation_fails_threshold) {
    m_bisimulation = false;
    ArgumentParser::get_instance().get_output_stream()
        << "\n[WARNING] Bisimulation (" << m_bisimulation_type
        << ") has failed more than "
        << std::to_string(bisimulation_fails_threshold)
        << " times so it will now be deactivated.\n";
  }
}

void Configuration::set_heuristic_enum() {
  if (m_heuristic_opt == "SUBGOALS") {
    m_heuristic_enum = Heuristics::SUBGOALS;
  } else if (m_heuristic_opt == "L_PG") {
    m_heuristic_enum = Heuristics::L_PG;
  } else if (m_heuristic_opt == "S_PG") {
    m_heuristic_enum = Heuristics::S_PG;
  } else if (m_heuristic_opt == "C_PG") {
    m_heuristic_enum = Heuristics::C_PG;
  } else if (m_heuristic_opt == "GNN") {
    m_heuristic_enum = Heuristics::GNN;
  } else {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::ArgParseError,
                                   "Invalid heuristic specified: " +
                                       m_heuristic_opt);
  }
}

void Configuration::print(std::ostream &os) const {
  os << "  Extra Information:\n";
  os << "    Bisimulation: " << (m_bisimulation ? "active" : "inactive")
     << '\n';
  if (m_bisimulation) {
    os << "    Bisimulation type: ";
    if (m_bisimulation_type == "FB")
      os << "Fast Bisimulation";
    else
      os << "Paige and Tarjan";
    os << '\n';
  }
  os << "    Already visited state check: "
     << (m_check_visited ? "active" : "inactive") << '\n';
  if (m_search_strategy_enum == SearchType::HFS &&
      m_heuristic_enum == Heuristics::GNN) {
    os << "    Path to GNN model: " << m_GNN_model_path << '\n';
  }
}
