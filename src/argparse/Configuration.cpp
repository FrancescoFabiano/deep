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
#include <sstream>

// Helper to trim trailing and leading spaces
static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    auto end = s.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

// Helper to convert string to bool
static bool str_to_bool(const std::string& s) {
    std::string val = trim(s);
    std::ranges::transform(val, val.begin(), ::tolower);
    return (val == "1" || val == "true" || val == "yes" || val == "on");
}

// Helper to convert string to int
static int str_to_int(const std::string& s) {
    return std::stoi(trim(s));
}

// Static thread-local instance pointer
thread_local Configuration* Configuration::instance = nullptr;

void Configuration::create_instance() {
    if (!instance) {
        instance = new Configuration();

        // Copy values from ArgumentParser singleton using setters
        const ArgumentParser& parser = ArgumentParser::get_instance();
        instance->set_debug(parser.get_debug());
        instance->set_bisimulation(parser.get_bisimulation());
        instance->set_bisimulation_type(parser.get_bisimulation_type());
        instance->set_bisimulation_type_bool(parser.get_bisimulation_type_bool());
        instance->set_check_visited(parser.get_check_visited());
        instance->set_dataset_mode(parser.get_dataset_mode());
        instance->set_dataset_depth(parser.get_dataset_size());
        instance->set_search_strategy(parser.get_search_strategy());
        instance->set_heuristic_opt(parser.get_heuristic());
        instance->set_parallel_type(parser.get_parallel_type());
        instance->set_parallel_wait(parser.get_parallel_wait());
        instance->set_exec_plan(parser.get_execute_plan());
        instance->set_exec_actions(parser.get_execution_actions());
        instance->set_output_results_file(parser.get_results_file());
        instance->set_plan_file(parser.get_plan_file());
        instance->set_log_enabled(parser.get_log_enabled());
        instance->set_log_file_path(parser.get_log_file_path());
        instance->set_input_file(parser.get_input_file());
    }
}

Configuration& Configuration::get_instance() {
    if (!instance) {
        create_instance();
    }
    return *instance;
}

Configuration::Configuration() = default;

// Getters and setters

bool Configuration::get_debug() const noexcept { return m_debug; }
void Configuration::set_debug(const std::string& val) { m_debug = str_to_bool(val); }
void Configuration::set_debug(const bool val) { m_debug = val; }

bool Configuration::get_bisimulation() const noexcept { return m_bisimulation; }
void Configuration::set_bisimulation(const std::string& val) { m_bisimulation = str_to_bool(val); }
void Configuration::set_bisimulation(const bool val) { m_bisimulation = val; }

const std::string& Configuration::get_bisimulation_type() const noexcept { return m_bisimulation_type; }
void Configuration::set_bisimulation_type(const std::string& val) { m_bisimulation_type = trim(val); }

bool Configuration::get_bisimulation_type_bool() const noexcept { return m_bisimulation_type_bool; }
void Configuration::set_bisimulation_type_bool(const std::string& val) { m_bisimulation_type_bool = str_to_bool(val); }
void Configuration::set_bisimulation_type_bool(const bool val) { m_bisimulation_type_bool = val; }

bool Configuration::get_check_visited() const noexcept { return m_check_visited; }
void Configuration::set_check_visited(const std::string& val) { m_check_visited = str_to_bool(val); }
void Configuration::set_check_visited(const bool val) { m_check_visited = val; }

bool Configuration::get_dataset_mode() const noexcept { return m_dataset_mode; }
void Configuration::set_dataset_mode(const std::string& val) { m_dataset_mode = str_to_bool(val); }
void Configuration::set_dataset_mode(const bool val) { m_dataset_mode = val; }

int Configuration::get_dataset_depth() const noexcept { return m_dataset_depth; }
void Configuration::set_dataset_depth(const std::string& val) { m_dataset_depth = str_to_int(val); }
void Configuration::set_dataset_depth(const int val) { m_dataset_depth = val; }

const std::string& Configuration::get_search_strategy() const noexcept { return m_search_strategy; }
void Configuration::set_search_strategy(const std::string& val) { m_search_strategy = trim(val); }

const std::string& Configuration::get_heuristic_opt() const noexcept { return m_heuristic_opt; }
void Configuration::set_heuristic_opt(const std::string& val) { m_heuristic_opt = trim(val); }

const std::string& Configuration::get_parallel_type() const noexcept { return m_parallel_type; }
void Configuration::set_parallel_type(const std::string& val) { m_parallel_type = trim(val); }

const std::string& Configuration::get_parallel_wait() const noexcept { return m_parallel_wait; }
void Configuration::set_parallel_wait(const std::string& val) { m_parallel_wait = trim(val); }

bool Configuration::get_exec_plan() const noexcept { return m_exec_plan; }
void Configuration::set_exec_plan(const std::string& val) { m_exec_plan = str_to_bool(val); }
void Configuration::set_exec_plan(const bool val) { m_exec_plan = val; }

const std::vector<std::string>& Configuration::get_exec_actions() const noexcept { return m_exec_actions; }
void Configuration::set_exec_actions(const std::string& val) {
    m_exec_actions.clear();
    std::istringstream ss(val);
    std::string action;
    while (std::getline(ss, action, ',')) {
        action = trim(action);
        if (!action.empty()) m_exec_actions.push_back(action);
    }
}
void Configuration::set_exec_actions(const std::vector<std::string>& val) { m_exec_actions = val; }

bool Configuration::get_output_results_file() const noexcept { return m_output_results_file; }
void Configuration::set_output_results_file(const std::string& val) { m_output_results_file = str_to_bool(val); }
void Configuration::set_output_results_file(const bool val) { m_output_results_file = val; }

const std::string& Configuration::get_plan_file() const noexcept { return m_plan_file; }
void Configuration::set_plan_file(const std::string& val) { m_plan_file = trim(val); }

bool Configuration::get_log_enabled() const noexcept { return m_log_enabled; }
void Configuration::set_log_enabled(const std::string& val) { m_log_enabled = str_to_bool(val); }
void Configuration::set_log_enabled(const bool val) { m_log_enabled = val; }

const std::string& Configuration::get_log_file_path() const noexcept { return m_log_file_path; }
void Configuration::set_log_file_path(const std::string& val) { m_log_file_path = trim(val); }

const std::string& Configuration::get_input_file() const noexcept { return m_input_file; }
void Configuration::set_input_file(const std::string& val) { m_input_file = trim(val); }
