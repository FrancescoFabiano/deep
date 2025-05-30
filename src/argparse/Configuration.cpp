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

// Helper to trim trailing and leading spaces
static std::string trim(const std::string& s) {
    const auto start = s.find_first_not_of(" \t\r\n");
    const auto end = s.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

// Helper to convert string to bool
static bool str_to_bool(const std::string& s) {
    std::string val = trim(s);
    std::ranges::transform(val, val.begin(), ::tolower);
    return (val == "1" || val == "true" || val == "yes" || val == "on");
}


// Static thread-local instance pointer
thread_local Configuration* Configuration::instance = nullptr;

void Configuration::create_instance() {
    if (!instance) {
        instance = new Configuration();

        // Copy values from ArgumentParser singleton using setters
        const ArgumentParser& parser = ArgumentParser::get_instance();
        instance->set_bisimulation(parser.get_bisimulation());
        instance->set_bisimulation_type(parser.get_bisimulation_type());
        instance->set_check_visited(parser.get_check_visited());
        instance->set_search_strategy(parser.get_search_strategy());
        instance->set_heuristic_opt(parser.get_heuristic());
        instance->set_log_file_path(parser.get_log_file_path());
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

bool Configuration::get_bisimulation() const noexcept { return m_bisimulation; }
void Configuration::set_bisimulation(const std::string& val) { m_bisimulation = str_to_bool(val); }
void Configuration::set_bisimulation(const bool val) { m_bisimulation = val; }

const std::string& Configuration::get_bisimulation_type() const noexcept { return m_bisimulation_type; }
void Configuration::set_bisimulation_type(const std::string& val)
{
    m_bisimulation_type = trim(val);
    set_bisimulation_type_bool();
}

bool Configuration::get_bisimulation_type_bool() const noexcept { return m_bisimulation_type_bool; }
void Configuration::set_bisimulation_type_bool() { m_bisimulation_type_bool = get_bisimulation_type() != "PT"; }

bool Configuration::get_check_visited() const noexcept { return m_check_visited; }
void Configuration::set_check_visited(const std::string& val) { m_check_visited = str_to_bool(val); }
void Configuration::set_check_visited(const bool val) { m_check_visited = val; }

const std::string& Configuration::get_search_strategy() const noexcept { return m_search_strategy; }
void Configuration::set_search_strategy(const std::string& val) { m_search_strategy = trim(val); }

const std::string& Configuration::get_heuristic_opt() const noexcept { return m_heuristic_opt; }
void Configuration::set_heuristic_opt(const std::string& val) { m_heuristic_opt = trim(val); }

const std::string& Configuration::get_log_file_path() const noexcept { return m_log_file_path; }
void Configuration::set_log_file_path(const std::string& val) { m_log_file_path = trim(val); }


// In Configuration.cpp
void Configuration::set_field_by_name(const std::string& field, const std::string& value) {
    if (field == "bis") set_bisimulation(value);
    else if (field == "bis_type") set_bisimulation_type(value);
    else if (field == "check_visited") set_check_visited(value);
    else if (field == "search_strategy") set_search_strategy(value);
    else if (field == "heuristic_opt") set_heuristic_opt(value);
    else if (field == "log_file_path") set_log_file_path(value);
    else
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::PortfolioConfigFieldError,
            "[PortfolioSearch] Error while reading config file; the field: " + field + " is not recognized." +
            " Please check the Line Arguments for the possible names of the fields. (Search related without the -- prefix)"
        );
    }
    // Optionally handle unknown fields
}

void Configuration::print(std::ostream& os) const {
    os << "Configuration Parameters:\n";
    os << "  bisimulation: " << std::boolalpha << m_bisimulation << '\n';
    os << "  bisimulation_type: " << m_bisimulation_type << '\n';
    os << "  check_visited: " << std::boolalpha << m_check_visited << '\n';
    os << "  search_strategy: " << m_search_strategy << '\n';
    os << "  heuristic_opt: " << m_heuristic_opt << '\n';
}
