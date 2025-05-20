#include "HelperPrint.h"
#include "ExitHandler.h"
#include "KripkeWorld.h"

#include "formulae/BeliefFormula.h"
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <ranges>
#include <sstream>

#include "FormulaHelper.h"
#include "KripkeState.h"

/**
 * \file HelperPrint.cpp
 * \brief Implementation of HelperPrint.h
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 2025
 */

HelperPrint* HelperPrint::instance = nullptr;


HelperPrint::HelperPrint()
{
	instance->set_grounder(Domain::get_instance().get_grounder());
}

HelperPrint& HelperPrint::get_instance()
{
	if (!instance) {
		instance = new HelperPrint();
	}
	return *instance;
}

void HelperPrint::set_grounder(const Grounder& gr)
{
    m_grounder = gr;
    m_set_grounder = true;
}

void HelperPrint::print_list(const StringsSet& to_print, std::ostream& os)
{
    bool first = true;
    for (const auto& str : to_print) {
        if (!first) os << ",";
        first = false;
        os << str;
    }
}

void HelperPrint::print_list(const StringSetsSet& to_print, std::ostream& os)
{
    bool first = true;
    for (const auto& set : to_print) {
        if (!first) os << " OR ";
        first = false;
        print_list(set, os);
    }
}

void HelperPrint::print_list(const FluentsSet& to_print, std::ostream& os) const
{
    if (!m_set_grounder) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::PrintUnsetGrounderError,
            "Tried to print fluents with degrounding, but grounder is not set."
        );
    }
    print_list(m_grounder.deground_fluent(to_print), os);
}

void HelperPrint::print_list(const FluentFormula& to_print, std::ostream& os) const
{
    if (!m_set_grounder) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::PrintUnsetGrounderError,
            "Tried to print fluent formula with degrounding, but grounder is not set."
        );
    }
    print_list(m_grounder.deground_fluent(to_print), os);
}


void HelperPrint::print_list(const FormulaeList& to_print, std::ostream& os)
{
    bool first = true;
    for (const auto& formula : to_print) {
        if (!first) os << " AND ";
        first = false;
        formula.print(os);
    }
}

void HelperPrint::print_list(const KripkeWorldPointersSet& to_print, std::ostream& os)
{
    bool first = true;
    for (const auto& ptr : to_print) {
        if (ptr.get_ptr() != nullptr) {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::PrintNullPointerError,
                "Null pointer encountered in KripkeWorldPointersSet during print."
            );
        }
        if (!first) os << "\n";
        first = false;
        os << ptr.get_ptr()->get_id();
    }
}

void HelperPrint::print_list(const ActionIdsList& to_print, std::ostream& os) const
{
    bool first = true;
    for (const auto& id : to_print) {
        if (!first) os << ", ";
        first = false;
        if (m_set_grounder) {
            os << m_grounder.deground_action(id);
        } else {
            os << id;
        }
    }
}

void HelperPrint::print_list_ag(const AgentsSet& to_print, std::ostream& os) const
{
    if (!m_set_grounder) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::PrintUnsetGrounderError,
            "Tried to print agents with degrounding, but grounder is not set."
        );
    }
    bool first = true;
    for (const auto& ag : to_print) {
        if (!first) os << ", ";
        first = false;
        os << m_grounder.deground_agent(ag);
    }
}


void HelperPrint::print_belief_formula(const BeliefFormula& to_print, std::ostream& os) const
{
    switch (to_print.get_formula_type()) {
    case BeliefFormulaType::FLUENT_FORMULA:
        print_list(to_print.get_fluent_formula(), os);
        break;
    case BeliefFormulaType::BELIEF_FORMULA:
        os << "B(" << m_grounder.deground_agent(to_print.get_agent()) << ",(";
        print_belief_formula(to_print.get_bf1(), os);
        os << "))";
        break;
    case BeliefFormulaType::C_FORMULA:
        os << "C([";
        print_list_ag(to_print.get_group_agents(), os);
        os << "],";
        print_belief_formula(to_print.get_bf1(), os);
        os << ")";
        break;
    case BeliefFormulaType::E_FORMULA:
        os << "E([";
        print_list_ag(to_print.get_group_agents(), os);
        os << "],";
        print_belief_formula(to_print.get_bf1(), os);
        os << ")";
        os << ")";
        break;
    case BeliefFormulaType::PROPOSITIONAL_FORMULA:
        if (to_print.get_operator() == BeliefFormulaOperator::BF_NOT)
            os << "NOT(";
        print_belief_formula(to_print.get_bf1(), os);
        if (to_print.get_operator() == BeliefFormulaOperator::BF_NOT)
            os << ")";
        if (to_print.get_operator() == BeliefFormulaOperator::BF_AND)
            os << " AND ";
        if (to_print.get_operator() == BeliefFormulaOperator::BF_OR)
            os << " OR ";
        else if (to_print.get_operator() == BeliefFormulaOperator::BF_FAIL) {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::BeliefFormulaOperatorUnset,
                "ERROR IN DECLARATION."
            );
        }
        if (!to_print.is_bf2_null()) {
            print_belief_formula(to_print.get_bf2(), os);
        }
        break;
    case BeliefFormulaType::BF_EMPTY:
        os << "Empty\n";
        break;
    case BeliefFormulaType::BF_TYPE_FAIL:
    default:
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BeliefFormulaTypeUnset,
            "Unknown BeliefFormula type."
        );
        break;
    }
}

std::string HelperPrint::generate_log_file_path(const std::string& domain_file) {
    namespace fs = std::filesystem;
    fs::create_directories("log");

    fs::path domain_path(domain_file);
    std::string domain_name = domain_path.stem().string();

    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif

    int repetition = 0;
    std::string path;
    do {
        std::ostringstream oss;
        oss << "log/"
            << domain_name << "_"
            << std::put_time(&tm, "%Y%m%d_%H%M%S");
        if (repetition > 0) {
            oss << "_" << repetition;
        }
        oss << ".log";
        path = oss.str();
        ++repetition;
    } while (fs::exists(path));
    return path;
}


void HelperPrint::print_state(const KripkeState & kstate, std::ostream& os) const
{
	os << std::endl;
	os << "The Pointed World has id ";
	print_list(kstate.get_pointed().get_fluent_set(), os);
	os << "-" << kstate.get_pointed().get_repetition();
	os << std::endl;
	os << "*******************************************************************" << std::endl;

	os << "World List:" << std::endl;
	int counter = 1;
	for (const auto& world_ptr : kstate.get_worlds()) {
		os << "W-" << counter << ": ";
		print_list(world_ptr.get_fluent_set(), os);
		os << " rep:" << world_ptr.get_repetition();
		os << std::endl;
		++counter;
	}

	os << std::endl;
	os << "*******************************************************************" << std::endl;
	os << "Edge List:" << std::endl;
	counter = 1;
	for (const auto& [from, from_map] : kstate.get_beliefs()) {
		for (const auto& [ag, to_set] : from_map) {
			for (const auto& to : to_set) {
				os << "E-" << counter << ": (";
				print_list(from.get_fluent_set(), os);
				os << "," << from.get_repetition();
				os << ") - (";
				print_list(to.get_fluent_set(), os);
				os << "," << to.get_repetition();
				os << ") ag:" << m_grounder.deground_agent(ag);
				os << std::endl;
				++counter;
			}
		}
	}
	os << "*******************************************************************" << std::endl;
}

///@TODO: Make this function so it can be used also to generate the dataset (maybe with a flag)
void HelperPrint::print_dot_format(const KripkeState & kstate, std::ostream & os) const
{
	auto& worlds = kstate.get_worlds();
	auto& pointed = kstate.get_pointed();

	os << "//WORLDS List:" << std::endl;
	std::map<FluentsSet, int> map_world_to_index;
	std::map<unsigned short, char> map_rep_to_name;
	char found_rep = static_cast<char>(Domain::get_instance().get_agents().size() + 'A');
	int found_fs = 0;

	for (const auto& world_ptr : worlds) {
	    os << "	node [shape = " << ((world_ptr == pointed) ? "doublecircle" : "circle") << "] ";

	    const auto& tmp_fs = world_ptr.get_fluent_set();
	    if (!map_world_to_index.contains(tmp_fs)) {
	        map_world_to_index[tmp_fs] = found_fs++;
		}
	    unsigned short tmp_unsh = world_ptr.get_repetition();
	    if (!map_rep_to_name.contains(tmp_unsh)) {
	        map_rep_to_name[tmp_unsh] = found_rep++;
		}
		os << "\"" << map_rep_to_name[tmp_unsh] << "_" << map_world_to_index[tmp_fs] << "\";";
		os << "// (";
	    auto tmp_stset = m_grounder.deground_fluent(tmp_fs);
	    bool print_first = false;
	    for (const auto& str : tmp_stset) {
	        if (print_first) os << ",";
			print_first = true;
	        os << str;
		}
		os << ")\n";
	}

	os << "\n\n";
	os << "//RANKS List:" << std::endl;

	std::map<int, KripkeWorldPointersSet> for_rank_print;
	for (const auto& world_ptr : worlds) {
	    for_rank_print[world_ptr.get_repetition()].insert(world_ptr);
	}

	for (const auto& set : for_rank_print | std::views::values) {
		os << "	{rank = same; ";
	    for (const auto& world_ptr : set) {
	        os << "\"" << map_rep_to_name[world_ptr.get_repetition()] << "_" << map_world_to_index[world_ptr.get_fluent_set()] << "\"; ";
		}
		os << "}\n";
	}

	os << "\n\n";
	os << "//EDGES List:" << std::endl;

	std::map < std::tuple<std::string, std::string>, std::set<std::string> > edges;

	for (const auto& [from, from_map] : kstate.get_beliefs()) {
	    for (const auto& [ag, to_set] : from_map) {
	        for (const auto& to : to_set) {
	            std::string from_str = "_" + std::to_string(map_world_to_index[from.get_fluent_set()]);
	            from_str.insert(0, 1, map_rep_to_name[from.get_repetition()]);
	            std::string to_str = "_" + std::to_string(map_world_to_index[to.get_fluent_set()]);
	            to_str.insert(0, 1, map_rep_to_name[to.get_repetition()]);
	            edges[{from_str, to_str}].insert(m_grounder.deground_agent(ag));
			}
		}
	}

	std::map < std::tuple<std::string, std::string>, std::set < std::string>> to_print_double;
	for (auto it_map = edges.begin(); it_map != edges.end();) {
	    bool erased = false;
	    for (auto it_map_2 = std::next(it_map); it_map_2 != edges.end();) {
	        if (std::get<0>(it_map->first) == std::get<1>(it_map_2->first) &&
	            std::get<1>(it_map->first) == std::get<0>(it_map_2->first) &&
	            it_map->second == it_map_2->second &&
	            std::get<0>(it_map->first) != std::get<1>(it_map->first)) {
							to_print_double[it_map->first] = it_map->second;
							edges.erase(it_map_2);
							it_map = edges.erase(it_map);
	            erased = true;
	            break;
	        } else {
	            ++it_map_2;
					}
				}
	    if (!erased) ++it_map;
	}

	for (const auto& [key, agents] : edges) {
		os << "	\"";
	    os << std::get<0>(key);
		os << "\" -> \"";
	    os << std::get<1>(key);
		os << "\" ";
		os << "[ label = \"";
	    std::string tmp_string;
	    for (const auto& ag : agents) {
	        tmp_string += ag + ",";
		}
	    if (!tmp_string.empty()) tmp_string.pop_back();
		os << tmp_string;
		os << "\" ];\n";
	}

	for (const auto& [key, agents] : to_print_double) {
		os << "	\"";
	    os << std::get<0>(key);
		os << "\" -> \"";
	    os << std::get<1>(key);
		os << "\" ";
		os << "[ dir=both label = \"";
	    std::string tmp_string;
	    for (const auto& ag : agents) {
	        tmp_string += ag + ",";
		}
	    if (!tmp_string.empty()) tmp_string.pop_back();
		os << tmp_string;
		os << "\" ];\n";
	}

	std::string color = "<font color=\"#ffffff\">";
	os << "\n\n//WORLDS description Table:" << std::endl;
	os << "	node [shape = plain]\n\n";
	os << "	description[label=<\n";
	os << "	<table border = \"0\" cellborder = \"1\" cellspacing = \"0\" >\n";
	for (const auto& world_ptr : worlds) {
	    const auto& tmp_fs = world_ptr.get_fluent_set();
	    bool print_first = false;
	    os << "		<tr><td>" << map_rep_to_name[world_ptr.get_repetition()] << "_" << map_world_to_index[tmp_fs] << "</td> <td>";
	    for (const auto& fluent : tmp_fs) {
	        if (print_first) os << ", ";
			print_first = true;
	        if (FormulaHelper::is_negated(fluent))
	            color = "<font color=\"#0000ff\"> ";
	        else
	            color = "<font color=\"#ff1020\">";
	        os << color << m_grounder.deground_fluent(fluent) << "</font>";
		}
		os << "</td></tr>\n";
	}
	os << "	</table>>]\n";
	os << "	{rank = max; description};\n";
}
