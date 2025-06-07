#include "HelperPrint.h"
#include "ExitHandler.h"
#include "KripkeWorld.h"

#include "formulae/BeliefFormula.h"
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <ranges>
#include <sstream>
#include <fstream>
#include <regex>

#include "ArgumentParser.h"
#include "Domain.h"
#include "FormulaHelper.h"
#include "KripkeState.h"
#include "neuralnets/TrainingDataset.h"

/**
 * \file HelperPrint.cpp
 * \brief Implementation of HelperPrint.h
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 2025
 */

HelperPrint::HelperPrint()
{
}

HelperPrint& HelperPrint::get_instance()
{
    static HelperPrint instance;
    return instance;
}

const Grounder& HelperPrint::get_grounder() const
{
    if (!m_set_grounder)
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::PrintUnsetGrounderError,
            "Tried to access grounder, but it is not set in HelperPrint."
        );
    }
    return m_grounder;
}

void HelperPrint::set_grounder(const Grounder& gr)
{
    m_grounder = gr;
    m_set_grounder = true;
}

void HelperPrint::print_list(const StringsSet& to_print)
{
    auto& os = ArgumentParser::get_instance().get_output_stream();
    bool first = true;
    for (const auto& str : to_print)
    {
        if (!first) os << ",";
        first = false;
        os << str;
    }
}

void HelperPrint::print_list(const StringSetsSet& to_print)
{
    auto& os = ArgumentParser::get_instance().get_output_stream();
    bool first = true;
    for (const auto& set : to_print)
    {
        if (!first) os << " OR ";
        first = false;
        print_list(set);
    }
}

void HelperPrint::print_list(const FluentsSet& to_print) const
{
    if (!m_set_grounder)
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::PrintUnsetGrounderError,
            "Tried to print fluents with degrounding, but grounder is not set."
        );
    }
    print_list(m_grounder.deground_fluent(to_print));
}

void HelperPrint::print_list(const FluentFormula& to_print) const
{
    if (!m_set_grounder)
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::PrintUnsetGrounderError,
            "Tried to print fluent formula with degrounding, but grounder is not set."
        );
    }
    print_list(m_grounder.deground_fluent(to_print));
}


void HelperPrint::print_list(const FormulaeList& to_print)
{
    auto& os = ArgumentParser::get_instance().get_output_stream();

    bool first = true;
    for (const auto& formula : to_print)
    {
        if (!first) os << " AND ";
        first = false;
        formula.print();
    }
}

void HelperPrint::print_list(const KripkeWorldPointersSet& to_print)
{
    bool first = true;
    for (const auto& ptr : to_print)
    {
        if (ptr.get_ptr() != nullptr)
        {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::PrintNullPointerError,
                "Null pointer encountered in KripkeWorldPointersSet during print."
            );
        }
        auto& os = ArgumentParser::get_instance().get_output_stream();
        if (!first) os << "\n";
        first = false;
        os << ptr.get_ptr()->get_id();
    }
}

void HelperPrint::print_list(const ActionIdsList& to_print) const
{
    auto& os = ArgumentParser::get_instance().get_output_stream();
    bool first = true;
    for (const auto& id : to_print)
    {
        if (!first) os << ", ";
        first = false;
        if (m_set_grounder)
        {
            os << m_grounder.deground_action(id);
        }
        else
        {
            os << id;
        }
    }
}

void HelperPrint::print_list_ag(const AgentsSet& to_print) const
{
    auto& os = ArgumentParser::get_instance().get_output_stream();
    if (!m_set_grounder)
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::PrintUnsetGrounderError,
            "Tried to print agents with degrounding, but grounder is not set."
        );
    }
    bool first = true;
    for (const auto& ag : to_print)
    {
        if (!first) os << ", ";
        first = false;
        os << m_grounder.deground_agent(ag);
    }
}

void HelperPrint::print_belief_formula_parsed(const BeliefFormulaParsed& to_print)
{
    auto& os = ArgumentParser::get_instance().get_output_stream();
    switch (to_print.get_formula_type())
    {
    case BeliefFormulaType::FLUENT_FORMULA:
        print_list(to_print.get_string_fluent_formula());
        break;
    case BeliefFormulaType::BELIEF_FORMULA:
        os << "B(" << to_print.get_string_agent() << ",(";
        print_belief_formula_parsed(to_print.get_bf1());
        os << "))";
        break;
    case BeliefFormulaType::C_FORMULA:
        os << "C([";
        print_list(to_print.get_group_agents());
        os << "],";
        print_belief_formula_parsed(to_print.get_bf1());
        os << ")";
        break;
    case BeliefFormulaType::E_FORMULA:
        os << "E([";
        print_list(to_print.get_group_agents());
        os << "],";
        print_belief_formula_parsed(to_print.get_bf1());
        os << ")";
        os << ")";
        break;
    case BeliefFormulaType::PROPOSITIONAL_FORMULA:
        if (to_print.get_operator() == BeliefFormulaOperator::BF_NOT)
            os << "NOT(";
        print_belief_formula_parsed(to_print.get_bf1());
        if (to_print.get_operator() == BeliefFormulaOperator::BF_NOT)
            os << ")";
        if (to_print.get_operator() == BeliefFormulaOperator::BF_AND)
            os << " AND ";
        if (to_print.get_operator() == BeliefFormulaOperator::BF_OR)
            os << " OR ";
        else if (to_print.get_operator() == BeliefFormulaOperator::BF_FAIL)
        {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::BeliefFormulaOperatorUnset,
                "ERROR IN DECLARATION."
            );
        }
        if (!to_print.is_bf2_null())
        {
            print_belief_formula_parsed(to_print.get_bf2());
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


void HelperPrint::print_belief_formula(const BeliefFormula& to_print) const
{
    auto& os = ArgumentParser::get_instance().get_output_stream();
    switch (to_print.get_formula_type())
    {
    case BeliefFormulaType::FLUENT_FORMULA:
        print_list(to_print.get_fluent_formula());
        break;
    case BeliefFormulaType::BELIEF_FORMULA:
        os << "B(" << m_grounder.deground_agent(to_print.get_agent()) << ",(";
        print_belief_formula(to_print.get_bf1());
        os << "))";
        break;
    case BeliefFormulaType::C_FORMULA:
        os << "C([";
        print_list_ag(to_print.get_group_agents());
        os << "],";
        print_belief_formula(to_print.get_bf1());
        os << ")";
        break;
    case BeliefFormulaType::E_FORMULA:
        os << "E([";
        print_list_ag(to_print.get_group_agents());
        os << "],";
        print_belief_formula(to_print.get_bf1());
        os << ")";
        os << ")";
        break;
    case BeliefFormulaType::PROPOSITIONAL_FORMULA:
        if (to_print.get_operator() == BeliefFormulaOperator::BF_NOT)
            os << "NOT(";
        print_belief_formula(to_print.get_bf1());
        if (to_print.get_operator() == BeliefFormulaOperator::BF_NOT)
            os << ")";
        if (to_print.get_operator() == BeliefFormulaOperator::BF_AND)
            os << " AND ";
        if (to_print.get_operator() == BeliefFormulaOperator::BF_OR)
            os << " OR ";
        else if (to_print.get_operator() == BeliefFormulaOperator::BF_FAIL)
        {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::BeliefFormulaOperatorUnset,
                "ERROR IN DECLARATION."
            );
        }
        if (!to_print.is_bf2_null())
        {
            print_belief_formula(to_print.get_bf2());
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

std::string HelperPrint::generate_log_file_path(const std::string& domain_file)
{
    namespace fs = std::filesystem;
    fs::create_directories(OutputPaths::LOGS_FOLDER);

    const fs::path domain_path(domain_file);
    const std::string domain_name = domain_path.stem().string();

    const auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif

    int repetition = 0;
    std::string path;
    do
    {
        std::ostringstream oss;
        oss << OutputPaths::LOGS_FOLDER << "/"
            << domain_name << "_"
            << std::put_time(&tm, "%Y%m%d_%H%M%S");
        if (repetition > 0)
        {
            oss << "_" << repetition;
        }
        oss << ".log";
        path = oss.str();
        ++repetition;
    }
    while (fs::exists(path));
    return path;
}


void HelperPrint::print_state(const KripkeState& kstate) const
{
    auto& os = ArgumentParser::get_instance().get_output_stream();

    os << std::endl;
    os << "The Pointed World has id ";
    print_list(kstate.get_pointed().get_fluent_set());
    os << "-" << kstate.get_pointed().get_repetition();
    os << std::endl;
    os << "*******************************************************************" << std::endl;

    os << "World List:" << std::endl;
    int counter = 1;
    for (const auto& world_ptr : kstate.get_worlds())
    {
        os << "W-" << counter << ": ";
        print_list(world_ptr.get_fluent_set());
        os << " rep:" << world_ptr.get_repetition();
        os << std::endl;
        ++counter;
    }

    os << std::endl;
    os << "*******************************************************************" << std::endl;
    os << "Edge List:" << std::endl;
    counter = 1;
    for (const auto& [from, from_map] : kstate.get_beliefs())
    {
        for (const auto& [ag, to_set] : from_map)
        {
            for (const auto& to : to_set)
            {
                os << "E-" << counter << ": (";
                print_list(from.get_fluent_set());
                os << "," << from.get_repetition();
                os << ") - (";
                print_list(to.get_fluent_set());
                os << "," << to.get_repetition();
                os << ") ag:" << m_grounder.deground_agent(ag);
                os << std::endl;
                ++counter;
            }
        }
    }
    os << "*******************************************************************" << std::endl;
}

void HelperPrint::print_dot_format(const KripkeState& kstate, std::ofstream& ofs) const
{
    auto& worlds = kstate.get_worlds();
    auto& pointed = kstate.get_pointed();
    ofs << "digraph K {" << std::endl;
    ofs << "//WORLDS List:" << std::endl;
    std::map<FluentsSet, int> map_world_to_index;
    std::map<unsigned short, char> map_rep_to_name;
    char found_rep = static_cast<char>(Domain::get_instance().get_agents().size() + 'A');
    int found_fs = 0;

    for (const auto& world_ptr : worlds)
    {
        ofs << "	node [shape = " << ((world_ptr == pointed) ? "doublecircle" : "circle") << "] ";

        const auto& tmp_fs = world_ptr.get_fluent_set();
        if (!map_world_to_index.contains(tmp_fs))
        {
            map_world_to_index[tmp_fs] = found_fs++;
        }
        unsigned short tmp_unsh = world_ptr.get_repetition();
        if (!map_rep_to_name.contains(tmp_unsh))
        {
            map_rep_to_name[tmp_unsh] = found_rep++;
        }
        ofs << "\"" << map_rep_to_name[tmp_unsh] << "_" << map_world_to_index[tmp_fs] << "\";";
        ofs << "// (";
        auto strings_set = m_grounder.deground_fluent(tmp_fs);
        bool print_first = false;
        for (const auto& str : strings_set)
        {
            if (print_first) ofs << ",";
            print_first = true;
            ofs << str;
        }
        ofs << ")\n";
    }

    ofs << "\n\n";
    ofs << "//RANKS List:" << std::endl;

    std::map<int, KripkeWorldPointersSet> for_rank_print;
    for (const auto& world_ptr : worlds)
    {
        for_rank_print[world_ptr.get_repetition()].insert(world_ptr);
    }

    for (const auto& set : for_rank_print | std::views::values)
    {
        ofs << "	{rank = same; ";
        for (const auto& world_ptr : set)
        {
            ofs << "\"" << map_rep_to_name[world_ptr.get_repetition()] << "_" << map_world_to_index[world_ptr.
                get_fluent_set()] << "\"; ";
        }
        ofs << "}\n";
    }

    ofs << "\n\n";
    ofs << "//EDGES List:" << std::endl;

    std::map<std::tuple<std::string, std::string>, std::set<std::string>> edges;

    for (const auto& [from, from_map] : kstate.get_beliefs())
    {
        for (const auto& [ag, to_set] : from_map)
        {
            for (const auto& to : to_set)
            {
                std::string from_str = "_" + std::to_string(map_world_to_index[from.get_fluent_set()]);
                from_str.insert(0, 1, map_rep_to_name[from.get_repetition()]);
                std::string to_str = "_" + std::to_string(map_world_to_index[to.get_fluent_set()]);
                to_str.insert(0, 1, map_rep_to_name[to.get_repetition()]);
                edges[{from_str, to_str}].insert(m_grounder.deground_agent(ag));
            }
        }
    }

    std::map<std::tuple<std::string, std::string>, std::set<std::string>> to_print_double;
    for (auto it_map = edges.begin(); it_map != edges.end();)
    {
        bool erased = false;
        for (auto it_map_2 = std::next(it_map); it_map_2 != edges.end();)
        {
            if (std::get<0>(it_map->first) == std::get<1>(it_map_2->first) &&
                std::get<1>(it_map->first) == std::get<0>(it_map_2->first) &&
                it_map->second == it_map_2->second &&
                std::get<0>(it_map->first) != std::get<1>(it_map->first))
            {
                to_print_double[it_map->first] = it_map->second;
                edges.erase(it_map_2);
                it_map = edges.erase(it_map);
                erased = true;
                break;
            }
            else
            {
                ++it_map_2;
            }
        }
        if (!erased) ++it_map;
    }

    for (const auto& [key, agents] : edges)
    {
        ofs << "	\"";
        ofs << std::get<0>(key);
        ofs << "\" -> \"";
        ofs << std::get<1>(key);
        ofs << "\" ";
        ofs << "[ label = \"";
        std::string tmp_string;
        for (const auto& ag : agents)
        {
            tmp_string += ag + ",";
        }
        if (!tmp_string.empty()) tmp_string.pop_back();
        ofs << tmp_string;
        ofs << "\" ];\n";
    }

    for (const auto& [key, agents] : to_print_double)
    {
        ofs << "	\"";
        ofs << std::get<0>(key);
        ofs << "\" -> \"";
        ofs << std::get<1>(key);
        ofs << "\" ";
        ofs << "[ dir=both label = \"";
        std::string tmp_string;
        for (const auto& ag : agents)
        {
            tmp_string += ag + ",";
        }
        if (!tmp_string.empty()) tmp_string.pop_back();
        ofs << tmp_string;
        ofs << "\" ];\n";
    }

    std::string color = "<font color=\"#ffffff\">";
    std::string true_fluent_color = "<font color=\"#228B22\">";
    std::string false_fluent_color = "<font color=\"#e53935\">";
    ofs << "\n\n//WORLDS description Table:" << std::endl;
    ofs << "	node [shape = plain]\n\n";
    ofs << "	description[label=<\n";
    ofs << "	<table border = \"0\" cellborder = \"1\" cellspacing = \"0\" >\n";
    for (const auto& world_ptr : worlds)
    {
        bool print_first_done = false;
        auto temp_fs = world_ptr.get_fluent_set();
        std::vector<std::pair<std::string, bool>> sorted_fluents;
        ofs << "		<tr><td>" << map_rep_to_name[world_ptr.get_repetition()] << "_" << map_world_to_index[temp_fs] << "</td> <td>";
        for (const auto& tmp_f : temp_fs)
        {
            bool is_neg = FormulaHelper::is_negated(tmp_f);
            std::string key = m_grounder.deground_fluent(tmp_f);
            if (is_neg && !key.empty())
            {
                key = key.substr(1); // Remove first char if negated
            }
            sorted_fluents.emplace_back(key, is_neg);
        }
        std::ranges::sort(sorted_fluents,
                          [](const auto& a, const auto& b) { return a.first < b.first; });

        for (const auto& [key, is_neg] : sorted_fluents)
        {
            if (print_first_done) ofs << ", ";
            print_first_done = true;
            std::string color_fluent = is_neg ? false_fluent_color : true_fluent_color;
            std::string prefix = is_neg ? NEGATION_SYMBOL : " ";
            ofs << color_fluent << prefix << key << "</font>";
        }
        ofs << "</td></tr>\n";
    }
    ofs << "	</table>>]\n";
    ofs << "	{rank = max; description};\n";

    ofs << "}" << std::endl;
}

void HelperPrint::print_dataset_format(const KripkeState& kstate, std::ofstream& ofs)
{
    std::unordered_map<KripkeWorldId, int> world_map;
    int world_counter = 1;
    const bool use_hash = !ArgumentParser::get_instance().get_dataset_mapped();


    // Assign compact IDs
    for (const auto& pw : kstate.get_worlds())
    {
        if (const auto hash = pw.get_id(); !world_map.contains(hash))
        {
            world_map[hash] = world_counter++;
        }
    }

    ofs << "digraph G {" << std::endl;

    // Print nodes Removed to minimize the size of the dataset
    /*for (const auto& [hash, id] : world_map) {
        ofs << (use_hash ? std::to_string(hash) : adjust_id_wrt_agents(id)) << ";" << std::endl;
    }*/

    // Pointed world
    const auto pointed_hash = kstate.get_pointed().get_id();
    ofs << (use_hash ? std::to_string(pointed_hash) : adjust_id_wrt_agents(world_map[pointed_hash]))
        << " [shape=doublecircle];" << std::endl;

    // Edges
    std::map<std::pair<KripkeWorldId, KripkeWorldId>, std::set<Agent>> edge_map;
    for (const auto& [from_pw, from_map] : kstate.get_beliefs())
    {
        for (const auto& [ag, to_set] : from_map)
        {
            for (const auto& to_pw : to_set)
            {
                auto from_hash = from_pw.get_id();
                auto to_hash = to_pw.get_id();

                edge_map[{from_hash, to_hash}].insert(ag);
            }
        }
    }

    /*     for (const auto& [edge, agents] : edge_map) {
            auto from_label = use_hash ? std::to_string(edge.first) : adjust_id_wrt_agents(world_map[edge.first]);
            auto to_label = use_hash ? std::to_string(edge.second) : adjust_id_wrt_agents(world_map[edge.second]);

            ofs << from_label << " -> " << to_label << " [label=\"";
            bool first = true;
            for (const auto& ag : agents) {
                if (!first) ofs << ",";
                ofs << domain::get_instance().get_grounder().deground_agent(ag);
                first = false;
            }
            ofs << "\"];" << std::endl;
        } */

    //One edge per agent
    const auto gnn_instance = &TrainingDataset<KripkeState>::get_instance();
    for (const auto& [edge, agents] : edge_map)
    {
        auto from_label = use_hash ? std::to_string(edge.first) : adjust_id_wrt_agents(world_map[edge.first]);
        auto to_label = use_hash ? std::to_string(edge.second) : adjust_id_wrt_agents(world_map[edge.second]);


        //Agents will always be printed as integer starting from 0 as in the goal generation
        for (const auto& ag : agents)
        {
            ofs << from_label << " -> " << to_label << " [label=\""
                << gnn_instance->get_unique_a_id_from_map(ag)
                << "\"];" << std::endl;
        }
    }

    ofs << "}" << std::endl;
}


std::string HelperPrint::adjust_id_wrt_agents(const int num)
{
    return std::to_string(num + Domain::get_instance().get_agents().size());
}

std::vector<std::string> HelperPrint::read_actions_from_file(const std::string& filename)
{
    std::ifstream infile(filename);
    if (!infile.is_open())
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::PortfolioConfigFileError,
            "Could not open actions file: " + filename
        );
    }

    std::string line;
    std::vector<std::string> actions;
    while (std::getline(infile, line))
    {
        // Replace all commas with spaces
        std::ranges::replace(line, ',', ' ');

        std::istringstream iss(line);
        std::string action;
        while (iss >> action)
        {
            if (action.empty())
            {
                ExitHandler::exit_with_message(
                    ExitHandler::ExitCode::PortfolioConfigFieldError,
                    "Malformed action (empty) in file: " + filename
                );
            }
            actions.push_back(action);
        }
    }

    if (actions.empty())
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::PortfolioConfigFieldError,
            "No actions found or malformed content in file: " + filename
        );
    }

    return actions;
}


std::string HelperPrint::pretty_print_duration(const std::chrono::duration<double>& duration)
{
    using namespace std::chrono;
    auto ms = duration_cast<milliseconds>(duration).count();
    auto s = ms / 1000;
    ms = ms % 1000;
    auto min = s / 60;
    s = s % 60;
    const auto h = min / 60;
    min = min % 60;

    std::ostringstream oss;
    if (h > 0)
        oss << h << "h ";
    if (min > 0 || h > 0)
        oss << min << "m ";
    oss << s << "s " << ms << "ms";
    return oss.str();
}
