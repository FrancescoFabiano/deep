/*
 * \brief Implementation of \ref Reader.h.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano.
 * \date May 17, 2025
 */

#include "Reader.h"

#include "ArgumentParser.h"
#include "../utilities/HelperPrint.h"

// prototype of bison-generated parser function
int yyparse();

std::unique_ptr<Reader> domain_reader;

Reader::Reader() = default;

int Reader::read()
{
    // Call to the parser function.
    return yyparse();
}

/**
 * \brief Print out all the information that are stored as std::string in the reader object.
 * \see HelperPrint, BeliefFormula::print() const, Proposition::print() const
 */
void Reader::print() const
{
    auto& os = ArgumentParser::get_instance().get_output_stream();
    os << "\n\nAGENT DECLARATION\n---------------------------\n";
    HelperPrint::print_list(m_agents);
    os << "\n\n";

    os << "FLUENT DECLARATION\n----------------------------\n";
    HelperPrint::print_list(m_fluents);
    os << "\n\n";

    os << "PROPOSITIONS\n----------------------------\n";
    for (const auto& prop : m_propositions)
    {
        prop.print();
        os << '\n';
    }

    os << "INIT\n----------------------------\n";
    for (const auto& formula : m_bf_initially)
    {
        formula.print();
        os << '\n';
    }
    os << '\n';

    os << "GOAL \n----------------------------\n";
    for (const auto& formula : m_bf_goal)
    {
        formula.print();
        os << '\n';
    }
    os << "\n\n\n";

    /*
    // print statistics
    os << "STATISTICS \n----------------------------\n";
    //os << "Total actions: " << m_actions.size() << std::endl;
    //os << "\tSensing actions: " << sensing_actions.size() << std::endl; //Ben
    //os << "\tOntic actions: " << ontic_actions.size() << std::endl;    //Ben
    //os << "\tAnnouncement actions: " << ann_actions.size() << std::endl; //Ben
    os << "Total fluents: " << m_fluents.size() << std::endl;
    //os << "Unknown fluents: " << std::endl;
    unsigned int i = 0;
    for (const auto& init_set : m_initially) {
        os << "\tState " << i++ << ": ";
        os << m_fluents.size() - init_set.size();
        os << std::endl;
    }
    //os << "done" << std::endl;
    */
}
