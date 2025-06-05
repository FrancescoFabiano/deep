/*
 * \brief Class that implements PlanningGraph.h.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 31, 2025
 */

#include "PlanningGraph.h"
#include "ArgumentParser.h"
#include "HelperPrint.h"
#include "utilities/ExitHandler.h"


PlanningGraph::PlanningGraph(const PlanningGraph& pg)
{
    set_pg(pg);
}


void PlanningGraph::init(const FormulaeList& goal, const StateLevel& pg_init)
{
    // Start timing for planning graph construction
    if (ArgumentParser::get_instance().get_debug())
    {
        m_clocks.start_clock1 = std::chrono::system_clock::now();
    }

    set_goal(goal);
    m_state_levels.push_back(pg_init);
    const auto& init_bf_score = pg_init.get_bf_map();
    for (const auto& [bf, score] : init_bf_score)
    {
        if (score < 0)
        {
            m_belief_formula_false.insert(bf);
        }
    }

    m_never_executed = Domain::get_instance().get_actions();
    set_length(0);
    set_sum(0);

    // Start timing for initial goal check
    if (ArgumentParser::get_instance().get_debug())
    {
        m_clocks.start_clock2 = std::chrono::system_clock::now();
    }

    bool not_goal = false;
    for (auto it_fl = m_goal.begin(); it_fl != m_goal.end();)
    {
        if (m_state_levels.back().pg_entailment(*it_fl))
        {
            it_fl = m_goal.erase(it_fl);
        }
        else
        {
            not_goal = true;
            ++it_fl;
        }
    }

    // End timing for initial goal check and reset other timers
    if (ArgumentParser::get_instance().get_debug())
    {
        const auto end_pg_goal_ini = std::chrono::system_clock::now();
        m_clocks.t5 = end_pg_goal_ini - m_clocks.start_clock2;
        m_clocks.t1 = std::chrono::milliseconds::zero();
        m_clocks.t2 = std::chrono::milliseconds::zero();
        m_clocks.t3 = std::chrono::milliseconds::zero();
        m_clocks.t4 = std::chrono::milliseconds::zero();
    }

    if (not_goal)
    {
        pg_build();
    }
    else
    {
        set_satisfiable(true);
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::PlanningGraphErrorInitialState,
            "BUILDING: The initial state is goal. PlanningGraph construction terminated early. You should check if the state is goal before creating the planning graph"
        );
    }

    // End timing for planning graph construction
    if (ArgumentParser::get_instance().get_debug())
    {
        auto& os = ArgumentParser::get_instance().get_output_stream();
        const auto end_pg_build = std::chrono::system_clock::now();
        std::chrono::duration<double> pg_build_time = end_pg_build - m_clocks.start_clock1;
        os << "\n\nGenerated Planning Graph of length " << get_length() << " in " << pg_build_time.count()
            << " seconds of which:";
        os << "\nFirst goal check:      " << m_clocks.t5.count();
        os << "\nAction Level creation: " << m_clocks.t1.count();
        os << "\n\nState Level Creation:  " << m_clocks.t2.count() << " of which:";
        os << "\nActions Execution:     " << m_clocks.t4.count();
        os << "\n\nGoals Check:           " << m_clocks.t3.count() << std::endl;
    }
}

void PlanningGraph::set_satisfiable(const bool sat)
{
    m_satisfiable = sat;
}

bool PlanningGraph::is_satisfiable() const
{
    return m_satisfiable;
}

void PlanningGraph::pg_build()
{
    const StateLevel s_level_curr = m_state_levels.back();
    ActionLevel a_level_curr;
    a_level_curr.set_depth(get_length());

    // Start timing for action level creation
    if (ArgumentParser::get_instance().get_debug())
    {
        m_clocks.start_clock1 = std::chrono::system_clock::now();
    }

    if (!m_action_levels.empty())
    {
        a_level_curr = m_action_levels.back();
    }

    for (auto it_actions_set = m_never_executed.begin(); it_actions_set != m_never_executed.end();)
    {
        if (s_level_curr.pg_executable(*it_actions_set))
        {
            a_level_curr.add_action(*it_actions_set);
            it_actions_set = m_never_executed.erase(it_actions_set);
        }
        else
        {
            ++it_actions_set;
        }
    }

    // End timing for action level creation
    if (ArgumentParser::get_instance().get_debug())
    {
        m_clocks.t1 += std::chrono::system_clock::now() - m_clocks.start_clock1;
    }

    add_action_level(a_level_curr);
    set_length(get_length() + 1);

    // The no-op is done with the copy
    StateLevel s_level_next = s_level_curr;
    s_level_next.set_depth(get_length());
    bool new_state_insertion = false;

    // Start timing for state level creation
    if (ArgumentParser::get_instance().get_debug())
    {
        m_clocks.start_clock1 = std::chrono::system_clock::now();
    }

    // Print debug info for planning graph length and timings
    if (ArgumentParser::get_instance().get_debug())
    {
        auto& os = ArgumentParser::get_instance().get_output_stream();
        os << "\n\nPlaning Graph Length: " << get_length();
        if (get_length() > 0)
        {
            os << "\nAction Level creation: " << m_clocks.t1.count();
            os << "\nState Level Creation:  " << m_clocks.t2.count() << " of which:";
            os << "\nActions execution:     " << m_clocks.t4.count();
        }
    }

    for (const auto& action : a_level_curr.get_actions())
    {
        // Start timing for action execution
        if (ArgumentParser::get_instance().get_debug())
        {
            m_clocks.start_clock2 = std::chrono::system_clock::now();
        }

        if (s_level_next.compute_successor(action, s_level_curr, m_belief_formula_false))
        {
            new_state_insertion = true;
        }

        // End timing for action execution
        if (ArgumentParser::get_instance().get_debug())
        {
            m_clocks.t4 += std::chrono::system_clock::now() - m_clocks.start_clock2;
        }
    }

    // End timing for state level creation
    if (ArgumentParser::get_instance().get_debug())
    {
        m_clocks.t2 += std::chrono::system_clock::now() - m_clocks.start_clock1;
    }

    add_state_level(s_level_next);

    bool not_goal = false;

    // Start timing for goal check
    if (ArgumentParser::get_instance().get_debug())
    {
        m_clocks.start_clock1 = std::chrono::system_clock::now();
    }

    // Remove each subgoal already satisfied: it will always be. We just need to check it once
    for (auto it_fl = m_goal.begin(); it_fl != m_goal.end();)
    {
        if (s_level_next.pg_entailment(*it_fl))
        {
            it_fl = m_goal.erase(it_fl);
            m_pg_sum += get_length();
        }
        else
        {
            ++it_fl;
            not_goal = true;
        }
    }

    // End timing for goal check
    if (ArgumentParser::get_instance().get_debug())
    {
        m_clocks.t3 += std::chrono::system_clock::now() - m_clocks.start_clock1;
    }

    if (!not_goal)
    {
        set_satisfiable(true);
        return;
    }
    else if (!new_state_insertion)
    {
        set_satisfiable(false);
        if (ArgumentParser::get_instance().get_debug())
        {
            if (get_length() > 5)
            {
                print();
            }
        }
        return;
    }
    else
    {
        pg_build();
    }
}

void PlanningGraph::add_state_level(const StateLevel& s_level)
{
    m_state_levels.push_back(s_level);
}

void PlanningGraph::add_action_level(const ActionLevel& a_level)
{
    m_action_levels.push_back(a_level);
}

void PlanningGraph::set_length(const unsigned short length)
{
    m_pg_length = length;
}

void PlanningGraph::set_sum(const unsigned short sum)
{
    m_pg_sum = sum;
}

unsigned short PlanningGraph::get_length() const
{
    return m_pg_length;
}

unsigned short PlanningGraph::get_sum() const
{
    return m_pg_sum;
}

void PlanningGraph::set_goal(const FormulaeList& goal)
{
    m_goal = goal;
}

const std::vector<StateLevel>& PlanningGraph::get_state_levels() const
{
    return m_state_levels;
}

const std::vector<ActionLevel>& PlanningGraph::get_action_levels() const
{
    return m_action_levels;
}

const PG_FluentsScoreMap& PlanningGraph::get_f_scores() const
{
    return m_state_levels.back().get_f_map();
}

const PG_BeliefFormulaeMap& PlanningGraph::get_bf_scores() const
{
    return m_state_levels.back().get_bf_map();
}

const FormulaeList& PlanningGraph::get_goal() const
{
    return m_goal;
}

const ActionsSet& PlanningGraph::get_never_executed() const
{
    return m_never_executed;
}

const FormulaeSet& PlanningGraph::get_belief_formula_false() const
{
    return m_belief_formula_false;
}

PlanningGraph& PlanningGraph::operator=(const PlanningGraph& to_assign)
{
    set_pg(to_assign);
    return *this;
}

void PlanningGraph::set_pg(const PlanningGraph& to_assign)
{
    m_state_levels = to_assign.get_state_levels();
    m_action_levels = to_assign.get_action_levels();
    m_pg_length = to_assign.get_length();
    m_pg_sum = to_assign.get_sum();
    m_satisfiable = to_assign.is_satisfiable();
    m_goal = to_assign.get_goal();
    m_never_executed = to_assign.get_never_executed();
    m_belief_formula_false = to_assign.get_belief_formula_false();
}

void PlanningGraph::print() const
{
    auto& os = ArgumentParser::get_instance().get_output_stream();

    os << "\n\n**********ePLANNING-GRAPH PRINT**********\n";
    unsigned short count = 0;
    for (const auto& m_state_level : m_state_levels)
    {
        const auto& fluents_score = m_state_level.get_f_map();
        const auto& bf_score = m_state_level.get_bf_map();

        os << "\n\t*******State Level " << count << "*******\n";
        os << "\n\t\t****Fluents****\n\n";
        for (const auto& [fluent, score] : fluents_score)
        {
            os << "\t\t\t" << HelperPrint::get_instance().get_grounder().deground_fluent(fluent) << " -> " << score <<
                std::endl;
        }
        os << "\n\t\t****Belief Formulae****\n\n";
        for (const auto& [bf, score] : bf_score)
        {
            os << "\t\t\t";
            bf.print();
            os << " -> " << score << std::endl;
        }
        os << "\n\t*******End State Level " << count << "*******\n";

        if (count < m_action_levels.size())
        {
            os << "\n\t*******Action Level " << count << "*******\n";
            const auto& act_set = m_action_levels.at(count).get_actions();
            for (const auto& act : act_set)
            {
                os << "\n\t\t" << act.get_name() << std::endl;
            }
            os << "\n\t*******End Action Level " << count++ << "*******\n";
        }
    }
    os << "\n*********END ePLANNING-GRAPH PRINT**********\n";
}
