#include "StateLevel.h"

#include "FormulaHelper.h"
#include "SetHelper.h"
#include "ExitHandler.h"

#include <ranges>
#include <utility>

StateLevel::StateLevel(const StateLevel& to_assign)
{
    set_state_level(to_assign);
}

StateLevel::StateLevel(const PG_FluentsScoreMap& f_map, const PG_BeliefFormulaeMap& bf_map, unsigned short depth)
{
    set_f_map(f_map);
    set_bf_map(bf_map);
    set_depth(depth);
}

void StateLevel::initialize(const FormulaeList& goal)
{
    build_init_f_map();
    build_init_bf_map(goal);
}

void StateLevel::build_init_f_map()
{
    const auto& initial_fluents = Domain::get_instance().get_initial_description().get_initially_known_fluents();
    for (const auto& fluent : initial_fluents)
    {
        m_pg_f_map.emplace(fluent, 0);
    }

    const auto& fluents = Domain::get_instance().get_fluents();
    for (const auto& fluent : fluents)
    {
        m_pg_f_map.emplace(fluent, -1);
    }
}

void StateLevel::insert_subformula_bf(const FormulaeList& fl, short value)
{
    for (const auto& formula : fl)
    {
        insert_subformula_bf(formula, value);
    }
}

void StateLevel::insert_subformula_bf(const BeliefFormula& bf, short value)
{
    //We set all the subformulas to be TRUE for initially. Maybe it is wrong
    //Maybe we don't need the subformulas at all
    switch (bf.get_formula_type())
    {
    case BeliefFormulaType::BELIEF_FORMULA:
        if (m_pg_bf_map.emplace(bf, value).second)
        {
            insert_subformula_bf(bf.get_bf1(), value);
        }
        break;
    case BeliefFormulaType::PROPOSITIONAL_FORMULA:
        switch (bf.get_operator())
        {
        case BeliefFormulaOperator::BF_NOT:
            if (m_pg_bf_map.emplace(bf, 0).second)
            {
                insert_subformula_bf(bf.get_bf1(), value);
            }
            break;
        case BeliefFormulaOperator::BF_OR:
        case BeliefFormulaOperator::BF_AND:
            if (m_pg_bf_map.emplace(bf, value).second)
            {
                insert_subformula_bf(bf.get_bf1(), value);
                insert_subformula_bf(bf.get_bf2(), value);
            }
            break;
        case BeliefFormulaOperator::BF_FAIL:
        default:
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::BeliefFormulaOperatorUnset,
                "Error: Unexpected operator in PROPOSITIONAL_FORMULA while generating subformulas for the Planning Graph."
            );
        }
        break;
    case BeliefFormulaType::C_FORMULA:
        if (m_pg_bf_map.emplace(bf, value).second)
        {
            insert_subformula_bf(bf.get_bf1(), value);
        }
        break;
    case BeliefFormulaType::FLUENT_FORMULA:
    case BeliefFormulaType::BF_EMPTY:
        break;
    case BeliefFormulaType::BF_TYPE_FAIL:
    default:
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BeliefFormulaTypeUnset,
            "Error: Unexpected formula type in insert_subformula_bf while generating subformulas for the Planning Graph."
        );
    }
}

void StateLevel::build_init_bf_map(const FormulaeList& goals)
{
    insert_subformula_bf(Domain::get_instance().get_initial_description().get_initial_conditions(), 0);
    insert_subformula_bf(goals, -1);

    const auto& actions = Domain::get_instance().get_actions();
    for (const auto& action : actions)
    {
        for (const auto& effects : action.get_effects() | std::views::values)
        {
            insert_subformula_bf(effects, -1);
        }
        if (!action.get_executability().empty())
        {
            insert_subformula_bf(action.get_executability(), -1);
        }
        for (const auto& fully_obs : action.get_fully_observants() | std::views::values)
        {
            insert_subformula_bf(fully_obs, -1);
        }
        for (const auto& part_obs : action.get_partially_observants() | std::views::values)
        {
            insert_subformula_bf(part_obs, -1);
        }
    }
}

void StateLevel::set_f_map(const PG_FluentsScoreMap& to_set)
{
    m_pg_f_map = to_set;
}

void StateLevel::set_bf_map(const PG_BeliefFormulaeMap& to_set)
{
    m_pg_bf_map = to_set;
}

void StateLevel::set_depth(unsigned short to_set)
{
    m_depth = to_set;
}

short StateLevel::get_fluent_value(const Fluent& key) const
{
    if (const auto it = m_pg_f_map.find(key); it != m_pg_f_map.end())
    {
        return it->second;
    }
    return -1;
}

short StateLevel::get_bf_value(const BeliefFormula& key) const
{
    if (auto it = m_pg_bf_map.find(key); it != m_pg_bf_map.end())
    {
        return it->second;
    }
    if (key.get_formula_type() == BeliefFormulaType::FLUENT_FORMULA)
    {
        const FluentFormula& ff_temp = key.get_fluent_formula();
        short ret_val = -1;
        if (ff_temp.size() == 1)
        {
            const auto& fs_temp = *ff_temp.begin();
            for (const auto& fluent : fs_temp)
            {
                if (!pg_entailment(fluent))
                {
                    return -1;
                }
                short tmp_val = m_pg_f_map.at(fluent);
                if (tmp_val > ret_val)
                {
                    ret_val = tmp_val;
                }
            }
        }
        else
        {
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::FormulaNonDeterminismError,
                "The planning BisGraph does not support non-deterministic action yet."
            );
        }
        return ret_val;
    }
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::BeliefFormulaNotGrounded,
        "Found bf formula never declared in the Planning Graph."
    );
    return -1; // Unreachable, but satisfies compiler
}

const PG_FluentsScoreMap& StateLevel::get_f_map() const
{
    return m_pg_f_map;
}

const PG_BeliefFormulaeMap& StateLevel::get_bf_map() const
{
    return m_pg_bf_map;
}

unsigned short StateLevel::get_depth() const
{
    return m_depth;
}

short StateLevel::get_score_from_depth() const
{
    return static_cast<short>(m_depth);
}

void StateLevel::modify_fluent_value(const Fluent& key, short value)
{
    auto it = m_pg_f_map.find(key);
    if (it != m_pg_f_map.end())
    {
        if (it->second < 0)
        {
            it->second = value;
        }
    }
    else
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::DomainUndeclaredFluent,
            "Found Fluent never declared in the Planning Graph."
        );
    }
}

void StateLevel::modify_bf_value(const BeliefFormula& key, short value)
{
    auto it = m_pg_bf_map.find(key);
    if (it != m_pg_bf_map.end())
    {
        if (it->second < 0)
        {
            it->second = value;
        }
    }
    else
    {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BeliefFormulaNotGrounded,
            "Found bf formula never declared in the Planning Graph."
        );
    }
}

bool StateLevel::pg_entailment(const Fluent& f) const
{
    return get_fluent_value(f) >= 0;
}

bool StateLevel::pg_entailment(const BeliefFormula& bf) const
{
    if (bf.get_formula_type() == BeliefFormulaType::BF_EMPTY)
    {
        return true;
    }
    return get_bf_value(bf) >= 0;
}

bool StateLevel::pg_entailment(const FormulaeList& fl) const
{
    return std::ranges::all_of(fl, [this](const auto& formula)
    {
        return pg_entailment(formula);
    });
}

bool StateLevel::pg_executable(const Action& act) const
{
    return pg_entailment(act.get_executability());
}

void StateLevel::get_base_fluents(const BeliefFormula& bf, FluentsSet& bf_base_fluents)
{
    switch (bf.get_formula_type())
    {
    case BeliefFormulaType::FLUENT_FORMULA:
        {
            FluentFormula ff = bf.get_fluent_formula();
            for (const auto& fs : ff)
            {
                for (const auto& fluent : fs)
                {
                    bf_base_fluents.insert(fluent);
                }
            }
            break;
        }
    case BeliefFormulaType::BELIEF_FORMULA:
    case BeliefFormulaType::C_FORMULA:
        get_base_fluents(bf.get_bf1(), bf_base_fluents);
        break;
    case BeliefFormulaType::PROPOSITIONAL_FORMULA:
        switch (bf.get_operator())
        {
        case BeliefFormulaOperator::BF_NOT:
            get_base_fluents(bf.get_bf1(), bf_base_fluents);
            break;
        case BeliefFormulaOperator::BF_OR:
        case BeliefFormulaOperator::BF_AND:
            get_base_fluents(bf.get_bf1(), bf_base_fluents);
            get_base_fluents(bf.get_bf2(), bf_base_fluents);
            break;
        case BeliefFormulaOperator::BF_FAIL:
        default:
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::BeliefFormulaOperatorUnset,
                "Error: Unexpected operator in get_base_fluents while searching for fluents in belief formulas."
            );
        }
        break;
    case BeliefFormulaType::BF_EMPTY:
        break;
    case BeliefFormulaType::BF_TYPE_FAIL:
    default:
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BeliefFormulaTypeUnset,
            "Error: Unexpected formula type in get_base_fluents while searching for fluents in belief formulas."
        );
    }
}

bool StateLevel::compute_successor(const Action& act, const StateLevel& predecessor, FormulaeSet& false_bf)
{
    switch (act.get_type())
    {
    case PropositionType::ONTIC:
        return exec_ontic(act, predecessor, false_bf);
    case PropositionType::SENSING:
    case PropositionType::ANNOUNCEMENT:
        return exec_epistemic(act, predecessor, false_bf);
    default:
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::ActionTypeConflict,
            "Action Type not properly declared (PG building)."
        );
    }
    return false; // Unreachable, but satisfies compiler
}

bool StateLevel::exec_ontic(const Action& act, const StateLevel& predecessor, FormulaeSet& false_bf)
{
    const auto& fully_observant_map = act.get_fully_observants();
    AgentsSet fully_obs;
    const auto& effects_map = act.get_effects();
    FluentsSet verified_fluents;
    FluentsSet bf_base_fluents;
    bool modified_pg = false;

    for (const auto& [ff_temp, effect_formulae] : effects_map)
    {
        if (predecessor.pg_entailment(effect_formulae))
        {
            if (ff_temp.size() == 1)
            {
                const auto& fs_temp = *ff_temp.begin();
                for (const auto& f_tmp : fs_temp)
                {
                    if (!pg_entailment(f_tmp))
                    {
                        modified_pg = true;
                        modify_fluent_value(f_tmp, get_score_from_depth());
                    }
                    verified_fluents.insert(f_tmp);
                }
            }
            else
            {
                ExitHandler::exit_with_message(
                    ExitHandler::ExitCode::FormulaNonDeterminismError,
                    "The planning BisGraph does not support non-deterministic ontic action yet."
                );
            }
        }
    }

    for (const auto& [agent, obs_formula] : fully_observant_map)
    {
        if (predecessor.pg_entailment(obs_formula))
        {
            fully_obs.insert(agent);
        }
    }

    FormulaeSet tmp_fl = false_bf;
    for (const auto& bf : tmp_fl)
    {
        bf_base_fluents.clear();
        get_base_fluents(bf, bf_base_fluents);
        if (!FormulaHelper::fluentset_empty_intersection(verified_fluents, bf_base_fluents))
        {
            bool tmp_modified = false;
            apply_ontic_effects(bf, false_bf, fully_obs, tmp_modified);
            if (!modified_pg && tmp_modified)
            {
                modified_pg = true;
            }
        }
    }

    return modified_pg;
}

bool StateLevel::exec_epistemic(const Action& act, const StateLevel& predecessor, FormulaeSet& false_bf)
{
    const auto& partially_observant_map = act.get_partially_observants();
    const auto& fully_observant_map = act.get_fully_observants();
    AgentsSet fully_obs;
    AgentsSet partially_obs;
    const auto& effects_map = act.get_effects();
    FluentsSet bf_base_fluents;

    for (const auto& [agent, obs_formula] : fully_observant_map)
    {
        if (predecessor.pg_entailment(obs_formula))
        {
            fully_obs.insert(agent);
        }
    }
    for (const auto& [agent, obs_formula] : partially_observant_map)
    {
        if (predecessor.pg_entailment(obs_formula))
        {
            partially_obs.insert(agent);
        }
    }

    bool modified_pg = false;
    FormulaeSet tmp_fl = false_bf;
    for (const auto& bf : tmp_fl)
    {
        bool tmp_modified = false;
        bf_base_fluents.clear();
        get_base_fluents(bf, bf_base_fluents);

        for (const auto& [ff_temp, effect_formulae] : effects_map)
        {
            if (predecessor.pg_entailment(effect_formulae))
            {
                if (ff_temp.size() == 1)
                {
                    const auto& fs_temp = *ff_temp.begin();
                    for (const auto& f_tmp : fs_temp)
                    {
                        if (bf_base_fluents.contains(f_tmp))
                        {
                            apply_epistemic_effects(f_tmp, bf, false_bf, fully_obs, partially_obs, tmp_modified, 0);
                            if (!modified_pg && tmp_modified)
                            {
                                modified_pg = true;
                            }
                        }
                    }
                }
            }
        }
    }
    return modified_pg;
}

bool StateLevel::apply_ontic_effects(const BeliefFormula& bf, FormulaeSet& fl, const AgentsSet& fully,
                                     bool& modified_pg)
{
    if (pg_entailment(bf))
    {
        return true;
    }

    switch (bf.get_formula_type())
    {
    case BeliefFormulaType::FLUENT_FORMULA:
        {
            FluentFormula ff = bf.get_fluent_formula();
            for (const auto& fs : ff)
            {
                bool ret = true;
                for (const auto& fluent : fs)
                {
                    if (!pg_entailment(fluent))
                    {
                        ret = false;
                        break;
                    }
                }
                if (ret)
                {
                    return true;
                }
            }
            break;
        }
    case BeliefFormulaType::BELIEF_FORMULA:
        {
            if (fully.contains(bf.get_agent()))
            {
                if (apply_ontic_effects(bf.get_bf1(), fl, fully, modified_pg))
                {
                    modified_pg = true;
                    modify_bf_value(bf, get_score_from_depth());
                    fl.erase(bf);
                    return true;
                }
            }
            break;
        }
    case BeliefFormulaType::PROPOSITIONAL_FORMULA:
        switch (bf.get_operator())
        {
        case BeliefFormulaOperator::BF_NOT:
            return pg_entailment(bf);
        case BeliefFormulaOperator::BF_OR:
            if (apply_ontic_effects(bf.get_bf1(), fl, fully, modified_pg) ||
                apply_ontic_effects(bf.get_bf2(), fl, fully, modified_pg))
            {
                modified_pg = true;
                modify_bf_value(bf, get_score_from_depth());
                fl.erase(bf);
                return true;
            }
            break;
        case BeliefFormulaOperator::BF_AND:
            if (apply_ontic_effects(bf.get_bf1(), fl, fully, modified_pg) &&
                apply_ontic_effects(bf.get_bf2(), fl, fully, modified_pg))
            {
                modified_pg = true;
                modify_bf_value(bf, get_score_from_depth());
                fl.erase(bf);
                return true;
            }
            break;
        case BeliefFormulaOperator::BF_FAIL:
        default:
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::BeliefFormulaOperatorUnset,
                "Error: Unexpected operator in apply_ontic_effects while searching for fluents in belief formulas."
            );
        }
        break;
    case BeliefFormulaType::C_FORMULA:
        {
            for (const auto& ag : bf.get_group_agents())
            {
                if (!fully.contains(ag))
                {
                    return false;
                }
            }
            if (apply_ontic_effects(bf.get_bf1(), fl, fully, modified_pg))
            {
                modified_pg = true;
                modify_bf_value(bf, get_score_from_depth());
                fl.erase(bf);
                return true;
            }
            break;
        }
    case BeliefFormulaType::BF_EMPTY:
        return true;
    case BeliefFormulaType::BF_TYPE_FAIL:
    default:
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BeliefFormulaTypeUnset,
            "Error: Unexpected formula type in apply_ontic_effects while searching for fluents in belief formulas."
        );
    }
    return false;
}

bool StateLevel::apply_epistemic_effects(
    const Fluent& effect,
    const BeliefFormula& bf,
    FormulaeSet& fl,
    const AgentsSet& fully,
    const AgentsSet& partially,
    bool& modified_pg,
    unsigned short vis_cond)
{
    if (pg_entailment(bf))
    {
        return true;
    }

    unsigned short temp_vis_cond = vis_cond;
    bool tmp_ret1 = false;
    bool tmp_ret2 = false;

    switch (bf.get_formula_type())
    {
    case BeliefFormulaType::FLUENT_FORMULA:
        {
            for (const FluentFormula& ff = bf.get_fluent_formula(); const auto& fs : ff)
            {
                bool ret = true;
                for (const auto& fluent : fs)
                {
                    if (!pg_entailment(fluent))
                    {
                        if (!((fluent == effect && (vis_cond < 2)) ||
                            (fluent == FormulaHelper::negate_fluent(effect) && (vis_cond == 1))))
                        {
                            ret = false;
                            break;
                        }
                    }
                }
                if (ret)
                {
                    return true;
                }
            }
            break;
        }
    case BeliefFormulaType::BELIEF_FORMULA:
        {
            if (fully.contains(bf.get_agent()))
            {
                if (vis_cond == 2)
                {
                    temp_vis_cond = 1;
                }
                if (apply_epistemic_effects(effect, bf.get_bf1(), fl, fully, partially, modified_pg, temp_vis_cond))
                {
                    modified_pg = true;
                    modify_bf_value(bf, get_score_from_depth());
                    fl.erase(bf);
                    tmp_ret1 = true;
                }
            }
            if (partially.contains(bf.get_agent()))
            {
                if (apply_epistemic_effects(effect, bf.get_bf1(), fl, fully, partially, modified_pg, 2))
                {
                    modified_pg = true;
                    modify_bf_value(bf, get_score_from_depth());
                    fl.erase(bf);
                    tmp_ret2 = true;
                }
            }
            return tmp_ret1 || tmp_ret2;
        }
    case BeliefFormulaType::PROPOSITIONAL_FORMULA:
        switch (bf.get_operator())
        {
        case BeliefFormulaOperator::BF_NOT:
            return pg_entailment(bf);
        case BeliefFormulaOperator::BF_OR:
            if (apply_epistemic_effects(effect, bf.get_bf1(), fl, fully, partially, modified_pg, vis_cond) ||
                apply_epistemic_effects(effect, bf.get_bf2(), fl, fully, partially, modified_pg, vis_cond))
            {
                modified_pg = true;
                modify_bf_value(bf, get_score_from_depth());
                fl.erase(bf);
                return true;
            }
            break;
        case BeliefFormulaOperator::BF_AND:
            if (apply_epistemic_effects(effect, bf.get_bf1(), fl, fully, partially, modified_pg, vis_cond) &&
                apply_epistemic_effects(effect, bf.get_bf2(), fl, fully, partially, modified_pg, vis_cond))
            {
                modified_pg = true;
                modify_bf_value(bf, get_score_from_depth());
                fl.erase(bf);
                return true;
            }
            break;
        case BeliefFormulaOperator::BF_FAIL:
        default:
            ExitHandler::exit_with_message(
                ExitHandler::ExitCode::BeliefFormulaOperatorUnset,
                "Error: Unexpected operator in apply_epistemic_effects while searching for fluents in belief formulas."
            );
        }
        break;
    case BeliefFormulaType::C_FORMULA:
        {
            bool only_fully = true;
            bool one_partial = false;
            for (const auto& ag : bf.get_group_agents())
            {
                if (!fully.contains(ag))
                {
                    if (!partially.contains(ag))
                    {
                        return false;
                    }
                    else
                    {
                        only_fully = false;
                        one_partial = true;
                    }
                }
                else if (!one_partial)
                {
                    one_partial = partially.contains(ag);
                }
            }
            if (only_fully)
            {
                if (vis_cond == 2)
                {
                    temp_vis_cond = 1;
                }
                if (apply_epistemic_effects(effect, bf.get_bf1(), fl, fully, partially, modified_pg, temp_vis_cond))
                {
                    modified_pg = true;
                    modify_bf_value(bf, get_score_from_depth());
                    fl.erase(bf);
                    tmp_ret1 = true;
                }
            }
            if (one_partial)
            {
                if (apply_epistemic_effects(effect, bf.get_bf1(), fl, fully, partially, modified_pg, 2))
                {
                    modified_pg = true;
                    modify_bf_value(bf, get_score_from_depth());
                    fl.erase(bf);
                    tmp_ret2 = true;
                }
            }
            return tmp_ret1 || tmp_ret2;
        }
    case BeliefFormulaType::BF_EMPTY:
        return true;
    case BeliefFormulaType::BF_TYPE_FAIL:
    default:
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::BeliefFormulaTypeUnset,
            "Error: Unexpected formula type in apply_epistemic_effects while searching for fluents in belief formulas."
        );
    }
    return false;
}

StateLevel& StateLevel::operator=(const StateLevel& to_assign)
{
    set_state_level(to_assign);
    return *this;
}

void StateLevel::set_state_level(const StateLevel& to_assign)
{
    set_f_map(to_assign.get_f_map());
    set_bf_map(to_assign.get_bf_map());
    set_depth(to_assign.get_depth());
}
