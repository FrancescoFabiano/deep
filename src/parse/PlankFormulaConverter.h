#pragma once

#include <vector>

#include "del/language/formulas.h"
#include "formulae/BeliefFormula.h"

class PlankFormulaConverter {
public:
    PlankFormulaConverter(
        const std::vector<Fluent> &fluents,
        const std::vector<Agent> &agents);

    [[nodiscard]]
    BeliefFormula convert(
        const plank::del::formula_ptr &formula) const;

private:
    const std::vector<Fluent> &m_fluents;
    const std::vector<Agent> &m_agents;

    [[nodiscard]]
    BeliefFormula convert_atom(
        const plank::del::atom_formula_ptr &formula) const;

    [[nodiscard]]
    BeliefFormula convert_not(
        const plank::del::not_formula_ptr &formula) const;

    [[nodiscard]]
    BeliefFormula convert_and(
        const plank::del::and_formula_ptr &formula) const;

    [[nodiscard]]
    BeliefFormula convert_or(
        const plank::del::or_formula_ptr &formula) const;

    [[nodiscard]]
    BeliefFormula convert_box(
        const plank::del::box_formula_ptr &formula) const;

    [[nodiscard]]
    BeliefFormula convert_common(
        const plank::del::c_box_formula_ptr &formula) const;

    [[nodiscard]]
    BeliefFormula convert_nary(
        const plank::del::formula_deque &formulas,
        BeliefFormulaOperator op) const;

    [[nodiscard]]
    AgentsSet convert_agents(
        const plank::del::agent_set &agents) const;

    [[nodiscard]]
    BeliefFormula convert_imply(
    const plank::del::imply_formula_ptr &formula) const;

    [[nodiscard]]
    BeliefFormula convert_diamond(
        const plank::del::diamond_formula_ptr &formula) const;

    [[nodiscard]]
    BeliefFormula convert_kw_box(
        const plank::del::kw_box_formula_ptr &formula) const;

    [[nodiscard]]
    BeliefFormula convert_kw_diamond(
        const plank::del::kw_diamond_formula_ptr &formula) const;

    [[nodiscard]]
    BeliefFormula convert_common_diamond(
        const plank::del::c_diamond_formula_ptr &formula) const;


    [[nodiscard]]
static BeliefFormula make_not(
    const BeliefFormula &formula);

    [[nodiscard]]
    static BeliefFormula make_or(
        const BeliefFormula &left,
        const BeliefFormula &right);

};