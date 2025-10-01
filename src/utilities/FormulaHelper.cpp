#include "FormulaHelper.h"
#include <cmath>
#include <ranges>

#include "ArgumentParser.h"
#include "Domain.h"
#include "ExitHandler.h"
#include "HelperPrint.h"
#include "KripkeEntailmentHelper.h"
#include "State.h"
#include "states/representations/kripke/KripkeState.h"

/**
 * \file FormualaHelper.cpp
 * \brief Implementation of FormulaHelper.h
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 2025
 */

Fluent FormulaHelper::negate_fluent(const Fluent &to_negate) {
  Fluent fluent_negated = to_negate;
  // Negate the last bit of the fluent, which represents its polarity.
  // If the last bit is false (negated), set it to true (positive), otherwise
  // set it to false (negated).
  const bool last_bit = to_negate[to_negate.size() - 1];
  fluent_negated.set(to_negate.size() - 1, !last_bit);
  return fluent_negated;
}

FluentFormula
FormulaHelper::negate_fluent_formula(const FluentFormula &to_negate) {
  if (to_negate.size() > 1) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::FormulaNonDeterminismError,
        "Error: Non-determinism is not supported yet in "
        "negate_fluent_formula.");
  } else if (to_negate.size() == 1) {
    const auto &sub_ff = *to_negate.begin();
    if (sub_ff.size() > 1) {
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::FormulaNonDeterminismError,
          "Error: You cannot negate multiple effects because non-determinism "
          "is not supported yet.");
    } else if (sub_ff.size() == 1) {
      FluentFormula neg_ff;
      FluentsSet neg_fs;
      neg_fs.insert(FormulaHelper::negate_fluent(*sub_ff.begin()));
      neg_ff.insert(neg_fs);
      return neg_ff;
    }
  }
  return to_negate;
}

Fluent FormulaHelper::normalize_fluent(const Fluent &to_normalize) {
  return is_negated(to_normalize) ? negate_fluent(to_normalize) : to_normalize;
}

bool FormulaHelper::is_negated(const Fluent &f) {
  return f[f.size() - 1] == false;
}

bool FormulaHelper::is_consistent(const FluentsSet &fl1,
                                  const FluentsSet &fl2) {
  return std::ranges::all_of(
      fl2, [&](const Fluent &f) { return !fl1.contains(negate_fluent(f)); });
}

FluentsSet FormulaHelper::and_ff(const FluentsSet &fl1, const FluentsSet &fl2) {
  FluentsSet ret;
  if (!fl1.empty() && !fl2.empty()) {
    if (is_consistent(fl1, fl2)) {
      ret = fl1;
      ret.insert(fl2.begin(), fl2.end());
    }
  } else if (fl1.empty()) {
    return fl2;
  } else if (fl2.empty()) {
    return fl1;
  } else {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::FormulaBadDeclaration,
        "Bad formula declaration in and_ff(FluentsSet, FluentsSet).");
  }
  return ret;
}

FluentFormula FormulaHelper::and_ff(const FluentFormula &to_merge_1,
                                    const FluentFormula &to_merge_2) {
  FluentFormula ret;
  if (!to_merge_1.empty() && !to_merge_2.empty()) {
    for (const auto &fs1 : to_merge_1) {
      for (const auto &fs2 : to_merge_2) {
        ret.insert(and_ff(fs1, fs2));
      }
    }
  } else if (to_merge_1.empty()) {
    return to_merge_2;
  } else if (to_merge_2.empty()) {
    return to_merge_1;
  } else {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::FormulaBadDeclaration,
        "Bad formula declaration in and_ff(FluentFormula, FluentFormula).");
  }
  return ret;
}

bool FormulaHelper::check_Bff_notBff(const BeliefFormula &to_check_1,
                                     const BeliefFormula &to_check_2,
                                     FluentFormula &ret) {
  if (to_check_1.get_formula_type() == BeliefFormulaType::BELIEF_FORMULA &&
      to_check_2.get_formula_type() == BeliefFormulaType::BELIEF_FORMULA) {
    const auto &to_check_nested_1 = to_check_1.get_bf1();
    const auto &to_check_nested_2 = to_check_2.get_bf1();

    if (to_check_nested_1.get_formula_type() ==
            BeliefFormulaType::FLUENT_FORMULA &&
        to_check_nested_2.get_formula_type() ==
            BeliefFormulaType::PROPOSITIONAL_FORMULA) {
      if (to_check_nested_2.get_operator() == BeliefFormulaOperator::BF_NOT) {
        auto tmp = *to_check_nested_1.get_fluent_formula().begin();
        const auto f1 = *tmp.begin();
        tmp = *to_check_nested_2.get_bf1().get_fluent_formula().begin();
        const auto f2 = *tmp.begin();
        if (f1 == f2) {
          ret.insert(tmp);
          return true;
        }
      }
    } else if (to_check_nested_2.get_formula_type() ==
                   BeliefFormulaType::FLUENT_FORMULA &&
               to_check_nested_1.get_formula_type() ==
                   BeliefFormulaType::PROPOSITIONAL_FORMULA) {
      if (to_check_nested_1.get_operator() == BeliefFormulaOperator::BF_NOT) {
        auto tmp = *to_check_nested_1.get_bf1().get_fluent_formula().begin();
        const auto f1 = *tmp.begin();
        tmp = *to_check_nested_2.get_fluent_formula().begin();
        const auto f2 = *tmp.begin();
        if (f1 == f2) {
          ret.insert(tmp);
          return true;
        }
      }
    }
  }
  return false;
}

void FormulaHelper::apply_effect(const Fluent &effect,
                                 FluentsSet &world_description) {
  world_description.erase(negate_fluent(effect));
  world_description.insert(effect);
}

void FormulaHelper::apply_effect(const FluentsSet &effect,
                                 FluentsSet &world_description) {
  for (const auto &f : effect) {
    apply_effect(f, world_description);
  }
}

void FormulaHelper::apply_effect(const FluentFormula &effect,
                                 FluentsSet &world_description) {
  if (effect.size() == 1) {
    apply_effect(*effect.begin(), world_description);
  } else if (effect.size() > 1) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::FormulaNonDeterminismError,
        "Non determinism in action effect is not supported.");
  } else {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::FormulaEmptyEffect,
                                   "Empty action effect.");
  }
}

int FormulaHelper::length_to_power_two(int length) {
  return static_cast<int>(std::ceil(std::log2(length)));
}

bool FormulaHelper::fluentset_empty_intersection(const FluentsSet &set1,
                                                 const FluentsSet &set2) {
  auto first1 = set1.begin();
  auto first2 = set2.begin();
  auto last1 = set1.end();
  auto last2 = set2.end();

  while (first1 != last1 && first2 != last2) {
    if (*first1 < *first2)
      ++first1;
    else if (*first2 < *first1)
      ++first2;
    else
      return false;
  }
  return true;
}

bool FormulaHelper::fluentset_negated_empty_intersection(
    const FluentsSet &set1, const FluentsSet &set2) {
  for (const auto &f1 : set1) {
    Fluent negated_f1 = negate_fluent(f1);
    for (const auto &f2 : set2) {
      if (f1 == f2 || negated_f1 == f2) {
        return false;
      }
    }
  }
  return true;
}

AgentsSet FormulaHelper::get_agents_if_entailed(const ObservabilitiesMap &map,
                                                const KripkeState &state) {
  AgentsSet ret;
  for (const auto &[agent, formula] : map) {
    if (KripkeEntailmentHelper::entails(formula, state)) {
      ret.insert(agent);
    }
  }
  return ret;
}

FluentFormula FormulaHelper::get_effects_if_entailed(const EffectsMap &map,
                                                     const KripkeState &state) {
  FluentFormula ret;
  for (const auto &[effect, formula] : map) {
    if (KripkeEntailmentHelper::entails(formula, state)) {
      ret = FormulaHelper::and_ff(ret, effect);
    }
  }
  if (ret.size() > 1) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::FormulaNonDeterminismError,
        "Non determinism in action effect is not supported "
        "(get_effects_if_entailed).");
  }
  return ret;
}

boost::dynamic_bitset<>
FormulaHelper::concatStringDyn(const boost::dynamic_bitset<> &bs1,
                               const boost::dynamic_bitset<> &bs2) {
  std::string s1;
  std::string s2;

  to_string(bs1, s1);
  to_string(bs2, s2);
  boost::dynamic_bitset<> res(s1 + s2);
  return res;
}

boost::dynamic_bitset<>
FormulaHelper::concatOperatorsDyn(const boost::dynamic_bitset<> &bs1,
                                  const boost::dynamic_bitset<> &bs2) {
  boost::dynamic_bitset<> bs1Copy(bs1);
  boost::dynamic_bitset<> bs2Copy(bs2);
  size_t totalSize = bs1.size() + bs2.size();
  bs1Copy.resize(totalSize);
  bs2Copy.resize(totalSize);
  bs1Copy <<= bs2.size();
  bs1Copy |= bs2Copy;
  return bs1Copy;
}

boost::dynamic_bitset<>
FormulaHelper::concatLoopDyn(const boost::dynamic_bitset<> &bs1,
                             const boost::dynamic_bitset<> &bs2) {
  boost::dynamic_bitset<> res(bs1);
  res.resize(bs1.size() + bs2.size());
  size_t bs1Size = bs1.size();

  for (size_t i = 0; i < bs2.size(); i++)
    res[i + bs1Size] = bs2[i];
  return res;
}

KripkeWorldId FormulaHelper::hash_fluents_into_id(const FluentsSet &fl) {
  return boost::hash_range(fl.begin(), fl.end());
}

KripkeWorldId FormulaHelper::hash_string_into_id(const std::string &string) {
  return boost::hash_range(string.begin(), string.end());
}

bool FormulaHelper::consistent(const FluentsSet &to_check) {
  for (auto it = to_check.begin(); it != to_check.end(); ++it) {
    /* If the pointed fluent is in modulo 2 it means is the positive and if
     * its successor (the negative version) is in the set then is not
     * consistent.*/
    Fluent neg = negate_fluent(*it);
    if (auto clash = to_check.find(neg); clash != to_check.end()) {
      std::string error =
          "Consistency check failed in FormulaHelper::consistent: set contains "
          "a fluent and its negation.\n";
      error += "Clashing fluents: \"";
      error += HelperPrint::get_instance().get_grounder().deground_fluent(*it);
      error += "\" and \"";
      error += HelperPrint::get_instance().get_grounder().deground_fluent(neg);
      // oss << "\nFull set: ";
      // HelperPrint::get_instance().print_list(to_check, oss);
      ExitHandler::exit_with_message(
          ExitHandler::ExitCode::FormulaConsistencyError, error);
    }
  }
  return true;
}

void FormulaHelper::checkSameKState(const KripkeState &first,
                                    const KripkeState &second) {

  bool are_bisimilar = true;

  auto &os = ArgumentParser::get_instance().get_output_stream();

  // ReSharper disable once CppDFAConstantConditions
  if (second == first) {
    // If the state is already bisimilar, no need to check further
    return;
  }
  // ReSharper disable once CppDFAUnreachableCode
  os << "[DEBUG] Checking equivalence for possibly different "
        "states.";

  std::string fail_case;

  auto &domain_instance = Domain::get_instance();
  auto to_check1 =
      domain_instance.get_initial_description().get_initial_conditions();
  if (first.entails(to_check1) != second.entails(to_check1)) {
    are_bisimilar = false;
    fail_case = "initial_conditions";
  }

  auto to_check2 = domain_instance.get_initial_description().get_ff_forS5();
  // ReSharper disable once CppDFAUnreachableCode
  // ReSharper disable once CppDFAUnreachableCode
  if (!to_check2.empty() &&
      (first.entails(to_check2) != second.entails(to_check2))) {
    are_bisimilar = false;
    fail_case = "ff_forS5";
  }

  auto to_check3 = domain_instance.get_goal_description();
  if (first.entails(to_check3) != second.entails(to_check3)) {
    are_bisimilar = false;
    fail_case = "goal_description";
  }

  for (const auto &tmp_action : domain_instance.get_actions()) {
    for (auto condition : tmp_action.get_effects() | std::views::values) {
      if (first.entails(condition) != second.entails(condition)) {
        are_bisimilar = false;
        fail_case = "action_effects of action " + tmp_action.get_name();
      }
    }
    auto to_check5 = tmp_action.get_executability();
    if (first.entails(to_check5) != second.entails(to_check5)) {
      are_bisimilar = false;
      fail_case = "action_executability of action  " + tmp_action.get_name();
    }
    for (auto condition :
         tmp_action.get_fully_observants() | std::views::values) {
      if (first.entails(condition) != second.entails(condition)) {
        are_bisimilar = false;
        fail_case = "Full Observability of action " + tmp_action.get_name();
      }
    }
    for (auto condition :
         tmp_action.get_partially_observants() | std::views::values) {
      if (first.entails(condition) != second.entails(condition)) {
        are_bisimilar = false;
        fail_case = "Full Observability of action " + tmp_action.get_name();
      }
    }
  }

  if (!are_bisimilar) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::SearchBisimulationError,
        "Bisimulation reduction failed: there is some discrepancy in " +
            fail_case + ". Use debugger to investigate.");
  }
  os << " All good:)" << std::endl;
}