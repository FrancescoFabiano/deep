/**
 * \brief Implementation of \ref Domain.h.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date May 14, 2025
 */
#include "Domain.h"
#include <boost/dynamic_bitset.hpp>

#include "ArgumentParser.h"
#include "Configuration.h"
#include "ExitHandler.h"
#include "HelperPrint.h"
#include "del/semantics/planning_task.h"
#include "epddl/grounder/grounder_helper.h"
#include "utilities/FormulaHelper.h"

Domain::Domain()
    : m_name(
          std::filesystem::path(
              ArgumentParser::get_instance().get_domain_file())
              .stem()
              .string()
          + "_" +
          std::filesystem::path(
              ArgumentParser::get_instance().get_problem_file())
              .stem()
              .string()) {

  const ArgumentParser &argument_parser =
      ArgumentParser::get_instance();

    auto library_files = argument_parser.get_library_files();

    const auto [specification_paths, failed] =
        plank::epddl::grounder::grounder_helper::
            get_specification_paths(
                argument_parser.get_domain_file(),
                argument_parser.get_problem_file(),
                library_files,
                "");

  if (failed) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::ParsingError,
        "Failed to load EPDDL specification paths.");
  }

  try {
    m_plank_task =
        plank::epddl::grounder::grounder_helper::
            build_ground_task(
                specification_paths,
                !argument_parser.get_verbose());
  } catch (const std::exception &e) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::ParsingError,
        e.what());
  }

    // Convert the grounded EPDDL task into DEEP's internal structures.
    build();

}

Domain &Domain::get_instance() {
  static Domain instance;
  return instance;
}

const FluentsSet &Domain::get_fluents() const noexcept { return m_fluents; }

const std::vector<Fluent> &Domain::get_positive_fluents() const noexcept {
  return m_positive_fluents;
}

unsigned int Domain::get_fluent_number() const noexcept {
  return static_cast<unsigned int>(m_fluents.size() / 2);
}

unsigned int Domain::get_size_fluent() const noexcept {
  auto fluent_first = m_fluents.begin();
  return fluent_first != m_fluents.end()
             ? static_cast<unsigned int>(fluent_first->size())
             : 0;
}

const ActionsSet &Domain::get_actions() const noexcept { return m_actions; }

const AgentsSet &Domain::get_agents() const noexcept { return m_agents; }

unsigned int Domain::get_agent_number() const noexcept {
  return static_cast<unsigned int>(m_agents.size());
}

const std::string &Domain::get_name() const noexcept { return m_name; }

const plank::del::state_ptr &
Domain::get_initial_state() const noexcept {
  return m_plank_task.initial_state;
}

const FormulaeList &Domain::get_goal_description() const noexcept {
  return m_goal_description;
}

void Domain::build() {
  auto &os = ArgumentParser::get_instance().get_output_stream();
  if (ArgumentParser::get_instance().get_verbose()) {
    os << "\n\n========== DOMAIN OUTPUT BEGIN ==========\n";
  }

  Grounder grounder;
  build_agents(grounder);
  build_fluents(grounder);
  build_actions(grounder);
  //build_goal();
  if (ArgumentParser::get_instance().get_verbose()) {
    os << "========== DOMAIN OUTPUT END ==========\n\n";
  }
}

void Domain::build_agents(Grounder &grounder) {
  auto &os = ArgumentParser::get_instance().get_output_stream();

  if (ArgumentParser::get_instance().get_verbose()) {
    os << "Building agent list..." << std::endl;
  }

  const auto language =
      m_plank_task.initial_state->get_language();

  const auto &agent_names =
      language->get_agents_names();

  const int agents_length =
      FormulaHelper::length_to_power_two(
          static_cast<int>(agent_names.size()));

  AgentsMap domain_agent_map;

  int i = 0;

  for (const auto &agent_name : agent_names) {
    Agent agent(agents_length, i);

    domain_agent_map.insert({agent_name, agent});
    m_agents.insert(agent);

#ifdef DEBUG
    if (ArgumentParser::get_instance().get_verbose()) {
      os << "Agent " << agent_name
         << " is " << agent << std::endl;
    }
#endif

    ++i;
  }

  grounder.set_agent_map(domain_agent_map);
}

void Domain::build_fluents(Grounder &grounder) {
  FluentMap domain_fluent_map;
  auto &os = ArgumentParser::get_instance().get_output_stream();

  if (ArgumentParser::get_instance().get_verbose()) {
    os << "Building fluent literals..." << std::endl;
  }

  const auto &fluent_names =
      m_plank_task.initial_state
          ->get_language()
          ->get_atoms_names();

  int i = 0;

  const int bit_size =
      FormulaHelper::length_to_power_two(
          static_cast<int>(fluent_names.size())) +
      1; // +1 for the negation bit

  for (const auto &fluent_name : fluent_names) {
    Fluent fluent_real(bit_size, i);
    fluent_real.set(fluent_real.size() - 1, true);

    domain_fluent_map.insert(
        {fluent_name, fluent_real});

    m_fluents.insert(fluent_real);
    m_positive_fluents.push_back(fluent_real);

    Fluent fluent_negate_real(bit_size, i);

    domain_fluent_map.insert(
        {NEGATION_SYMBOL + fluent_name,
         fluent_negate_real});

    m_fluents.insert(fluent_negate_real);

#ifdef DEBUG
    if (ArgumentParser::get_instance().get_verbose()) {
      os << "Literal " << fluent_name
         << " is " << fluent_real << std::endl;

      os << "Literal not " << fluent_name
         << " is " << fluent_negate_real
         << std::endl;
    }
#endif

    ++i;
  }

  grounder.set_fluent_map(domain_fluent_map);
}

void Domain::build_actions(Grounder &grounder) {
  ActionNamesMap domain_action_name_map;

  auto &os =
      ArgumentParser::get_instance().get_output_stream();

  if (ArgumentParser::get_instance().get_verbose()) {
    os << "Building action list from EPDDL..." << std::endl;
  }

  const auto &plank_actions = m_plank_task.actions;

  if (plank_actions.empty()) {
    ExitHandler::exit_with_message(
        ExitHandler::ExitCode::DomainBuildError,
        "EPDDL grounder produced no actions.");
  }

  const int bit_size =
      FormulaHelper::length_to_power_two(
          static_cast<int>(plank_actions.size()));

  int i = 0;

  for (const auto &plank_action : plank_actions) {
    const std::string action_name =
        plank_action->get_name();

    ActionId action_id(bit_size, i);

    Action action(action_name, action_id);

    domain_action_name_map.emplace(
        action_name,
        action_id);

    m_actions.insert(std::move(action));

    if (ArgumentParser::get_instance().get_verbose()) {
      os << "Action "
         << action_name
         << " is "
         << action_id
         << std::endl;
    }

    ++i;
  }

  grounder.set_action_name_map(
      domain_action_name_map);

  HelperPrint::get_instance().set_grounder(
      grounder);

  if (ArgumentParser::get_instance().get_verbose()) {
    os << "Built "
       << m_actions.size()
       << " grounded EPDDL actions."
       << std::endl;
  }
}
void Domain::build_goal() {
  // auto &os = ArgumentParser::get_instance().get_output_stream();
  // if (ArgumentParser::get_instance().get_verbose()) {
  //   os << "Adding to Goal..." << std::endl;
  // }
  //
  // ////\todo This will be replaced by epddl parser. Reader needs to be changed
  // /// and make sure to have getter and setter
  // for (auto &formula_parsed : domain_reader->m_bf_goal) {
  //   const auto formula = BeliefFormula(formula_parsed);
  //   m_goal_description.push_back(formula);
  //   if (ArgumentParser::get_instance().get_verbose()) {
  //     os << "    ";
  //     formula.print();
  //     os << std::endl;
  //   }
  // }
}
