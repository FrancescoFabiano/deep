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
#include "PlankFormulaConverter.h"
#include "del/semantics/planning_task.h"
#include "epddl/grounder/grounder_helper.h"
#include "utilities/FormulaHelper.h"

Domain::Domain()
    : m_name(std::filesystem::path(
                 ArgumentParser::get_instance().get_domain_file())
                 .stem()
                 .string() +
             "_" +
             std::filesystem::path(
                 ArgumentParser::get_instance().get_problem_file())
                 .stem()
                 .string()) {

  const ArgumentParser &argument_parser = ArgumentParser::get_instance();

  auto library_files = argument_parser.get_library_files();

  const auto [specification_paths, failed] =
      plank::epddl::grounder::grounder_helper::get_specification_paths(
          argument_parser.get_domain_file(), argument_parser.get_problem_file(),
          library_files, "");

  if (failed) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::ParsingError,
                                   "Failed to load EPDDL specification paths.");
  }

  try {
    m_plank_task = plank::epddl::grounder::grounder_helper::build_ground_task(
        specification_paths, !argument_parser.get_verbose());
  } catch (const std::exception &e) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::ParsingError,
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

const plank::del::state_ptr &Domain::get_initial_state() const noexcept {
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
  build_goal();
  if (ArgumentParser::get_instance().get_verbose()) {
    os << "========== DOMAIN OUTPUT END ==========\n\n";
  }
}

void Domain::build_agents(Grounder &grounder) {
  auto &os = ArgumentParser::get_instance().get_output_stream();

  if (ArgumentParser::get_instance().get_verbose()) {
    os << "Building agent list..." << std::endl;
  }

  const auto language = m_plank_task.initial_state->get_language();

  const auto &agent_names = language->get_agents_names();

  const int agents_length =
      FormulaHelper::length_to_power_two(static_cast<int>(agent_names.size()));

  AgentsMap domain_agent_map;

  int i = 0;

  for (const auto &agent_name : agent_names) {
    Agent agent(agents_length, i);

    domain_agent_map.insert({agent_name, agent});
    m_agents.insert(agent);
    m_ordered_agents.push_back(agent);

#ifdef DEBUG
    if (ArgumentParser::get_instance().get_verbose()) {
      os << "Agent " << agent_name << " is " << agent << std::endl;
    }
#endif

    ++i;
  }

  grounder.set_agent_map(domain_agent_map);
}

const std::vector<Agent> &Domain::get_ordered_agents() const noexcept {
  return m_ordered_agents;
}

void Domain::build_fluents(Grounder &grounder) {
  FluentMap domain_fluent_map;
  auto &os = ArgumentParser::get_instance().get_output_stream();

  if (ArgumentParser::get_instance().get_verbose()) {
    os << "Building fluent literals..." << std::endl;
  }

  const auto &fluent_names =
      m_plank_task.initial_state->get_language()->get_atoms_names();

  int i = 0;

  const int bit_size = FormulaHelper::length_to_power_two(
                           static_cast<int>(fluent_names.size())) +
                       1; // +1 for the negation bit

  for (const auto &fluent_name : fluent_names) {
    Fluent fluent_real(bit_size, i);
    fluent_real.set(fluent_real.size() - 1, true);

    domain_fluent_map.insert({fluent_name, fluent_real});

    m_fluents.insert(fluent_real);
    m_positive_fluents.push_back(fluent_real);

    Fluent fluent_negate_real(bit_size, i);

    domain_fluent_map.insert(
        {NEGATION_SYMBOL + fluent_name, fluent_negate_real});

    m_fluents.insert(fluent_negate_real);

#ifdef DEBUG
    if (ArgumentParser::get_instance().get_verbose()) {
      os << "Literal " << fluent_name << " is " << fluent_real << std::endl;

      os << "Literal not " << fluent_name << " is " << fluent_negate_real
         << std::endl;
    }
#endif

    ++i;
  }

  grounder.set_fluent_map(domain_fluent_map);
}

void Domain::build_actions(Grounder &grounder) {
  const PlankFormulaConverter converter(m_positive_fluents, m_ordered_agents);

  ActionNamesMap domain_action_name_map;

  auto &os = ArgumentParser::get_instance().get_output_stream();

  if (ArgumentParser::get_instance().get_verbose()) {
    os << "Building action list from EPDDL..." << std::endl;
  }

  const auto &plank_actions = m_plank_task.actions;

  if (plank_actions.empty()) {
    ExitHandler::exit_with_message(ExitHandler::ExitCode::DomainBuildError,
                                   "EPDDL grounder produced no actions.");
  }

  const int bit_size = FormulaHelper::length_to_power_two(
      static_cast<int>(plank_actions.size()));

  int action_index = 0;

  for (const auto &plank_action : plank_actions) {
    const std::string action_name = plank_action->get_name();

    ActionId action_id(bit_size, action_index);

    Action action(action_name, action_id);

    /*
     * ============================================================
     * Events
     * ============================================================
     */
    for (const plank::del::event_id event_id : plank_action->get_events()) {

      Event event(static_cast<EventId>(event_id),
                  plank_action->get_event_name(event_id));

      /*
       * Preconditions.
       */
      event.set_precondition(
          converter.convert(plank_action->get_precondition(event_id)));

      /*
       * Postconditions.
       *
       * plank:
       *   atom_id -> formula
       *
       * DEEP:
       *   Fluent -> BeliefFormula
       */
      const auto &postconditions = plank_action->get_postconditions(event_id);

      for (const auto &[atom_id, postcondition] : postconditions) {

        if (atom_id >= m_positive_fluents.size()) {
          ExitHandler::exit_with_message(
              ExitHandler::ExitCode::DomainBuildError,
              "Invalid atom id in EPDDL action postcondition.");
        }

        event.add_postcondition(m_positive_fluents[atom_id],
                                converter.convert(postcondition));
      }

      action.add_event(event);

      /*
       * Designated events.
       */
      if (plank_action->is_designated(event_id)) {
        action.add_designated_event(static_cast<EventId>(event_id));
      }
    }

    /*
     * ============================================================
     * Observability-type relations
     * ============================================================
     *
     * plank:
     *
     *   obs_type -> relation over events
     *
     * We preserve this directly instead of resolving it to
     * agent relations now. Resolution depends on the current state.
     */
    for (plank::del::obs_type obs_type = 0;
         obs_type < plank_action->get_obs_types_number(); ++obs_type) {

      for (const plank::del::event_id from : plank_action->get_events()) {

        const auto &possible_events =
            plank_action->get_obs_type_possible_events(obs_type, from);

        for (const plank::del::event_id to : possible_events) {

          action.add_observability_edge(
              static_cast<ObservabilityType>(obs_type),
              static_cast<EventId>(from), static_cast<EventId>(to));
        }
      }
    }

    /*
     * ============================================================
     * Agent observability conditions
     * ============================================================
     *
     * plank:
     *
     *   agent -> { obs_type -> formula }
     *
     * DEEP:
     *
     *   Agent -> { ObservabilityType -> BeliefFormula }
     *
     * m_ordered_agents preserves plank's agent-id ordering.
     */
    for (std::size_t agent_id = 0; agent_id < m_ordered_agents.size();
         ++agent_id) {

      const auto plank_agent = static_cast<plank::del::agent>(agent_id);

      const auto &conditions =
          plank_action->get_agent_obs_conditions(plank_agent);

      for (const auto &[obs_type, condition] : conditions) {

        action.add_observability_condition(
            m_ordered_agents[agent_id],
            static_cast<ObservabilityType>(obs_type),
            converter.convert(condition));
      }
    }

    /*
     * ============================================================
     * Store action
     * ============================================================
     */

    domain_action_name_map.emplace(action_name, action_id);

#ifdef DEBUG
    if (ArgumentParser::get_instance().get_verbose()) {
      os << "Action " << action_name << " is " << action_id << std::endl;

      os << "  Events: " << action.get_events().size()
         << ", designated: " << action.get_designated_events().size()
         << ", observability types: "
         << action.get_observability_relations().size()
         << ", agents with observability conditions: "
         << action.get_observability_conditions().size() << std::endl;
    }
#endif

    m_actions.insert(std::move(action));
    ++action_index;
  }

  grounder.set_action_name_map(domain_action_name_map);

  HelperPrint::get_instance().set_grounder(grounder);

  if (ArgumentParser::get_instance().get_verbose()) {
    os << "Built " << m_actions.size() << " grounded EPDDL actions."
       << std::endl;
  }
}

void Domain::build_goal() {
  auto &os = ArgumentParser::get_instance().get_output_stream();

  if (ArgumentParser::get_instance().get_verbose()) {
    os << "Building goal from EPDDL..." << std::endl;
  }

  m_goal_description.clear();

  const PlankFormulaConverter converter(m_positive_fluents, m_ordered_agents);

  m_goal_description.push_back(converter.convert(m_plank_task.goal));

  if (ArgumentParser::get_instance().get_verbose()) {
    os << "Goal: ";
    m_goal_description.back().print();
    os << std::endl;
  }
}