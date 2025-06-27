#include "ActionLevel.h"
#include <ostream>

#include "ArgumentParser.h"

ActionLevel::ActionLevel(const ActionsSet &actions) {
  set_actions(actions);
  set_depth(0);
}

ActionLevel::ActionLevel(const ActionsSet &actions, unsigned short depth) {
  set_actions(actions);
  set_depth(depth);
}

void ActionLevel::set_actions(const ActionsSet &actions) {
  m_actions = actions;
}

void ActionLevel::add_action(const Action &act) {
  // Insert only if not already present (std::set handles this)
  m_actions.insert(act);
}

void ActionLevel::set_depth(const unsigned short depth) noexcept {
  m_depth = depth;
}

unsigned short ActionLevel::get_depth() const noexcept { return m_depth; }

const ActionsSet &ActionLevel::get_actions() const noexcept {
  return m_actions;
}

ActionLevel &ActionLevel::operator=(const ActionLevel &to_assign) {
  set_actions(to_assign.get_actions());
  set_depth(to_assign.get_depth());
  return (*this);
}

void ActionLevel::print() const {
  /**
   * \brief Prints this ActionLevel to the given output stream.
   * \param os The output stream to print to.
   */
  auto &os = ArgumentParser::get_instance().get_output_stream();

  os << "\nPrinting an Action Level:";
  if (m_actions.empty()) {
    os << "\nActionLevel is empty.\n";
    return;
  }
  for (const auto &act : m_actions) {
    os << "\nAction " << act.get_name() << std::endl;
  }
}
