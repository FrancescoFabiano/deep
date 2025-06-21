#include "SatisfiedGoals.h"

void SatisfiedGoals::set(const FormulaeList &goals) {
  set_goals(goals);
  set_goals_number(goals.size());
}

SatisfiedGoals &SatisfiedGoals::get_instance() {
  static SatisfiedGoals instance;
  return instance;
}

const FormulaeList &SatisfiedGoals::get_goals() const noexcept {
  return m_goals;
}

void SatisfiedGoals::set_goals(const FormulaeList &to_set) { m_goals = to_set; }

unsigned short SatisfiedGoals::get_goals_number() const noexcept {
  return m_goals_number;
}

void SatisfiedGoals::set_goals_number(const unsigned short to_set) {
  m_goals_number = to_set;
}
