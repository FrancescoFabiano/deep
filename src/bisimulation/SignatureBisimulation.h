/**
 * \file SignatureBisimulation.h
 * \brief Bisimulation contraction of a Kripke state by signature refinement.
 *
 * \details Worlds start partitioned by (valuation, designated); each round
 * splits every block by the set of (agent, successor block) pairs of its
 * worlds, until no block splits. The result is the coarsest bisimulation
 * that keeps valuations and designatedness, and the state is replaced by its
 * quotient (one representative world per block). It works on the Kripke
 * structure directly, without the automaton conversion of the FB/PT
 * implementations in Bisimulation.
 *
 * \copyright GNU Public License.
 * \date October 2, 2026
 */

#pragma once

#include "states/representations/kripke/KripkeState.h"

namespace SignatureBisimulation {
/**
 * \brief Replace \p kstate by its bisimulation contraction, in place.
 *
 * \details A state that is already minimal is left untouched.
 */
void contract(KripkeState &kstate);
} // namespace SignatureBisimulation
