#const maxT  = 3.
#const n_fl  = 8.
#const n_ag  = 3.
#const n_ags = 7.

%time(0..maxT).

% *** FLUENTS ***
fluent(has_key_a).
formula(has_key_a).
fluent(has_key_b).
formula(has_key_b).
fluent(has_key_c).
formula(has_key_c).
fluent(looking_a).
formula(looking_a).
fluent(looking_b).
formula(looking_b).
fluent(looking_c).
formula(looking_c).
fluent(opened).
formula(opened).
fluent(tail).
formula(tail).


% *** AGENTS ***
agent(a).
agent(b).
agent(c).


% *** ACTIONS ***
action(distract_a_b).
ontic(distract_a_b).
formula(-looking_b).
causes(distract_a_b, -looking_b, true).
formula(b(a,looking_a)).
formula(b(a,looking_b)).
formula(and(b(a,looking_a),b(a,looking_b))).
formula(and(looking_a,looking_b)).
formula(and(and(b(a,looking_a),b(a,looking_b)),and(looking_a,looking_b))).
executable(distract_a_b, and(and(b(a,looking_a),b(a,looking_b)),and(looking_a,looking_b))).
observes(distract_a_b, a, true).
observes(distract_a_b, b, true).
oblivious(distract_a_b, c).


action(distract_a_c).
ontic(distract_a_c).
formula(-looking_c).
causes(distract_a_c, -looking_c, true).
formula(b(a,looking_c)).
formula(and(b(a,looking_a),b(a,looking_c))).
formula(and(looking_a,looking_c)).
formula(and(and(b(a,looking_a),b(a,looking_c)),and(looking_a,looking_c))).
executable(distract_a_c, and(and(b(a,looking_a),b(a,looking_c)),and(looking_a,looking_c))).
observes(distract_a_c, a, true).
observes(distract_a_c, c, true).
oblivious(distract_a_c, b).


action(distract_b_a).
ontic(distract_b_a).
formula(-looking_a).
causes(distract_b_a, -looking_a, true).
formula(b(b,looking_b)).
formula(b(b,looking_a)).
formula(and(b(b,looking_b),b(b,looking_a))).
formula(and(and(b(b,looking_b),b(b,looking_a)),and(looking_a,looking_b))).
executable(distract_b_a, and(and(b(b,looking_b),b(b,looking_a)),and(looking_a,looking_b))).
observes(distract_b_a, a, true).
observes(distract_b_a, b, true).
oblivious(distract_b_a, c).


action(distract_b_c).
ontic(distract_b_c).
causes(distract_b_c, -looking_c, true).
formula(b(b,looking_c)).
formula(and(b(b,looking_b),b(b,looking_c))).
formula(and(looking_b,looking_c)).
formula(and(and(b(b,looking_b),b(b,looking_c)),and(looking_b,looking_c))).
executable(distract_b_c, and(and(b(b,looking_b),b(b,looking_c)),and(looking_b,looking_c))).
observes(distract_b_c, b, true).
observes(distract_b_c, c, true).
oblivious(distract_b_c, a).


action(distract_c_a).
ontic(distract_c_a).
causes(distract_c_a, -looking_a, true).
formula(b(c,looking_c)).
formula(b(c,looking_a)).
formula(and(b(c,looking_c),b(c,looking_a))).
formula(and(and(b(c,looking_c),b(c,looking_a)),and(looking_a,looking_c))).
executable(distract_c_a, and(and(b(c,looking_c),b(c,looking_a)),and(looking_a,looking_c))).
observes(distract_c_a, a, true).
observes(distract_c_a, c, true).
oblivious(distract_c_a, b).


action(distract_c_b).
ontic(distract_c_b).
causes(distract_c_b, -looking_b, true).
formula(b(c,looking_b)).
formula(and(b(c,looking_c),b(c,looking_b))).
formula(and(and(b(c,looking_c),b(c,looking_b)),and(looking_b,looking_c))).
executable(distract_c_b, and(and(b(c,looking_c),b(c,looking_b)),and(looking_b,looking_c))).
observes(distract_c_b, b, true).
observes(distract_c_b, c, true).
oblivious(distract_c_b, a).


action(open_a).
ontic(open_a).
causes(open_a, opened, true).
formula(b(a,has_key_a)).
formula(and(b(a,has_key_a),has_key_a)).
executable(open_a, and(b(a,has_key_a),has_key_a)).
observes(open_a, a, true).
observes(open_a, b, looking_b).
observes(open_a, c, looking_c).


action(open_b).
ontic(open_b).
causes(open_b, opened, true).
formula(b(b,has_key_b)).
formula(and(b(b,has_key_b),has_key_b)).
executable(open_b, and(b(b,has_key_b),has_key_b)).
observes(open_b, a, looking_a).
observes(open_b, b, true).
observes(open_b, c, looking_c).


action(open_c).
ontic(open_c).
causes(open_c, opened, true).
formula(b(c,has_key_c)).
formula(and(b(c,has_key_c),has_key_c)).
executable(open_c, and(b(c,has_key_c),has_key_c)).
observes(open_c, a, looking_a).
observes(open_c, b, looking_b).
observes(open_c, c, true).


action(peek_a).
sensing(peek_a).
determines(peek_a, tail, true).
formula(b(a,opened)).
formula(and(b(a,opened),b(a,looking_a))).
formula(and(looking_a,opened)).
formula(and(and(b(a,opened),b(a,looking_a)),and(looking_a,opened))).
executable(peek_a, and(and(b(a,opened),b(a,looking_a)),and(looking_a,opened))).
observes(peek_a, a, true).
aware_of(peek_a, b, looking_b).
aware_of(peek_a, c, looking_c).


action(peek_b).
sensing(peek_b).
determines(peek_b, tail, true).
formula(b(b,opened)).
formula(and(b(b,opened),b(b,looking_b))).
formula(and(looking_b,opened)).
formula(and(and(b(b,opened),b(b,looking_b)),and(looking_b,opened))).
executable(peek_b, and(and(b(b,opened),b(b,looking_b)),and(looking_b,opened))).
observes(peek_b, b, true).
aware_of(peek_b, a, looking_a).
aware_of(peek_b, c, looking_c).


action(peek_c).
sensing(peek_c).
determines(peek_c, tail, true).
formula(b(c,opened)).
formula(and(b(c,opened),b(c,looking_c))).
formula(and(looking_c,opened)).
formula(and(and(b(c,opened),b(c,looking_c)),and(looking_c,opened))).
executable(peek_c, and(and(b(c,opened),b(c,looking_c)),and(looking_c,opened))).
observes(peek_c, c, true).
aware_of(peek_c, a, looking_a).
aware_of(peek_c, b, looking_b).


action(shout_tail_a).
announcement(shout_tail_a).
announces(shout_tail_a, tail, true).
formula(b(a,tail)).
formula(and(b(a,tail),tail)).
executable(shout_tail_a, and(b(a,tail),tail)).
observes(shout_tail_a, a, true).
observes(shout_tail_a, b, looking_b).
observes(shout_tail_a, c, looking_c).


action(shout_tail_b).
announcement(shout_tail_b).
announces(shout_tail_b, tail, true).
formula(b(b,tail)).
formula(and(b(b,tail),tail)).
executable(shout_tail_b, and(b(b,tail),tail)).
observes(shout_tail_b, a, looking_a).
observes(shout_tail_b, b, true).
observes(shout_tail_b, c, looking_c).


action(shout_tail_c).
announcement(shout_tail_c).
announces(shout_tail_c, tail, true).
formula(b(c,tail)).
formula(and(b(c,tail),tail)).
executable(shout_tail_c, and(b(c,tail),tail)).
observes(shout_tail_c, a, looking_a).
observes(shout_tail_c, b, looking_b).
observes(shout_tail_c, c, true).


action(signal_a_b).
ontic(signal_a_b).
causes(signal_a_b, looking_b, true).
formula(b(a,-looking_b)).
formula(and(b(a,looking_a),b(a,-looking_b))).
formula(and(looking_a,-looking_b)).
formula(and(and(b(a,looking_a),b(a,-looking_b)),and(looking_a,-looking_b))).
executable(signal_a_b, and(and(b(a,looking_a),b(a,-looking_b)),and(looking_a,-looking_b))).
observes(signal_a_b, a, true).
observes(signal_a_b, b, true).
oblivious(signal_a_b, c).


action(signal_a_c).
ontic(signal_a_c).
causes(signal_a_c, looking_c, true).
formula(b(a,-looking_c)).
formula(and(b(a,looking_a),b(a,-looking_c))).
formula(and(looking_a,-looking_c)).
formula(and(and(b(a,looking_a),b(a,-looking_c)),and(looking_a,-looking_c))).
executable(signal_a_c, and(and(b(a,looking_a),b(a,-looking_c)),and(looking_a,-looking_c))).
observes(signal_a_c, a, true).
observes(signal_a_c, c, true).
oblivious(signal_a_c, b).


action(signal_b_a).
ontic(signal_b_a).
causes(signal_b_a, looking_a, true).
formula(b(b,-looking_a)).
formula(and(b(b,looking_b),b(b,-looking_a))).
formula(and(-looking_a,looking_b)).
formula(and(and(b(b,looking_b),b(b,-looking_a)),and(-looking_a,looking_b))).
executable(signal_b_a, and(and(b(b,looking_b),b(b,-looking_a)),and(-looking_a,looking_b))).
observes(signal_b_a, a, true).
observes(signal_b_a, b, true).
oblivious(signal_b_a, c).


action(signal_b_c).
ontic(signal_b_c).
causes(signal_b_c, looking_c, true).
formula(b(b,-looking_c)).
formula(and(b(b,looking_b),b(b,-looking_c))).
formula(and(looking_b,-looking_c)).
formula(and(and(b(b,looking_b),b(b,-looking_c)),and(looking_b,-looking_c))).
executable(signal_b_c, and(and(b(b,looking_b),b(b,-looking_c)),and(looking_b,-looking_c))).
observes(signal_b_c, b, true).
observes(signal_b_c, c, true).
oblivious(signal_b_c, a).


action(signal_c_a).
ontic(signal_c_a).
causes(signal_c_a, looking_a, true).
formula(b(c,-looking_a)).
formula(and(b(c,looking_c),b(c,-looking_a))).
formula(and(-looking_a,looking_c)).
formula(and(and(b(c,looking_c),b(c,-looking_a)),and(-looking_a,looking_c))).
executable(signal_c_a, and(and(b(c,looking_c),b(c,-looking_a)),and(-looking_a,looking_c))).
observes(signal_c_a, a, true).
observes(signal_c_a, c, true).
oblivious(signal_c_a, b).


action(signal_c_b).
ontic(signal_c_b).
causes(signal_c_b, looking_b, true).
formula(b(c,-looking_b)).
formula(and(b(c,looking_c),b(c,-looking_b))).
formula(and(-looking_b,looking_c)).
formula(and(and(b(c,looking_c),b(c,-looking_b)),and(-looking_b,looking_c))).
executable(signal_c_b, and(and(b(c,looking_c),b(c,-looking_b)),and(-looking_b,looking_c))).
observes(signal_c_b, b, true).
observes(signal_c_b, c, true).
oblivious(signal_c_b, a).




% *** INITIALLY ***

% ** initial pointed **
initially(has_key_a).
formula(-has_key_b).
initially(-has_key_b).
formula(-has_key_c).
initially(-has_key_c).
initially(looking_a).
initially(-looking_b).
initially(-looking_c).
formula(-opened).
initially(-opened).
initially(tail).

% ** initial beliefs **
agent_set(all_agents).
contains_ag(all_agents,a).
contains_ag(all_agents,b).
contains_ag(all_agents,c).
formula(c(all_agents,has_key_a)).
initially(c(all_agents,has_key_a)).
formula(c(all_agents,-has_key_b)).
initially(c(all_agents,-has_key_b)).
formula(c(all_agents,-has_key_c)).
initially(c(all_agents,-has_key_c)).
formula(c(all_agents,-opened)).
initially(c(all_agents,-opened)).
formula(c(all_agents,looking_a)).
initially(c(all_agents,looking_a)).
formula(c(all_agents,-looking_b)).
initially(c(all_agents,-looking_b)).
formula(c(all_agents,-looking_c)).
initially(c(all_agents,-looking_c)).



% ** commonly known fluent formulae **
init_sf(has_key_a).
init_sf(-has_key_b).
init_sf(-has_key_c).
init_sf(-opened).
init_sf(looking_a).
init_sf(-looking_b).
init_sf(-looking_c).


% *** GOALS ***
goal(b(b,tail)).


