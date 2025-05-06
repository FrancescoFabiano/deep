#const maxT  = 3.
#const n_fl  = 13.
#const n_ag  = 3.
#const n_ags = 7.

%time(0..maxT).

% *** FLUENTS ***
fluent(at_a_1).
formula(at_a_1).
fluent(at_a_2).
formula(at_a_2).
fluent(at_a_3).
formula(at_a_3).
fluent(at_b1_1).
formula(at_b1_1).
fluent(at_b1_3).
formula(at_b1_3).
fluent(at_b2_1).
formula(at_b2_1).
fluent(at_b2_3).
formula(at_b2_3).
fluent(at_b_1).
formula(at_b_1).
fluent(at_b_2).
formula(at_b_2).
fluent(at_b_3).
formula(at_b_3).
fluent(at_c_1).
formula(at_c_1).
fluent(at_c_2).
formula(at_c_2).
fluent(at_c_3).
formula(at_c_3).


% *** AGENTS ***
agent(a).
agent(b).
agent(c).


% *** ACTIONS ***
action(a_check_1).
sensing(a_check_1).
determines(a_check_1, at_b1_1, true).
formula(b(a,at_a_1)).
executable(a_check_1, b(a,at_a_1)).
observes(a_check_1, a, true).
oblivious(a_check_1, b).
oblivious(a_check_1, c).


action(a_check_3).
sensing(a_check_3).
determines(a_check_3, at_b2_3, true).
formula(b(a,at_a_3)).
executable(a_check_3, b(a,at_a_3)).
observes(a_check_3, a, true).
oblivious(a_check_3, b).
oblivious(a_check_3, c).


action(b_check_1).
sensing(b_check_1).
determines(b_check_1, at_b1_1, true).
formula(b(b,at_b_1)).
executable(b_check_1, b(b,at_b_1)).
observes(b_check_1, b, true).
oblivious(b_check_1, a).
oblivious(b_check_1, c).


action(b_check_3).
sensing(b_check_3).
determines(b_check_3, at_b2_3, true).
formula(b(b,at_b_3)).
executable(b_check_3, b(b,at_b_3)).
observes(b_check_3, b, true).
oblivious(b_check_3, a).
oblivious(b_check_3, c).


action(c_check_1).
sensing(c_check_1).
determines(c_check_1, at_b1_1, true).
formula(b(c,at_c_1)).
executable(c_check_1, b(c,at_c_1)).
observes(c_check_1, c, true).
oblivious(c_check_1, a).
oblivious(c_check_1, b).


action(c_check_3).
sensing(c_check_3).
determines(c_check_3, at_b2_3, true).
formula(b(c,at_c_3)).
executable(c_check_3, b(c,at_c_3)).
observes(c_check_3, c, true).
oblivious(c_check_3, a).
oblivious(c_check_3, b).


action(left_a).
ontic(left_a).
ce_formula(at_a_2).
causes(left_a, at_a_1, at_a_2).
ce_formula(at_a_3).
causes(left_a, at_a_2, at_a_3).
formula(-at_a_2).
causes(left_a, -at_a_2, at_a_2).
formula(-at_a_3).
causes(left_a, -at_a_3, at_a_3).
formula(-at_a_1).
formula(b(a,-at_a_1)).
executable(left_a, b(a,-at_a_1)).
observes(left_a, a, true).
observes(left_a, b, true).
observes(left_a, c, true).


action(left_b).
ontic(left_b).
ce_formula(at_b_2).
causes(left_b, at_b_1, at_b_2).
ce_formula(at_b_3).
causes(left_b, at_b_2, at_b_3).
formula(-at_b_2).
causes(left_b, -at_b_2, at_b_2).
formula(-at_b_3).
causes(left_b, -at_b_3, at_b_3).
formula(-at_b_1).
formula(b(b,-at_b_1)).
executable(left_b, b(b,-at_b_1)).
observes(left_b, a, true).
observes(left_b, b, true).
observes(left_b, c, true).


action(left_c).
ontic(left_c).
ce_formula(at_c_2).
causes(left_c, at_c_1, at_c_2).
ce_formula(at_c_3).
causes(left_c, at_c_2, at_c_3).
formula(-at_c_2).
causes(left_c, -at_c_2, at_c_2).
formula(-at_c_3).
causes(left_c, -at_c_3, at_c_3).
formula(-at_c_1).
formula(b(c,-at_c_1)).
executable(left_c, b(c,-at_c_1)).
observes(left_c, a, true).
observes(left_c, b, true).
observes(left_c, c, true).


action(right_a).
ontic(right_a).
ce_formula(at_a_1).
causes(right_a, -at_a_1, at_a_1).
causes(right_a, at_a_2, at_a_1).
causes(right_a, -at_a_2, at_a_2).
causes(right_a, at_a_3, at_a_2).
formula(b(a,-at_a_3)).
executable(right_a, b(a,-at_a_3)).
observes(right_a, a, true).
observes(right_a, b, true).
observes(right_a, c, true).


action(right_b).
ontic(right_b).
ce_formula(at_b_1).
causes(right_b, -at_b_1, at_b_1).
causes(right_b, at_b_2, at_b_1).
causes(right_b, -at_b_2, at_b_2).
causes(right_b, at_b_3, at_b_2).
formula(b(b,-at_b_3)).
executable(right_b, b(b,-at_b_3)).
observes(right_b, a, true).
observes(right_b, b, true).
observes(right_b, c, true).


action(right_c).
ontic(right_c).
ce_formula(at_c_1).
causes(right_c, -at_c_1, at_c_1).
causes(right_c, at_c_2, at_c_1).
causes(right_c, -at_c_2, at_c_2).
causes(right_c, at_c_3, at_c_2).
formula(b(c,-at_c_3)).
executable(right_c, b(c,-at_c_3)).
observes(right_c, a, true).
observes(right_c, b, true).
observes(right_c, c, true).


action(tell_a_b1_1).
announcement(tell_a_b1_1).
announces(tell_a_b1_1, at_b1_1, true).
formula(b(a,at_b1_1)).
executable(tell_a_b1_1, b(a,at_b1_1)).
observes(tell_a_b1_1, a, true).
observes(tell_a_b1_1, b, true).
observes(tell_a_b1_1, c, true).


action(tell_a_b1_3).
announcement(tell_a_b1_3).
announces(tell_a_b1_3, at_b1_3, true).
formula(b(a,at_b1_3)).
executable(tell_a_b1_3, b(a,at_b1_3)).
observes(tell_a_b1_3, a, true).
observes(tell_a_b1_3, b, true).
observes(tell_a_b1_3, c, true).


action(tell_a_b2_1).
announcement(tell_a_b2_1).
announces(tell_a_b2_1, at_b2_1, true).
formula(b(a,at_b2_1)).
executable(tell_a_b2_1, b(a,at_b2_1)).
observes(tell_a_b2_1, a, true).
observes(tell_a_b2_1, b, true).
observes(tell_a_b2_1, c, true).


action(tell_a_b2_3).
announcement(tell_a_b2_3).
announces(tell_a_b2_3, at_b2_3, true).
formula(b(a,at_b2_3)).
executable(tell_a_b2_3, b(a,at_b2_3)).
observes(tell_a_b2_3, a, true).
observes(tell_a_b2_3, b, true).
observes(tell_a_b2_3, c, true).


action(tell_b_b1_1).
announcement(tell_b_b1_1).
announces(tell_b_b1_1, at_b1_1, true).
formula(b(b,at_b1_1)).
executable(tell_b_b1_1, b(b,at_b1_1)).
observes(tell_b_b1_1, a, true).
observes(tell_b_b1_1, b, true).
observes(tell_b_b1_1, c, true).


action(tell_b_b1_3).
announcement(tell_b_b1_3).
announces(tell_b_b1_3, at_b1_3, true).
formula(b(b,at_b1_3)).
executable(tell_b_b1_3, b(b,at_b1_3)).
observes(tell_b_b1_3, a, true).
observes(tell_b_b1_3, b, true).
observes(tell_b_b1_3, c, true).


action(tell_b_b2_1).
announcement(tell_b_b2_1).
announces(tell_b_b2_1, at_b2_1, true).
formula(b(b,at_b2_1)).
executable(tell_b_b2_1, b(b,at_b2_1)).
observes(tell_b_b2_1, a, true).
observes(tell_b_b2_1, b, true).
observes(tell_b_b2_1, c, true).


action(tell_b_b2_3).
announcement(tell_b_b2_3).
announces(tell_b_b2_3, at_b2_3, true).
formula(b(b,at_b2_3)).
executable(tell_b_b2_3, b(b,at_b2_3)).
observes(tell_b_b2_3, a, true).
observes(tell_b_b2_3, b, true).
observes(tell_b_b2_3, c, true).


action(tell_c_b1_1).
announcement(tell_c_b1_1).
announces(tell_c_b1_1, at_b1_1, true).
formula(b(c,at_b1_1)).
executable(tell_c_b1_1, b(c,at_b1_1)).
observes(tell_c_b1_1, a, true).
observes(tell_c_b1_1, b, true).
observes(tell_c_b1_1, c, true).


action(tell_c_b1_3).
announcement(tell_c_b1_3).
announces(tell_c_b1_3, at_b1_3, true).
formula(b(c,at_b1_3)).
executable(tell_c_b1_3, b(c,at_b1_3)).
observes(tell_c_b1_3, a, true).
observes(tell_c_b1_3, b, true).
observes(tell_c_b1_3, c, true).


action(tell_c_b2_1).
announcement(tell_c_b2_1).
announces(tell_c_b2_1, at_b2_1, true).
formula(b(c,at_b2_1)).
executable(tell_c_b2_1, b(c,at_b2_1)).
observes(tell_c_b2_1, a, true).
observes(tell_c_b2_1, b, true).
observes(tell_c_b2_1, c, true).


action(tell_c_b2_3).
announcement(tell_c_b2_3).
announces(tell_c_b2_3, at_b2_3, true).
formula(b(c,at_b2_3)).
executable(tell_c_b2_3, b(c,at_b2_3)).
observes(tell_c_b2_3, a, true).
observes(tell_c_b2_3, b, true).
observes(tell_c_b2_3, c, true).




% *** INITIALLY ***

% ** initial pointed **
initially(-at_a_1).
initially(at_a_2).
initially(-at_a_3).
initially(at_b1_1).
formula(-at_b1_3).
initially(-at_b1_3).
formula(-at_b2_1).
initially(-at_b2_1).
initially(at_b2_3).
initially(-at_b_1).
initially(at_b_2).
initially(-at_b_3).
initially(-at_c_1).
initially(at_c_2).
initially(-at_c_3).

% ** initial beliefs **
agent_set(all_agents).
contains_ag(all_agents,a).
contains_ag(all_agents,b).
contains_ag(all_agents,c).
formula(c(all_agents,at_a_2)).
initially(c(all_agents,at_a_2)).
formula(c(all_agents,at_b_2)).
initially(c(all_agents,at_b_2)).
formula(c(all_agents,at_c_2)).
initially(c(all_agents,at_c_2)).
formula(c(all_agents,-at_a_1)).
initially(c(all_agents,-at_a_1)).
formula(c(all_agents,-at_b_1)).
initially(c(all_agents,-at_b_1)).
formula(c(all_agents,-at_c_1)).
initially(c(all_agents,-at_c_1)).
formula(c(all_agents,-at_a_3)).
initially(c(all_agents,-at_a_3)).
formula(c(all_agents,-at_b_3)).
initially(c(all_agents,-at_b_3)).
formula(c(all_agents,-at_c_3)).
initially(c(all_agents,-at_c_3)).
formula(and(at_b1_1,-at_b1_3)).
formula(-at_b1_1).
formula(and(-at_b1_1,at_b1_3)).
formula(or(and(at_b1_1,-at_b1_3),and(-at_b1_1,at_b1_3))).
formula(c(all_agents,or(and(at_b1_1,-at_b1_3),and(-at_b1_1,at_b1_3)))).
initially(c(all_agents,or(and(at_b1_1,-at_b1_3),and(-at_b1_1,at_b1_3)))).
formula(-at_b2_3).
formula(and(at_b2_1,-at_b2_3)).
formula(and(-at_b2_1,at_b2_3)).
formula(or(and(at_b2_1,-at_b2_3),and(-at_b2_1,at_b2_3))).
formula(c(all_agents,or(and(at_b2_1,-at_b2_3),and(-at_b2_1,at_b2_3)))).
initially(c(all_agents,or(and(at_b2_1,-at_b2_3),and(-at_b2_1,at_b2_3)))).



% ** commonly known fluent formulae **
init_sf(at_a_2).
init_sf(at_b_2).
init_sf(at_c_2).
init_sf(-at_a_1).
init_sf(-at_b_1).
init_sf(-at_c_1).
init_sf(-at_a_3).
init_sf(-at_b_3).
init_sf(-at_c_3).
init_sf(at_b1_1).
init_sf(-at_b1_3).
init_sf(and(at_b1_1,-at_b1_3)).
init_sf(-at_b1_1).
init_sf(at_b1_3).
init_sf(and(-at_b1_1,at_b1_3)).
init_sf(or(and(at_b1_1,-at_b1_3),and(-at_b1_1,at_b1_3))).
init_sf(at_b2_1).
init_sf(-at_b2_3).
init_sf(and(at_b2_1,-at_b2_3)).
init_sf(-at_b2_1).
init_sf(at_b2_3).
init_sf(and(-at_b2_1,at_b2_3)).
init_sf(or(and(at_b2_1,-at_b2_3),and(-at_b2_1,at_b2_3))).


% *** GOALS ***
agent_set(ab).
contains_ag(ab,a).
contains_ag(ab,b).
formula(c(ab,at_b1_1)).
goal(c(ab,at_b1_1)).


