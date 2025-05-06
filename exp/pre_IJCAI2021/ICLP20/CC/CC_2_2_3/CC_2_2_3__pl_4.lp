#const maxT  = 4.
#const n_fl  = 10.
#const n_ag  = 2.
#const n_ags = 3.

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


% *** AGENTS ***
agent(a).
agent(b).


% *** ACTIONS ***
action(a_check_1).
sensing(a_check_1).
determines(a_check_1, at_b1_1, true).
formula(b(a,at_a_1)).
executable(a_check_1, b(a,at_a_1)).
observes(a_check_1, a, true).
oblivious(a_check_1, b).


action(a_check_3).
sensing(a_check_3).
determines(a_check_3, at_b2_3, true).
formula(b(a,at_a_3)).
executable(a_check_3, b(a,at_a_3)).
observes(a_check_3, a, true).
oblivious(a_check_3, b).


action(b_check_1).
sensing(b_check_1).
determines(b_check_1, at_b1_1, true).
formula(b(b,at_b_1)).
executable(b_check_1, b(b,at_b_1)).
observes(b_check_1, b, true).
oblivious(b_check_1, a).


action(b_check_3).
sensing(b_check_3).
determines(b_check_3, at_b2_3, true).
formula(b(b,at_b_3)).
executable(b_check_3, b(b,at_b_3)).
observes(b_check_3, b, true).
oblivious(b_check_3, a).


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
formula(neg(at_a_1)).
formula(and(neg(at_a_1),b(a,-at_a_1))).
executable(left_a, and(neg(at_a_1),b(a,-at_a_1))).
observes(left_a, a, true).
observes(left_a, b, true).


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
formula(neg(at_b_1)).
formula(-at_b_1).
formula(b(b,-at_b_1)).
formula(and(b(b,-at_b_1),neg(at_b_1))).
executable(left_b, and(b(b,-at_b_1),neg(at_b_1))).
observes(left_b, a, true).
observes(left_b, b, true).


action(right_a).
ontic(right_a).
ce_formula(at_a_1).
causes(right_a, -at_a_1, at_a_1).
causes(right_a, at_a_2, at_a_1).
causes(right_a, -at_a_2, at_a_2).
causes(right_a, at_a_3, at_a_2).
formula(neg(at_a_3)).
formula(b(a,-at_a_3)).
formula(and(b(a,-at_a_3),neg(at_a_3))).
executable(right_a, and(b(a,-at_a_3),neg(at_a_3))).
observes(right_a, a, true).
observes(right_a, b, true).


action(right_b).
ontic(right_b).
ce_formula(at_b_1).
causes(right_b, -at_b_1, at_b_1).
causes(right_b, at_b_2, at_b_1).
causes(right_b, -at_b_2, at_b_2).
causes(right_b, at_b_3, at_b_2).
formula(neg(at_b_3)).
formula(b(b,-at_b_3)).
formula(and(b(b,-at_b_3),neg(at_b_3))).
executable(right_b, and(b(b,-at_b_3),neg(at_b_3))).
observes(right_b, a, true).
observes(right_b, b, true).


action(tell_a_b1_1).
announcement(tell_a_b1_1).
announces(tell_a_b1_1, at_b1_1, true).
formula(b(a,at_b1_1)).
executable(tell_a_b1_1, b(a,at_b1_1)).
observes(tell_a_b1_1, b, true).
oblivious(tell_a_b1_1, a).


action(tell_a_b1_3).
announcement(tell_a_b1_3).
announces(tell_a_b1_3, at_b1_3, true).
formula(b(a,at_b1_3)).
executable(tell_a_b1_3, b(a,at_b1_3)).
observes(tell_a_b1_3, a, true).
observes(tell_a_b1_3, b, true).


action(tell_a_b2_1).
announcement(tell_a_b2_1).
announces(tell_a_b2_1, at_b2_1, true).
formula(b(a,at_b2_1)).
executable(tell_a_b2_1, b(a,at_b2_1)).
observes(tell_a_b2_1, a, true).
observes(tell_a_b2_1, b, true).


action(tell_a_b2_3).
announcement(tell_a_b2_3).
announces(tell_a_b2_3, at_b2_3, true).
formula(b(a,at_b2_3)).
executable(tell_a_b2_3, b(a,at_b2_3)).
observes(tell_a_b2_3, a, true).
observes(tell_a_b2_3, b, true).


action(tell_b_b1_1).
announcement(tell_b_b1_1).
announces(tell_b_b1_1, at_b1_1, true).
formula(b(b,at_b1_1)).
executable(tell_b_b1_1, b(b,at_b1_1)).
observes(tell_b_b1_1, a, true).
observes(tell_b_b1_1, b, true).


action(tell_b_b1_3).
announcement(tell_b_b1_3).
announces(tell_b_b1_3, at_b1_3, true).
formula(b(b,at_b1_3)).
executable(tell_b_b1_3, b(b,at_b1_3)).
observes(tell_b_b1_3, a, true).
observes(tell_b_b1_3, b, true).


action(tell_b_b2_1).
announcement(tell_b_b2_1).
announces(tell_b_b2_1, at_b2_1, true).
formula(b(b,at_b2_1)).
executable(tell_b_b2_1, b(b,at_b2_1)).
observes(tell_b_b2_1, a, true).
observes(tell_b_b2_1, b, true).


action(tell_b_b2_3).
announcement(tell_b_b2_3).
announces(tell_b_b2_3, at_b2_3, true).
formula(b(b,at_b2_3)).
executable(tell_b_b2_3, b(b,at_b2_3)).
observes(tell_b_b2_3, a, true).
observes(tell_b_b2_3, b, true).




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

% ** initial beliefs **
agent_set(all_agents).
contains_ag(all_agents,a).
contains_ag(all_agents,b).
formula(c(all_agents,at_a_2)).
initially(c(all_agents,at_a_2)).
formula(c(all_agents,at_b_2)).
initially(c(all_agents,at_b_2)).
formula(c(all_agents,-at_a_1)).
initially(c(all_agents,-at_a_1)).
formula(c(all_agents,-at_b_1)).
initially(c(all_agents,-at_b_1)).
formula(c(all_agents,-at_a_3)).
initially(c(all_agents,-at_a_3)).
formula(c(all_agents,-at_b_3)).
initially(c(all_agents,-at_b_3)).
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
init_sf(-at_a_1).
init_sf(-at_b_1).
init_sf(-at_a_3).
init_sf(-at_b_3).
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
goal(b(b,at_b2_3)).
goal(b(a,at_b1_1)).


