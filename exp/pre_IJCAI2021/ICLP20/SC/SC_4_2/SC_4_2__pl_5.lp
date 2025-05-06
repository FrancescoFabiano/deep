#const maxT  = 5.
#const n_fl  = 5.
#const n_ag  = 7.
#const n_ags = 127.

%time(0..maxT).

% *** FLUENTS ***
fluent(at_1).
formula(at_1).
fluent(at_2).
formula(at_2).
fluent(at_3).
formula(at_3).
fluent(at_4).
formula(at_4).
fluent(q).
formula(q).


% *** AGENTS ***
agent(a).
agent(b).
agent(c).
agent(d).
agent(e).
agent(f).
agent(g).


% *** ACTIONS ***
action(left).
ontic(left).
ce_formula(at_2).
causes(left, at_1, at_2).
ce_formula(at_3).
causes(left, at_2, at_3).
formula(-at_2).
causes(left, -at_2, at_2).
ce_formula(at_4).
causes(left, at_3, at_4).
formula(-at_3).
causes(left, -at_3, at_3).
formula(-at_4).
causes(left, -at_4, at_4).
formula(-at_1).
executable(left, -at_1).
observes(left, a, true).
observes(left, b, true).
observes(left, c, true).
oblivious(left, d).
oblivious(left, e).
oblivious(left, f).
oblivious(left, g).


action(right).
ontic(right).
ce_formula(at_1).
causes(right, -at_1, at_1).
causes(right, at_2, at_1).
causes(right, -at_2, at_2).
causes(right, at_3, at_2).
causes(right, -at_3, at_3).
causes(right, at_4, at_3).
executable(right, -at_4).
observes(right, a, true).
observes(right, b, true).
observes(right, c, true).
oblivious(right, d).
oblivious(right, e).
oblivious(right, f).
oblivious(right, g).


action(sense).
sensing(sense).
determines(sense, q, true).
executable(sense, at_2).
observes(sense, a, true).
oblivious(sense, b).
oblivious(sense, c).
oblivious(sense, d).
oblivious(sense, e).
oblivious(sense, f).
oblivious(sense, g).


action(shout_1).
announcement(shout_1).
announces(shout_1, q, true).
formula(b(a,q)).
formula(and(b(a,q),at_1)).
executable(shout_1, and(b(a,q),at_1)).
observes(shout_1, a, true).
observes(shout_1, b, true).
observes(shout_1, d, true).
observes(shout_1, e, true).
observes(shout_1, f, true).
observes(shout_1, g, true).
oblivious(shout_1, c).


action(shout_2).
announcement(shout_2).
announces(shout_2, q, true).
formula(and(b(a,q),at_2)).
executable(shout_2, and(b(a,q),at_2)).
observes(shout_2, a, true).
observes(shout_2, b, true).
observes(shout_2, c, true).
observes(shout_2, d, true).
observes(shout_2, e, true).
observes(shout_2, f, true).
observes(shout_2, g, true).


action(shout_3).
announcement(shout_3).
announces(shout_3, q, true).
formula(and(b(a,q),at_3)).
executable(shout_3, and(b(a,q),at_3)).
observes(shout_3, a, true).
observes(shout_3, b, true).
observes(shout_3, c, true).
observes(shout_3, d, true).
observes(shout_3, e, true).
observes(shout_3, f, true).
observes(shout_3, g, true).


action(shout_4).
announcement(shout_4).
announces(shout_4, q, true).
formula(and(b(a,q),at_4)).
executable(shout_4, and(b(a,q),at_4)).
observes(shout_4, a, true).
observes(shout_4, c, true).
oblivious(shout_4, b).
oblivious(shout_4, d).
oblivious(shout_4, e).
oblivious(shout_4, f).
oblivious(shout_4, g).




% *** INITIALLY ***

% ** initial pointed **
initially(at_1).
initially(-at_2).
initially(-at_3).
initially(-at_4).
initially(q).

% ** initial beliefs **
agent_set(all_agents).
contains_ag(all_agents,a).
contains_ag(all_agents,b).
contains_ag(all_agents,c).
contains_ag(all_agents,d).
contains_ag(all_agents,e).
contains_ag(all_agents,f).
contains_ag(all_agents,g).
formula(c(all_agents,at_1)).
initially(c(all_agents,at_1)).
formula(c(all_agents,-at_2)).
initially(c(all_agents,-at_2)).
formula(c(all_agents,-at_3)).
initially(c(all_agents,-at_3)).
formula(c(all_agents,-at_4)).
initially(c(all_agents,-at_4)).



% ** commonly known fluent formulae **
init_sf(at_1).
init_sf(-at_2).
init_sf(-at_3).
init_sf(-at_4).


% *** GOALS ***
formula(b(c,q)).
goal(b(c,q)).
formula(b(b,q)).
formula(neg(b(b,q))).
goal(neg(b(b,q))).


