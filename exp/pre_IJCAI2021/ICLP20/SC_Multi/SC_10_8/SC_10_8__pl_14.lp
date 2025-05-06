#const maxT  = 14.
#const n_fl  = 11.
#const n_ag  = 8.
#const n_ags = 255.

%time(0..maxT).

% *** FLUENTS ***
fluent(at_1).
formula(at_1).
fluent(at_10).
formula(at_10).
fluent(at_2).
formula(at_2).
fluent(at_3).
formula(at_3).
fluent(at_4).
formula(at_4).
fluent(at_5).
formula(at_5).
fluent(at_6).
formula(at_6).
fluent(at_7).
formula(at_7).
fluent(at_8).
formula(at_8).
fluent(at_9).
formula(at_9).
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
agent(h).


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
ce_formula(at_5).
causes(left, at_4, at_5).
formula(-at_4).
causes(left, -at_4, at_4).
ce_formula(at_6).
causes(left, at_5, at_6).
formula(-at_5).
causes(left, -at_5, at_5).
ce_formula(at_7).
causes(left, at_6, at_7).
formula(-at_6).
causes(left, -at_6, at_6).
ce_formula(at_8).
causes(left, at_7, at_8).
formula(-at_7).
causes(left, -at_7, at_7).
formula(-at_8).
causes(left, -at_8, at_8).
formula(-at_1).
executable(left, -at_1).
observes(left, a, true).
observes(left, b, true).
observes(left, c, true).
observes(left, d, true).
observes(left, e, true).
observes(left, f, true).
observes(left, g, true).
observes(left, h, true).


action(right).
ontic(right).
ce_formula(at_1).
causes(right, -at_1, at_1).
causes(right, at_2, at_1).
causes(right, -at_2, at_2).
causes(right, at_3, at_2).
causes(right, -at_3, at_3).
causes(right, at_4, at_3).
causes(right, -at_4, at_4).
causes(right, at_5, at_4).
causes(right, -at_5, at_5).
causes(right, at_6, at_5).
causes(right, -at_6, at_6).
causes(right, at_7, at_6).
causes(right, -at_7, at_7).
causes(right, at_8, at_7).
executable(right, -at_8).
observes(right, a, true).
observes(right, b, true).
observes(right, c, true).
observes(right, d, true).
observes(right, e, true).
observes(right, f, true).
observes(right, g, true).
observes(right, h, true).


action(sense).
sensing(sense).
determines(sense, q, true).
executable(sense, at_8).
observes(sense, a, true).
oblivious(sense, b).
oblivious(sense, c).
oblivious(sense, d).
oblivious(sense, e).
oblivious(sense, f).
oblivious(sense, g).
oblivious(sense, h).


action(shout_1).
announcement(shout_1).
announces(shout_1, q, true).
formula(b(a,q)).
formula(and(b(a,q),at_1)).
executable(shout_1, and(b(a,q),at_1)).
observes(shout_1, a, true).
observes(shout_1, b, true).
observes(shout_1, f, true).
oblivious(shout_1, c).
oblivious(shout_1, d).
oblivious(shout_1, e).
oblivious(shout_1, g).
oblivious(shout_1, h).


action(shout_10).
announcement(shout_10).
announces(shout_10, q, true).
formula(and(b(a,q),at_10)).
executable(shout_10, and(b(a,q),at_10)).
observes(shout_10, a, true).
observes(shout_10, h, true).
oblivious(shout_10, b).
oblivious(shout_10, c).
oblivious(shout_10, d).
oblivious(shout_10, e).
oblivious(shout_10, f).
oblivious(shout_10, g).


action(shout_2).
announcement(shout_2).
announces(shout_2, q, true).
formula(and(b(a,q),at_2)).
executable(shout_2, and(b(a,q),at_2)).
observes(shout_2, a, true).
observes(shout_2, b, true).
observes(shout_2, c, true).
observes(shout_2, f, true).
oblivious(shout_2, d).
oblivious(shout_2, e).
oblivious(shout_2, g).
oblivious(shout_2, h).


action(shout_3).
announcement(shout_3).
announces(shout_3, q, true).
formula(and(b(a,q),at_3)).
executable(shout_3, and(b(a,q),at_3)).
observes(shout_3, a, true).
observes(shout_3, b, true).
observes(shout_3, c, true).
oblivious(shout_3, d).
oblivious(shout_3, e).
oblivious(shout_3, f).
oblivious(shout_3, g).
oblivious(shout_3, h).


action(shout_4).
announcement(shout_4).
announces(shout_4, q, true).
formula(and(b(a,q),at_4)).
executable(shout_4, and(b(a,q),at_4)).
observes(shout_4, a, true).
observes(shout_4, c, true).
observes(shout_4, d, true).
oblivious(shout_4, b).
oblivious(shout_4, e).
oblivious(shout_4, f).
oblivious(shout_4, g).
oblivious(shout_4, h).


action(shout_5).
announcement(shout_5).
announces(shout_5, q, true).
formula(and(b(a,q),at_5)).
executable(shout_5, and(b(a,q),at_5)).
observes(shout_5, a, true).
observes(shout_5, d, true).
observes(shout_5, e, true).
oblivious(shout_5, b).
oblivious(shout_5, c).
oblivious(shout_5, f).
oblivious(shout_5, g).
oblivious(shout_5, h).


action(shout_6).
announcement(shout_6).
announces(shout_6, q, true).
formula(and(b(a,q),at_6)).
executable(shout_6, and(b(a,q),at_6)).
observes(shout_6, a, true).
observes(shout_6, d, true).
observes(shout_6, e, true).
observes(shout_6, g, true).
oblivious(shout_6, b).
oblivious(shout_6, c).
oblivious(shout_6, f).
oblivious(shout_6, h).


action(shout_7).
announcement(shout_7).
announces(shout_7, q, true).
formula(and(b(a,q),at_7)).
executable(shout_7, and(b(a,q),at_7)).
observes(shout_7, a, true).
observes(shout_7, e, true).
observes(shout_7, g, true).
observes(shout_7, h, true).
oblivious(shout_7, b).
oblivious(shout_7, c).
oblivious(shout_7, d).
oblivious(shout_7, f).


action(shout_8).
announcement(shout_8).
announces(shout_8, q, true).
formula(and(b(a,q),at_8)).
executable(shout_8, and(b(a,q),at_8)).
observes(shout_8, a, true).
observes(shout_8, g, true).
observes(shout_8, h, true).
oblivious(shout_8, b).
oblivious(shout_8, c).
oblivious(shout_8, d).
oblivious(shout_8, e).
oblivious(shout_8, f).


action(shout_9).
announcement(shout_9).
announces(shout_9, q, true).
formula(and(b(a,q),at_9)).
executable(shout_9, and(b(a,q),at_9)).
observes(shout_9, a, true).
observes(shout_9, h, true).
oblivious(shout_9, b).
oblivious(shout_9, c).
oblivious(shout_9, d).
oblivious(shout_9, e).
oblivious(shout_9, f).
oblivious(shout_9, g).




% *** INITIALLY ***

% ** initial pointed **
initially(at_1).
formula(-at_10).
initially(-at_10).
initially(-at_2).
initially(-at_3).
initially(-at_4).
initially(-at_5).
initially(-at_6).
initially(-at_7).
initially(-at_8).
formula(-at_9).
initially(-at_9).
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
contains_ag(all_agents,h).
formula(c(all_agents,at_1)).
initially(c(all_agents,at_1)).
formula(c(all_agents,-at_2)).
initially(c(all_agents,-at_2)).
formula(c(all_agents,-at_3)).
initially(c(all_agents,-at_3)).
formula(c(all_agents,-at_4)).
initially(c(all_agents,-at_4)).
formula(c(all_agents,-at_5)).
initially(c(all_agents,-at_5)).
formula(c(all_agents,-at_6)).
initially(c(all_agents,-at_6)).
formula(c(all_agents,-at_7)).
initially(c(all_agents,-at_7)).
formula(c(all_agents,-at_8)).
initially(c(all_agents,-at_8)).
formula(c(all_agents,-at_9)).
initially(c(all_agents,-at_9)).
formula(c(all_agents,-at_10)).
initially(c(all_agents,-at_10)).



% ** commonly known fluent formulae **
init_sf(at_1).
init_sf(-at_2).
init_sf(-at_3).
init_sf(-at_4).
init_sf(-at_5).
init_sf(-at_6).
init_sf(-at_7).
init_sf(-at_8).
init_sf(-at_9).
init_sf(-at_10).


% *** GOALS ***
formula(b(b,b(a,q))).
goal(b(b,b(a,q))).


