#const maxT  = 4.
#const n_fl  = 9.
#const n_ag  = 3.
#const n_ags = 7.

%time(0..maxT).

% *** FLUENTS ***
fluent(at_a_1).
formula(at_a_1).
fluent(at_a_2).
formula(at_a_2).
fluent(at_b_1).
formula(at_b_1).
fluent(at_b_2).
formula(at_b_2).
fluent(at_c_1).
formula(at_c_1).
fluent(at_c_2).
formula(at_c_2).
fluent(sa).
formula(sa).
fluent(sb).
formula(sb).
fluent(sc).
formula(sc).


% *** AGENTS ***
agent(a).
agent(b).
agent(c).


% *** ACTIONS ***
action(left_a).
ontic(left_a).
causes(left_a, at_a_1, true).
formula(-at_a_2).
causes(left_a, -at_a_2, true).
executable(left_a, at_a_2).
observes(left_a, a, at_a_2).
observes(left_a, b, at_b_2).
observes(left_a, c, at_c_2).


action(left_b).
ontic(left_b).
causes(left_b, at_b_1, true).
formula(-at_b_2).
causes(left_b, -at_b_2, true).
executable(left_b, at_b_2).
observes(left_b, a, at_a_2).
observes(left_b, b, at_b_2).
observes(left_b, c, at_c_2).


action(left_c).
ontic(left_c).
causes(left_c, at_c_1, true).
formula(-at_c_2).
causes(left_c, -at_c_2, true).
executable(left_c, at_c_2).
observes(left_c, a, at_a_2).
observes(left_c, b, at_b_2).
observes(left_c, c, at_c_2).


action(right_a).
ontic(right_a).
formula(-at_a_1).
causes(right_a, -at_a_1, true).
causes(right_a, at_a_2, true).
executable(right_a, at_a_1).
observes(right_a, a, at_a_1).
observes(right_a, b, at_b_1).
observes(right_a, c, at_c_1).


action(right_b).
ontic(right_b).
formula(-at_b_1).
causes(right_b, -at_b_1, true).
causes(right_b, at_b_2, true).
executable(right_b, at_b_1).
observes(right_b, a, at_a_1).
observes(right_b, b, at_b_1).
observes(right_b, c, at_c_1).


action(right_c).
ontic(right_c).
formula(-at_c_1).
causes(right_c, -at_c_1, true).
causes(right_c, at_c_2, true).
executable(right_c, at_c_1).
observes(right_c, a, at_a_1).
observes(right_c, b, at_b_1).
observes(right_c, c, at_c_1).


action(share_a_sa_1).
announcement(share_a_sa_1).
announces(share_a_sa_1, sa, true).
formula(b(a,sa)).
formula(and(b(a,sa),at_a_1)).
executable(share_a_sa_1, and(b(a,sa),at_a_1)).
observes(share_a_sa_1, a, at_a_1).
observes(share_a_sa_1, b, at_b_1).
observes(share_a_sa_1, c, at_c_1).


action(share_a_sa_2).
announcement(share_a_sa_2).
announces(share_a_sa_2, sa, true).
formula(and(b(a,sa),at_a_2)).
executable(share_a_sa_2, and(b(a,sa),at_a_2)).
observes(share_a_sa_2, a, at_a_2).
observes(share_a_sa_2, b, at_b_2).
observes(share_a_sa_2, c, at_c_2).


action(share_a_sb_1).
announcement(share_a_sb_1).
announces(share_a_sb_1, sb, true).
formula(b(a,sb)).
formula(and(b(a,sb),at_a_1)).
executable(share_a_sb_1, and(b(a,sb),at_a_1)).
observes(share_a_sb_1, a, at_a_1).
observes(share_a_sb_1, b, at_b_1).
observes(share_a_sb_1, c, at_c_1).


action(share_a_sb_2).
announcement(share_a_sb_2).
announces(share_a_sb_2, sb, true).
formula(and(b(a,sb),at_a_2)).
executable(share_a_sb_2, and(b(a,sb),at_a_2)).
observes(share_a_sb_2, a, at_a_2).
observes(share_a_sb_2, b, at_b_2).
observes(share_a_sb_2, c, at_c_2).


action(share_a_sc_1).
announcement(share_a_sc_1).
announces(share_a_sc_1, sc, true).
formula(b(a,sc)).
formula(and(b(a,sc),at_a_1)).
executable(share_a_sc_1, and(b(a,sc),at_a_1)).
observes(share_a_sc_1, a, at_a_1).
observes(share_a_sc_1, b, at_b_1).
observes(share_a_sc_1, c, at_c_1).


action(share_a_sc_2).
announcement(share_a_sc_2).
announces(share_a_sc_2, sc, true).
formula(and(b(a,sc),at_a_2)).
executable(share_a_sc_2, and(b(a,sc),at_a_2)).
observes(share_a_sc_2, a, at_a_2).
observes(share_a_sc_2, b, at_b_2).
observes(share_a_sc_2, c, at_c_2).


action(share_b_sa_1).
announcement(share_b_sa_1).
announces(share_b_sa_1, sa, true).
formula(b(b,sa)).
formula(and(b(b,sa),at_b_1)).
executable(share_b_sa_1, and(b(b,sa),at_b_1)).
observes(share_b_sa_1, a, at_a_1).
observes(share_b_sa_1, b, at_b_1).
observes(share_b_sa_1, c, at_c_1).


action(share_b_sa_2).
announcement(share_b_sa_2).
announces(share_b_sa_2, sa, true).
formula(and(b(b,sa),at_b_2)).
executable(share_b_sa_2, and(b(b,sa),at_b_2)).
observes(share_b_sa_2, a, at_a_2).
observes(share_b_sa_2, b, at_b_2).
observes(share_b_sa_2, c, at_c_2).


action(share_b_sb_1).
announcement(share_b_sb_1).
announces(share_b_sb_1, sb, true).
formula(b(b,sb)).
formula(and(b(b,sb),at_b_1)).
executable(share_b_sb_1, and(b(b,sb),at_b_1)).
observes(share_b_sb_1, a, at_a_1).
observes(share_b_sb_1, b, at_b_1).
observes(share_b_sb_1, c, at_c_1).


action(share_b_sb_2).
announcement(share_b_sb_2).
announces(share_b_sb_2, sb, true).
formula(and(b(b,sb),at_b_2)).
executable(share_b_sb_2, and(b(b,sb),at_b_2)).
observes(share_b_sb_2, a, at_a_2).
observes(share_b_sb_2, b, at_b_2).
observes(share_b_sb_2, c, at_c_2).


action(share_b_sc_1).
announcement(share_b_sc_1).
announces(share_b_sc_1, sc, true).
formula(b(b,sc)).
formula(and(b(b,sc),at_b_1)).
executable(share_b_sc_1, and(b(b,sc),at_b_1)).
observes(share_b_sc_1, a, at_a_1).
observes(share_b_sc_1, b, at_b_1).
observes(share_b_sc_1, c, at_c_1).


action(share_b_sc_2).
announcement(share_b_sc_2).
announces(share_b_sc_2, sc, true).
formula(and(b(b,sc),at_b_2)).
executable(share_b_sc_2, and(b(b,sc),at_b_2)).
observes(share_b_sc_2, a, at_a_2).
observes(share_b_sc_2, b, at_b_2).
observes(share_b_sc_2, c, at_c_2).


action(share_c_sa_1).
announcement(share_c_sa_1).
announces(share_c_sa_1, sa, true).
formula(b(c,sa)).
formula(and(b(c,sa),at_c_1)).
executable(share_c_sa_1, and(b(c,sa),at_c_1)).
observes(share_c_sa_1, a, at_a_1).
observes(share_c_sa_1, b, at_b_1).
observes(share_c_sa_1, c, at_c_1).


action(share_c_sa_2).
announcement(share_c_sa_2).
announces(share_c_sa_2, sa, true).
formula(and(b(c,sa),at_c_2)).
executable(share_c_sa_2, and(b(c,sa),at_c_2)).
observes(share_c_sa_2, a, at_a_2).
observes(share_c_sa_2, b, at_b_2).
observes(share_c_sa_2, c, at_c_2).


action(share_c_sb_1).
announcement(share_c_sb_1).
announces(share_c_sb_1, sb, true).
formula(b(c,sb)).
formula(and(b(c,sb),at_c_1)).
executable(share_c_sb_1, and(b(c,sb),at_c_1)).
observes(share_c_sb_1, a, at_a_1).
observes(share_c_sb_1, b, at_b_1).
observes(share_c_sb_1, c, at_c_1).


action(share_c_sb_2).
announcement(share_c_sb_2).
announces(share_c_sb_2, sb, true).
formula(and(b(c,sb),at_c_2)).
executable(share_c_sb_2, and(b(c,sb),at_c_2)).
observes(share_c_sb_2, a, at_a_2).
observes(share_c_sb_2, b, at_b_2).
observes(share_c_sb_2, c, at_c_2).


action(share_c_sc_1).
announcement(share_c_sc_1).
announces(share_c_sc_1, sc, true).
formula(b(c,sc)).
formula(and(b(c,sc),at_c_1)).
executable(share_c_sc_1, and(b(c,sc),at_c_1)).
observes(share_c_sc_1, a, at_a_1).
observes(share_c_sc_1, b, at_b_1).
observes(share_c_sc_1, c, at_c_1).


action(share_c_sc_2).
announcement(share_c_sc_2).
announces(share_c_sc_2, sc, true).
formula(and(b(c,sc),at_c_2)).
executable(share_c_sc_2, and(b(c,sc),at_c_2)).
observes(share_c_sc_2, a, at_a_2).
observes(share_c_sc_2, b, at_b_2).
observes(share_c_sc_2, c, at_c_2).




% *** INITIALLY ***

% ** initial pointed **
initially(at_a_1).
initially(-at_a_2).
initially(at_b_1).
initially(-at_b_2).
initially(at_c_1).
initially(-at_c_2).
initially(sa).
initially(sb).
initially(sc).

% ** initial beliefs **
agent_set(all_agents).
contains_ag(all_agents,a).
contains_ag(all_agents,b).
contains_ag(all_agents,c).
formula(c(all_agents,at_a_1)).
initially(c(all_agents,at_a_1)).
formula(c(all_agents,at_b_1)).
initially(c(all_agents,at_b_1)).
formula(c(all_agents,at_c_1)).
initially(c(all_agents,at_c_1)).
formula(c(all_agents,-at_a_2)).
initially(c(all_agents,-at_a_2)).
formula(c(all_agents,-at_b_2)).
initially(c(all_agents,-at_b_2)).
formula(c(all_agents,-at_c_2)).
initially(c(all_agents,-at_c_2)).
formula(neg(sa)).
formula(b(a,neg(sa))).
formula(or(b(a,sa),b(a,neg(sa)))).
formula(c(all_agents,or(b(a,sa),b(a,neg(sa))))).
initially(c(all_agents,or(b(a,sa),b(a,neg(sa))))).
formula(neg(sb)).
formula(b(b,neg(sb))).
formula(or(b(b,sb),b(b,neg(sb)))).
formula(c(all_agents,or(b(b,sb),b(b,neg(sb))))).
initially(c(all_agents,or(b(b,sb),b(b,neg(sb))))).
formula(neg(sc)).
formula(b(c,neg(sc))).
formula(or(b(c,sc),b(c,neg(sc)))).
formula(c(all_agents,or(b(c,sc),b(c,neg(sc))))).
initially(c(all_agents,or(b(c,sc),b(c,neg(sc))))).



% ** commonly known fluent formulae **
init_sf(at_a_1).
init_sf(at_b_1).
init_sf(at_c_1).
init_sf(-at_a_2).
init_sf(-at_b_2).
init_sf(-at_c_2).


% *** GOALS ***
goal(b(a,sb)).
goal(b(b,sc)).
goal(b(c,sa)).
formula(neg(b(a,sc))).
goal(neg(b(a,sc))).


