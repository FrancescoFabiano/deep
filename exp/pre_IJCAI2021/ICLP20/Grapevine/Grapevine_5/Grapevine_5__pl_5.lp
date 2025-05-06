#const maxT  = 5.
#const n_fl  = 15.
#const n_ag  = 5.
#const n_ags = 31.

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
fluent(at_d_1).
formula(at_d_1).
fluent(at_d_2).
formula(at_d_2).
fluent(at_e_1).
formula(at_e_1).
fluent(at_e_2).
formula(at_e_2).
fluent(sa).
formula(sa).
fluent(sb).
formula(sb).
fluent(sc).
formula(sc).
fluent(sd).
formula(sd).
fluent(se).
formula(se).


% *** AGENTS ***
agent(a).
agent(b).
agent(c).
agent(d).
agent(e).


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
observes(left_a, d, at_d_2).
observes(left_a, e, at_e_2).


action(left_b).
ontic(left_b).
causes(left_b, at_b_1, true).
formula(-at_b_2).
causes(left_b, -at_b_2, true).
executable(left_b, at_b_2).
observes(left_b, a, at_a_2).
observes(left_b, b, at_b_2).
observes(left_b, c, at_c_2).
observes(left_b, d, at_d_2).
observes(left_b, e, at_e_2).


action(left_c).
ontic(left_c).
causes(left_c, at_c_1, true).
formula(-at_c_2).
causes(left_c, -at_c_2, true).
executable(left_c, at_c_2).
observes(left_c, a, at_a_2).
observes(left_c, b, at_b_2).
observes(left_c, c, at_c_2).
observes(left_c, d, at_d_2).
observes(left_c, e, at_e_2).


action(left_d).
ontic(left_d).
causes(left_d, at_d_1, true).
formula(-at_d_2).
causes(left_d, -at_d_2, true).
executable(left_d, at_d_2).
observes(left_d, a, at_a_2).
observes(left_d, b, at_b_2).
observes(left_d, c, at_c_2).
observes(left_d, d, at_d_2).
observes(left_d, e, at_e_2).


action(left_e).
ontic(left_e).
causes(left_e, at_e_1, true).
formula(-at_e_2).
causes(left_e, -at_e_2, true).
executable(left_e, at_e_2).
observes(left_e, a, at_a_2).
observes(left_e, b, at_b_2).
observes(left_e, c, at_c_2).
observes(left_e, d, at_d_2).
observes(left_e, e, at_e_2).


action(right_a).
ontic(right_a).
formula(-at_a_1).
causes(right_a, -at_a_1, true).
causes(right_a, at_a_2, true).
executable(right_a, at_a_1).
observes(right_a, a, at_a_1).
observes(right_a, b, at_b_1).
observes(right_a, c, at_c_1).
observes(right_a, d, at_d_1).
observes(right_a, e, at_e_1).


action(right_b).
ontic(right_b).
formula(-at_b_1).
causes(right_b, -at_b_1, true).
causes(right_b, at_b_2, true).
executable(right_b, at_b_1).
observes(right_b, a, at_a_1).
observes(right_b, b, at_b_1).
observes(right_b, c, at_c_1).
observes(right_b, d, at_d_1).
observes(right_b, e, at_e_1).


action(right_c).
ontic(right_c).
formula(-at_c_1).
causes(right_c, -at_c_1, true).
causes(right_c, at_c_2, true).
executable(right_c, at_c_1).
observes(right_c, a, at_a_1).
observes(right_c, b, at_b_1).
observes(right_c, c, at_c_1).
observes(right_c, d, at_d_1).
observes(right_c, e, at_e_1).


action(right_d).
ontic(right_d).
formula(-at_d_1).
causes(right_d, -at_d_1, true).
causes(right_d, at_d_2, true).
executable(right_d, at_d_1).
observes(right_d, a, at_a_1).
observes(right_d, b, at_b_1).
observes(right_d, c, at_c_1).
observes(right_d, d, at_d_1).
observes(right_d, e, at_e_1).


action(right_e).
ontic(right_e).
formula(-at_e_1).
causes(right_e, -at_e_1, true).
causes(right_e, at_e_2, true).
executable(right_e, at_e_1).
observes(right_e, a, at_a_1).
observes(right_e, b, at_b_1).
observes(right_e, c, at_c_1).
observes(right_e, d, at_d_1).
observes(right_e, e, at_e_1).


action(share_a_sa_1).
announcement(share_a_sa_1).
announces(share_a_sa_1, sa, true).
formula(b(a,sa)).
formula(and(b(a,sa),at_a_1)).
executable(share_a_sa_1, and(b(a,sa),at_a_1)).
observes(share_a_sa_1, a, at_a_1).
observes(share_a_sa_1, b, at_b_1).
observes(share_a_sa_1, c, at_c_1).
observes(share_a_sa_1, d, at_d_1).
observes(share_a_sa_1, e, at_e_1).


action(share_a_sa_2).
announcement(share_a_sa_2).
announces(share_a_sa_2, sa, true).
formula(and(b(a,sa),at_a_2)).
executable(share_a_sa_2, and(b(a,sa),at_a_2)).
observes(share_a_sa_2, a, at_a_2).
observes(share_a_sa_2, b, at_b_2).
observes(share_a_sa_2, c, at_c_2).
observes(share_a_sa_2, d, at_d_2).
observes(share_a_sa_2, e, at_e_2).


action(share_a_sb_1).
announcement(share_a_sb_1).
announces(share_a_sb_1, sb, true).
formula(b(a,sb)).
formula(and(b(a,sb),at_a_1)).
executable(share_a_sb_1, and(b(a,sb),at_a_1)).
observes(share_a_sb_1, a, at_a_1).
observes(share_a_sb_1, b, at_b_1).
observes(share_a_sb_1, c, at_c_1).
observes(share_a_sb_1, d, at_d_1).
observes(share_a_sb_1, e, at_e_1).


action(share_a_sb_2).
announcement(share_a_sb_2).
announces(share_a_sb_2, sb, true).
formula(and(b(a,sb),at_a_2)).
executable(share_a_sb_2, and(b(a,sb),at_a_2)).
observes(share_a_sb_2, a, at_a_2).
observes(share_a_sb_2, b, at_b_2).
observes(share_a_sb_2, c, at_c_2).
observes(share_a_sb_2, d, at_d_2).
observes(share_a_sb_2, e, at_e_2).


action(share_a_sc_1).
announcement(share_a_sc_1).
announces(share_a_sc_1, sc, true).
formula(b(a,sc)).
formula(and(b(a,sc),at_a_1)).
executable(share_a_sc_1, and(b(a,sc),at_a_1)).
observes(share_a_sc_1, a, at_a_1).
observes(share_a_sc_1, b, at_b_1).
observes(share_a_sc_1, c, at_c_1).
observes(share_a_sc_1, d, at_d_1).
observes(share_a_sc_1, e, at_e_1).


action(share_a_sc_2).
announcement(share_a_sc_2).
announces(share_a_sc_2, sc, true).
formula(and(b(a,sc),at_a_2)).
executable(share_a_sc_2, and(b(a,sc),at_a_2)).
observes(share_a_sc_2, a, at_a_2).
observes(share_a_sc_2, b, at_b_2).
observes(share_a_sc_2, c, at_c_2).
observes(share_a_sc_2, d, at_d_2).
observes(share_a_sc_2, e, at_e_2).


action(share_a_sd_1).
announcement(share_a_sd_1).
announces(share_a_sd_1, sd, true).
formula(b(a,sd)).
formula(and(b(a,sd),at_a_1)).
executable(share_a_sd_1, and(b(a,sd),at_a_1)).
observes(share_a_sd_1, a, at_a_1).
observes(share_a_sd_1, b, at_b_1).
observes(share_a_sd_1, c, at_c_1).
observes(share_a_sd_1, d, at_d_1).
observes(share_a_sd_1, e, at_e_1).


action(share_a_sd_2).
announcement(share_a_sd_2).
announces(share_a_sd_2, sd, true).
formula(and(b(a,sd),at_a_2)).
executable(share_a_sd_2, and(b(a,sd),at_a_2)).
observes(share_a_sd_2, a, at_a_2).
observes(share_a_sd_2, b, at_b_2).
observes(share_a_sd_2, c, at_c_2).
observes(share_a_sd_2, d, at_d_2).
observes(share_a_sd_2, e, at_e_2).


action(share_a_se_1).
announcement(share_a_se_1).
announces(share_a_se_1, se, true).
formula(b(a,se)).
formula(and(b(a,se),at_a_1)).
executable(share_a_se_1, and(b(a,se),at_a_1)).
observes(share_a_se_1, a, at_a_1).
observes(share_a_se_1, b, at_b_1).
observes(share_a_se_1, c, at_c_1).
observes(share_a_se_1, d, at_d_1).
observes(share_a_se_1, e, at_e_1).


action(share_a_se_2).
announcement(share_a_se_2).
announces(share_a_se_2, se, true).
formula(and(b(a,se),at_a_2)).
executable(share_a_se_2, and(b(a,se),at_a_2)).
observes(share_a_se_2, a, at_a_2).
observes(share_a_se_2, b, at_b_2).
observes(share_a_se_2, c, at_c_2).
observes(share_a_se_2, d, at_d_2).
observes(share_a_se_2, e, at_e_2).


action(share_b_sa_1).
announcement(share_b_sa_1).
announces(share_b_sa_1, sa, true).
formula(b(b,sa)).
formula(and(b(b,sa),at_b_1)).
executable(share_b_sa_1, and(b(b,sa),at_b_1)).
observes(share_b_sa_1, a, at_a_1).
observes(share_b_sa_1, b, at_b_1).
observes(share_b_sa_1, c, at_c_1).
observes(share_b_sa_1, d, at_d_1).
observes(share_b_sa_1, e, at_e_1).


action(share_b_sa_2).
announcement(share_b_sa_2).
announces(share_b_sa_2, sa, true).
formula(and(b(b,sa),at_b_2)).
executable(share_b_sa_2, and(b(b,sa),at_b_2)).
observes(share_b_sa_2, a, at_a_2).
observes(share_b_sa_2, b, at_b_2).
observes(share_b_sa_2, c, at_c_2).
observes(share_b_sa_2, d, at_d_2).
observes(share_b_sa_2, e, at_e_2).


action(share_b_sb_1).
announcement(share_b_sb_1).
announces(share_b_sb_1, sb, true).
formula(b(b,sb)).
formula(and(b(b,sb),at_b_1)).
executable(share_b_sb_1, and(b(b,sb),at_b_1)).
observes(share_b_sb_1, a, at_a_1).
observes(share_b_sb_1, b, at_b_1).
observes(share_b_sb_1, c, at_c_1).
observes(share_b_sb_1, d, at_d_1).
observes(share_b_sb_1, e, at_e_1).


action(share_b_sb_2).
announcement(share_b_sb_2).
announces(share_b_sb_2, sb, true).
formula(and(b(b,sb),at_b_2)).
executable(share_b_sb_2, and(b(b,sb),at_b_2)).
observes(share_b_sb_2, a, at_a_2).
observes(share_b_sb_2, b, at_b_2).
observes(share_b_sb_2, c, at_c_2).
observes(share_b_sb_2, d, at_d_2).
observes(share_b_sb_2, e, at_e_2).


action(share_b_sc_1).
announcement(share_b_sc_1).
announces(share_b_sc_1, sc, true).
formula(b(b,sc)).
formula(and(b(b,sc),at_b_1)).
executable(share_b_sc_1, and(b(b,sc),at_b_1)).
observes(share_b_sc_1, a, at_a_1).
observes(share_b_sc_1, b, at_b_1).
observes(share_b_sc_1, c, at_c_1).
observes(share_b_sc_1, d, at_d_1).
observes(share_b_sc_1, e, at_e_1).


action(share_b_sc_2).
announcement(share_b_sc_2).
announces(share_b_sc_2, sc, true).
formula(and(b(b,sc),at_b_2)).
executable(share_b_sc_2, and(b(b,sc),at_b_2)).
observes(share_b_sc_2, a, at_a_2).
observes(share_b_sc_2, b, at_b_2).
observes(share_b_sc_2, c, at_c_2).
observes(share_b_sc_2, d, at_d_2).
observes(share_b_sc_2, e, at_e_2).


action(share_b_sd_1).
announcement(share_b_sd_1).
announces(share_b_sd_1, sd, true).
formula(b(b,sd)).
formula(and(b(b,sd),at_b_1)).
executable(share_b_sd_1, and(b(b,sd),at_b_1)).
observes(share_b_sd_1, a, at_a_1).
observes(share_b_sd_1, b, at_b_1).
observes(share_b_sd_1, c, at_c_1).
observes(share_b_sd_1, d, at_d_1).
observes(share_b_sd_1, e, at_e_1).


action(share_b_sd_2).
announcement(share_b_sd_2).
announces(share_b_sd_2, sd, true).
formula(and(b(b,sd),at_b_2)).
executable(share_b_sd_2, and(b(b,sd),at_b_2)).
observes(share_b_sd_2, a, at_a_2).
observes(share_b_sd_2, b, at_b_2).
observes(share_b_sd_2, c, at_c_2).
observes(share_b_sd_2, d, at_d_2).
observes(share_b_sd_2, e, at_e_2).


action(share_b_se_1).
announcement(share_b_se_1).
announces(share_b_se_1, se, true).
formula(b(b,se)).
formula(and(b(b,se),at_b_1)).
executable(share_b_se_1, and(b(b,se),at_b_1)).
observes(share_b_se_1, a, at_a_1).
observes(share_b_se_1, b, at_b_1).
observes(share_b_se_1, c, at_c_1).
observes(share_b_se_1, d, at_d_1).
observes(share_b_se_1, e, at_e_1).


action(share_b_se_2).
announcement(share_b_se_2).
announces(share_b_se_2, se, true).
formula(and(b(b,se),at_b_2)).
executable(share_b_se_2, and(b(b,se),at_b_2)).
observes(share_b_se_2, a, at_a_2).
observes(share_b_se_2, b, at_b_2).
observes(share_b_se_2, c, at_c_2).
observes(share_b_se_2, d, at_d_2).
observes(share_b_se_2, e, at_e_2).


action(share_c_sa_1).
announcement(share_c_sa_1).
announces(share_c_sa_1, sa, true).
formula(b(c,sa)).
formula(and(b(c,sa),at_c_1)).
executable(share_c_sa_1, and(b(c,sa),at_c_1)).
observes(share_c_sa_1, a, at_a_1).
observes(share_c_sa_1, b, at_b_1).
observes(share_c_sa_1, c, at_c_1).
observes(share_c_sa_1, d, at_d_1).
observes(share_c_sa_1, e, at_e_1).


action(share_c_sa_2).
announcement(share_c_sa_2).
announces(share_c_sa_2, sa, true).
formula(and(b(c,sa),at_c_2)).
executable(share_c_sa_2, and(b(c,sa),at_c_2)).
observes(share_c_sa_2, a, at_a_2).
observes(share_c_sa_2, b, at_b_2).
observes(share_c_sa_2, c, at_c_2).
observes(share_c_sa_2, d, at_d_2).
observes(share_c_sa_2, e, at_e_2).


action(share_c_sb_1).
announcement(share_c_sb_1).
announces(share_c_sb_1, sb, true).
formula(b(c,sb)).
formula(and(b(c,sb),at_c_1)).
executable(share_c_sb_1, and(b(c,sb),at_c_1)).
observes(share_c_sb_1, a, at_a_1).
observes(share_c_sb_1, b, at_b_1).
observes(share_c_sb_1, c, at_c_1).
observes(share_c_sb_1, d, at_d_1).
observes(share_c_sb_1, e, at_e_1).


action(share_c_sb_2).
announcement(share_c_sb_2).
announces(share_c_sb_2, sb, true).
formula(and(b(c,sb),at_c_2)).
executable(share_c_sb_2, and(b(c,sb),at_c_2)).
observes(share_c_sb_2, a, at_a_2).
observes(share_c_sb_2, b, at_b_2).
observes(share_c_sb_2, c, at_c_2).
observes(share_c_sb_2, d, at_d_2).
observes(share_c_sb_2, e, at_e_2).


action(share_c_sc_1).
announcement(share_c_sc_1).
announces(share_c_sc_1, sc, true).
formula(b(c,sc)).
formula(and(b(c,sc),at_c_1)).
executable(share_c_sc_1, and(b(c,sc),at_c_1)).
observes(share_c_sc_1, a, at_a_1).
observes(share_c_sc_1, b, at_b_1).
observes(share_c_sc_1, c, at_c_1).
observes(share_c_sc_1, d, at_d_1).
observes(share_c_sc_1, e, at_e_1).


action(share_c_sc_2).
announcement(share_c_sc_2).
announces(share_c_sc_2, sc, true).
formula(and(b(c,sc),at_c_2)).
executable(share_c_sc_2, and(b(c,sc),at_c_2)).
observes(share_c_sc_2, a, at_a_2).
observes(share_c_sc_2, b, at_b_2).
observes(share_c_sc_2, c, at_c_2).
observes(share_c_sc_2, d, at_d_2).
observes(share_c_sc_2, e, at_e_2).


action(share_c_sd_1).
announcement(share_c_sd_1).
announces(share_c_sd_1, sd, true).
formula(b(c,sd)).
formula(and(b(c,sd),at_c_1)).
executable(share_c_sd_1, and(b(c,sd),at_c_1)).
observes(share_c_sd_1, a, at_a_1).
observes(share_c_sd_1, b, at_b_1).
observes(share_c_sd_1, c, at_c_1).
observes(share_c_sd_1, d, at_d_1).
observes(share_c_sd_1, e, at_e_1).


action(share_c_sd_2).
announcement(share_c_sd_2).
announces(share_c_sd_2, sd, true).
formula(and(b(c,sd),at_c_2)).
executable(share_c_sd_2, and(b(c,sd),at_c_2)).
observes(share_c_sd_2, a, at_a_2).
observes(share_c_sd_2, b, at_b_2).
observes(share_c_sd_2, c, at_c_2).
observes(share_c_sd_2, d, at_d_2).
observes(share_c_sd_2, e, at_e_2).


action(share_c_se_1).
announcement(share_c_se_1).
announces(share_c_se_1, se, true).
formula(b(c,se)).
formula(and(b(c,se),at_c_1)).
executable(share_c_se_1, and(b(c,se),at_c_1)).
observes(share_c_se_1, a, at_a_1).
observes(share_c_se_1, b, at_b_1).
observes(share_c_se_1, c, at_c_1).
observes(share_c_se_1, d, at_d_1).
observes(share_c_se_1, e, at_e_1).


action(share_c_se_2).
announcement(share_c_se_2).
announces(share_c_se_2, se, true).
formula(and(b(c,se),at_c_2)).
executable(share_c_se_2, and(b(c,se),at_c_2)).
observes(share_c_se_2, a, at_a_2).
observes(share_c_se_2, b, at_b_2).
observes(share_c_se_2, c, at_c_2).
observes(share_c_se_2, d, at_d_2).
observes(share_c_se_2, e, at_e_2).


action(share_d_sa_1).
announcement(share_d_sa_1).
announces(share_d_sa_1, sa, true).
formula(b(d,sa)).
formula(and(b(d,sa),at_d_1)).
executable(share_d_sa_1, and(b(d,sa),at_d_1)).
observes(share_d_sa_1, a, at_a_1).
observes(share_d_sa_1, b, at_b_1).
observes(share_d_sa_1, c, at_c_1).
observes(share_d_sa_1, d, at_d_1).
observes(share_d_sa_1, e, at_e_1).


action(share_d_sa_2).
announcement(share_d_sa_2).
announces(share_d_sa_2, sa, true).
formula(and(b(d,sa),at_d_2)).
executable(share_d_sa_2, and(b(d,sa),at_d_2)).
observes(share_d_sa_2, a, at_a_2).
observes(share_d_sa_2, b, at_b_2).
observes(share_d_sa_2, c, at_c_2).
observes(share_d_sa_2, d, at_d_2).
observes(share_d_sa_2, e, at_e_2).


action(share_d_sb_1).
announcement(share_d_sb_1).
announces(share_d_sb_1, sb, true).
formula(b(d,sb)).
formula(and(b(d,sb),at_d_1)).
executable(share_d_sb_1, and(b(d,sb),at_d_1)).
observes(share_d_sb_1, a, at_a_1).
observes(share_d_sb_1, b, at_b_1).
observes(share_d_sb_1, c, at_c_1).
observes(share_d_sb_1, d, at_d_1).
observes(share_d_sb_1, e, at_e_1).


action(share_d_sb_2).
announcement(share_d_sb_2).
announces(share_d_sb_2, sb, true).
formula(and(b(d,sb),at_d_2)).
executable(share_d_sb_2, and(b(d,sb),at_d_2)).
observes(share_d_sb_2, a, at_a_2).
observes(share_d_sb_2, b, at_b_2).
observes(share_d_sb_2, c, at_c_2).
observes(share_d_sb_2, d, at_d_2).
observes(share_d_sb_2, e, at_e_2).


action(share_d_sc_1).
announcement(share_d_sc_1).
announces(share_d_sc_1, sc, true).
formula(b(d,sc)).
formula(and(b(d,sc),at_d_1)).
executable(share_d_sc_1, and(b(d,sc),at_d_1)).
observes(share_d_sc_1, a, at_a_1).
observes(share_d_sc_1, b, at_b_1).
observes(share_d_sc_1, c, at_c_1).
observes(share_d_sc_1, d, at_d_1).
observes(share_d_sc_1, e, at_e_1).


action(share_d_sc_2).
announcement(share_d_sc_2).
announces(share_d_sc_2, sc, true).
formula(and(b(d,sc),at_d_2)).
executable(share_d_sc_2, and(b(d,sc),at_d_2)).
observes(share_d_sc_2, a, at_a_2).
observes(share_d_sc_2, b, at_b_2).
observes(share_d_sc_2, c, at_c_2).
observes(share_d_sc_2, d, at_d_2).
observes(share_d_sc_2, e, at_e_2).


action(share_d_sd_1).
announcement(share_d_sd_1).
announces(share_d_sd_1, sd, true).
formula(b(d,sd)).
formula(and(b(d,sd),at_d_1)).
executable(share_d_sd_1, and(b(d,sd),at_d_1)).
observes(share_d_sd_1, a, at_a_1).
observes(share_d_sd_1, b, at_b_1).
observes(share_d_sd_1, c, at_c_1).
observes(share_d_sd_1, d, at_d_1).
observes(share_d_sd_1, e, at_e_1).


action(share_d_sd_2).
announcement(share_d_sd_2).
announces(share_d_sd_2, sd, true).
formula(and(b(d,sd),at_d_2)).
executable(share_d_sd_2, and(b(d,sd),at_d_2)).
observes(share_d_sd_2, a, at_a_2).
observes(share_d_sd_2, b, at_b_2).
observes(share_d_sd_2, c, at_c_2).
observes(share_d_sd_2, d, at_d_2).
observes(share_d_sd_2, e, at_e_2).


action(share_d_se_1).
announcement(share_d_se_1).
announces(share_d_se_1, se, true).
formula(b(d,se)).
formula(and(b(d,se),at_d_1)).
executable(share_d_se_1, and(b(d,se),at_d_1)).
observes(share_d_se_1, a, at_a_1).
observes(share_d_se_1, b, at_b_1).
observes(share_d_se_1, c, at_c_1).
observes(share_d_se_1, d, at_d_1).
observes(share_d_se_1, e, at_e_1).


action(share_d_se_2).
announcement(share_d_se_2).
announces(share_d_se_2, se, true).
formula(and(b(d,se),at_d_2)).
executable(share_d_se_2, and(b(d,se),at_d_2)).
observes(share_d_se_2, a, at_a_2).
observes(share_d_se_2, b, at_b_2).
observes(share_d_se_2, c, at_c_2).
observes(share_d_se_2, d, at_d_2).
observes(share_d_se_2, e, at_e_2).


action(share_e_sa_1).
announcement(share_e_sa_1).
announces(share_e_sa_1, sa, true).
formula(b(e,sa)).
formula(and(b(e,sa),at_e_1)).
executable(share_e_sa_1, and(b(e,sa),at_e_1)).
observes(share_e_sa_1, a, at_a_1).
observes(share_e_sa_1, b, at_b_1).
observes(share_e_sa_1, c, at_c_1).
observes(share_e_sa_1, d, at_d_1).
observes(share_e_sa_1, e, at_e_1).


action(share_e_sa_2).
announcement(share_e_sa_2).
announces(share_e_sa_2, sa, true).
formula(and(b(e,sa),at_e_2)).
executable(share_e_sa_2, and(b(e,sa),at_e_2)).
observes(share_e_sa_2, a, at_a_2).
observes(share_e_sa_2, b, at_b_2).
observes(share_e_sa_2, c, at_c_2).
observes(share_e_sa_2, d, at_d_2).
observes(share_e_sa_2, e, at_e_2).


action(share_e_sb_1).
announcement(share_e_sb_1).
announces(share_e_sb_1, sb, true).
formula(b(e,sb)).
formula(and(b(e,sb),at_e_1)).
executable(share_e_sb_1, and(b(e,sb),at_e_1)).
observes(share_e_sb_1, a, at_a_1).
observes(share_e_sb_1, b, at_b_1).
observes(share_e_sb_1, c, at_c_1).
observes(share_e_sb_1, d, at_d_1).
observes(share_e_sb_1, e, at_e_1).


action(share_e_sb_2).
announcement(share_e_sb_2).
announces(share_e_sb_2, sb, true).
formula(and(b(e,sb),at_e_2)).
executable(share_e_sb_2, and(b(e,sb),at_e_2)).
observes(share_e_sb_2, a, at_a_2).
observes(share_e_sb_2, b, at_b_2).
observes(share_e_sb_2, c, at_c_2).
observes(share_e_sb_2, d, at_d_2).
observes(share_e_sb_2, e, at_e_2).


action(share_e_sc_1).
announcement(share_e_sc_1).
announces(share_e_sc_1, sc, true).
formula(b(e,sc)).
formula(and(b(e,sc),at_e_1)).
executable(share_e_sc_1, and(b(e,sc),at_e_1)).
observes(share_e_sc_1, a, at_a_1).
observes(share_e_sc_1, b, at_b_1).
observes(share_e_sc_1, c, at_c_1).
observes(share_e_sc_1, d, at_d_1).
observes(share_e_sc_1, e, at_e_1).


action(share_e_sc_2).
announcement(share_e_sc_2).
announces(share_e_sc_2, sc, true).
formula(and(b(e,sc),at_e_2)).
executable(share_e_sc_2, and(b(e,sc),at_e_2)).
observes(share_e_sc_2, a, at_a_2).
observes(share_e_sc_2, b, at_b_2).
observes(share_e_sc_2, c, at_c_2).
observes(share_e_sc_2, d, at_d_2).
observes(share_e_sc_2, e, at_e_2).


action(share_e_sd_1).
announcement(share_e_sd_1).
announces(share_e_sd_1, sd, true).
formula(b(e,sd)).
formula(and(b(e,sd),at_e_1)).
executable(share_e_sd_1, and(b(e,sd),at_e_1)).
observes(share_e_sd_1, a, at_a_1).
observes(share_e_sd_1, b, at_b_1).
observes(share_e_sd_1, c, at_c_1).
observes(share_e_sd_1, d, at_d_1).
observes(share_e_sd_1, e, at_e_1).


action(share_e_sd_2).
announcement(share_e_sd_2).
announces(share_e_sd_2, sd, true).
formula(and(b(e,sd),at_e_2)).
executable(share_e_sd_2, and(b(e,sd),at_e_2)).
observes(share_e_sd_2, a, at_a_2).
observes(share_e_sd_2, b, at_b_2).
observes(share_e_sd_2, c, at_c_2).
observes(share_e_sd_2, d, at_d_2).
observes(share_e_sd_2, e, at_e_2).


action(share_e_se_1).
announcement(share_e_se_1).
announces(share_e_se_1, se, true).
formula(b(e,se)).
formula(and(b(e,se),at_e_1)).
executable(share_e_se_1, and(b(e,se),at_e_1)).
observes(share_e_se_1, a, at_a_1).
observes(share_e_se_1, b, at_b_1).
observes(share_e_se_1, c, at_c_1).
observes(share_e_se_1, d, at_d_1).
observes(share_e_se_1, e, at_e_1).


action(share_e_se_2).
announcement(share_e_se_2).
announces(share_e_se_2, se, true).
formula(and(b(e,se),at_e_2)).
executable(share_e_se_2, and(b(e,se),at_e_2)).
observes(share_e_se_2, a, at_a_2).
observes(share_e_se_2, b, at_b_2).
observes(share_e_se_2, c, at_c_2).
observes(share_e_se_2, d, at_d_2).
observes(share_e_se_2, e, at_e_2).




% *** INITIALLY ***

% ** initial pointed **
initially(at_a_1).
initially(-at_a_2).
initially(at_b_1).
initially(-at_b_2).
initially(at_c_1).
initially(-at_c_2).
initially(at_d_1).
initially(-at_d_2).
initially(at_e_1).
initially(-at_e_2).
initially(sa).
initially(sb).
initially(sc).
initially(sd).
initially(se).

% ** initial beliefs **
agent_set(all_agents).
contains_ag(all_agents,a).
contains_ag(all_agents,b).
contains_ag(all_agents,c).
contains_ag(all_agents,d).
contains_ag(all_agents,e).
formula(c(all_agents,at_a_1)).
initially(c(all_agents,at_a_1)).
formula(c(all_agents,at_b_1)).
initially(c(all_agents,at_b_1)).
formula(c(all_agents,at_c_1)).
initially(c(all_agents,at_c_1)).
formula(c(all_agents,at_d_1)).
initially(c(all_agents,at_d_1)).
formula(c(all_agents,at_e_1)).
initially(c(all_agents,at_e_1)).
formula(c(all_agents,-at_a_2)).
initially(c(all_agents,-at_a_2)).
formula(c(all_agents,-at_b_2)).
initially(c(all_agents,-at_b_2)).
formula(c(all_agents,-at_c_2)).
initially(c(all_agents,-at_c_2)).
formula(c(all_agents,-at_d_2)).
initially(c(all_agents,-at_d_2)).
formula(c(all_agents,-at_e_2)).
initially(c(all_agents,-at_e_2)).
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
formula(neg(sd)).
formula(b(d,neg(sd))).
formula(or(b(d,sd),b(d,neg(sd)))).
formula(c(all_agents,or(b(d,sd),b(d,neg(sd))))).
initially(c(all_agents,or(b(d,sd),b(d,neg(sd))))).
formula(neg(se)).
formula(b(e,neg(se))).
formula(or(b(e,se),b(e,neg(se)))).
formula(c(all_agents,or(b(e,se),b(e,neg(se))))).
initially(c(all_agents,or(b(e,se),b(e,neg(se))))).



% ** commonly known fluent formulae **
init_sf(at_a_1).
init_sf(at_b_1).
init_sf(at_c_1).
init_sf(at_d_1).
init_sf(at_e_1).
init_sf(-at_a_2).
init_sf(-at_b_2).
init_sf(-at_c_2).
init_sf(-at_d_2).
init_sf(-at_e_2).


% *** GOALS ***
goal(b(a,sb)).
goal(b(b,sc)).
goal(b(c,sa)).
formula(neg(b(a,sc))).
goal(neg(b(a,sc))).
formula(neg(b(c,sb))).
goal(neg(b(c,sb))).


