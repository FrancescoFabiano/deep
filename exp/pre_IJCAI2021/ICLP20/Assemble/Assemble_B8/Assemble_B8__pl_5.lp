#const maxT  = 5.
#const n_fl  = 4.
#const n_ag  = 2.
#const n_ags = 3.

%time(0..maxT).

% *** FLUENTS ***
fluent(assemble).
formula(assemble).
fluent(qualify_a).
formula(qualify_a).
fluent(qualify_b).
formula(qualify_b).
fluent(restart).
formula(restart).


% *** AGENTS ***
agent(a).
agent(b).


% *** ACTIONS ***
action(act_assemble).
ontic(act_assemble).
causes(act_assemble, assemble, true).
formula(b(a,qualify_a)).
formula(b(b,b(a,qualify_a))).
formula(b(a,b(b,b(a,qualify_a)))).
formula(b(b,b(a,b(b,b(a,qualify_a))))).
formula(b(a,b(b,b(a,b(b,b(a,qualify_a)))))).
formula(b(b,b(a,b(b,b(a,b(b,b(a,qualify_a))))))).
formula(b(a,b(b,b(a,b(b,b(a,b(b,b(a,qualify_a)))))))).
formula(b(b,b(a,b(b,b(a,b(b,b(a,b(b,b(a,qualify_a))))))))).
formula(b(b,qualify_b)).
formula(b(a,b(b,qualify_b))).
formula(b(b,b(a,b(b,qualify_b)))).
formula(b(a,b(b,b(a,b(b,qualify_b))))).
formula(b(b,b(a,b(b,b(a,b(b,qualify_b)))))).
formula(b(a,b(b,b(a,b(b,b(a,b(b,qualify_b))))))).
formula(b(b,b(a,b(b,b(a,b(b,b(a,b(b,qualify_b)))))))).
formula(b(a,b(b,b(a,b(b,b(a,b(b,b(a,b(b,qualify_b))))))))).
formula(and(b(b,b(a,b(b,b(a,b(b,b(a,b(b,b(a,qualify_a)))))))),b(a,b(b,b(a,b(b,b(a,b(b,b(a,b(b,qualify_b)))))))))).
executable(act_assemble, and(b(b,b(a,b(b,b(a,b(b,b(a,b(b,b(a,qualify_a)))))))),b(a,b(b,b(a,b(b,b(a,b(b,b(a,b(b,qualify_b)))))))))).
observes(act_assemble, a, true).
observes(act_assemble, b, true).


action(act_res).
ontic(act_res).
causes(act_res, restart, true).
formula(-qualify_a).
formula(b(b,-qualify_a)).
formula(b(a,b(b,-qualify_a))).
formula(-qualify_b).
formula(b(a,-qualify_b)).
formula(b(b,b(a,-qualify_b))).
formula(or(b(a,b(b,-qualify_a)),b(b,b(a,-qualify_b)))).
executable(act_res, or(b(a,b(b,-qualify_a)),b(b,b(a,-qualify_b)))).
observes(act_res, a, true).
observes(act_res, b, true).


action(sense_a).
sensing(sense_a).
determines(sense_a, qualify_a, true).
formula(neg(b(a,qualify_a))).
formula(b(a,-qualify_a)).
formula(neg(b(a,-qualify_a))).
formula(and(neg(b(a,qualify_a)),neg(b(a,-qualify_a)))).
executable(sense_a, and(neg(b(a,qualify_a)),neg(b(a,-qualify_a)))).
observes(sense_a, a, true).
oblivious(sense_a, b).


action(sense_b).
sensing(sense_b).
determines(sense_b, qualify_b, true).
formula(neg(b(b,qualify_b))).
formula(b(b,-qualify_b)).
formula(neg(b(b,-qualify_b))).
formula(and(neg(b(b,qualify_b)),neg(b(b,-qualify_b)))).
executable(sense_b, and(neg(b(b,qualify_b)),neg(b(b,-qualify_b)))).
observes(sense_b, b, true).
oblivious(sense_b, a).


action(tell_a).
announcement(tell_a).
announces(tell_a, qualify_b, true).
executable(tell_a, b(b,qualify_b)).
observes(tell_a, a, true).
observes(tell_a, b, true).


action(tell_b).
announcement(tell_b).
announces(tell_b, qualify_a, true).
executable(tell_b, b(a,qualify_a)).
observes(tell_b, a, true).
observes(tell_b, b, true).




% *** INITIALLY ***

% ** initial pointed **
formula(-assemble).
initially(-assemble).
initially(qualify_a).
initially(qualify_b).
formula(-restart).
initially(-restart).

% ** initial beliefs **



% ** commonly known fluent formulae **


% *** GOALS ***
formula(b(a,assemble)).
goal(b(a,assemble)).
formula(b(b,assemble)).
goal(b(b,assemble)).


