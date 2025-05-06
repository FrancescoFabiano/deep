###COMMENTS
I will try to explain in few lines the ideas behind using EFP 2.0 on doxastic domains
(it is still an "in development" part of the code since we do not have a precise semantics for lies or non-consistent beliefs yet).


- All the new introductions were only made to the code related to the new e-state representation, i.e., possibilities, so it is 
important to use the parameter "-st POSS" while experimenting with doxastic domains in EFP 2.0.


- To experiment with doxastic domain we introduced an action called "Lie". When executed a Lie: 
	- informs the fully_observant agents with the wrong value of the fluent considered;
	- informs the fully_observant agents that all the other fully_observant believe the wrong value of the fluent considered (even if in their prespective is the correct one);
	- informs the partially_observant agents that all the fully_observant agents believe the wrong value of the fluent if they already know the true value;
	- informs the partially_observant agents that all the fully_observant agents knows the fluent's value without knowing which one themselves.


- We created the 4/8 agents version of the Grapevine problems. The 8 agents one does not perform well at all but the one with 4 has good performances.
Anyway the encodings exploit the newly introduced type of action. It is important to notice that when executing an action in EFP 2.0 all the
nested formulae are generated following the observability relations. It possibile at any time to ask queries of any depth and nesting or on common knowledge.


- Since the concept of Lies and non-consistent belief is not fully formalized in our semantic we had to make some design choices.
	- First of all it is important to notice that if an agent knows "f" he can't learn "-f". We impose this limitation because we don't know
	which truth value of the fluent the agent should keep and we assumed that the first acquired is one she wants to keep.
	We forced this through the observability conditions, namely an agent is fully observant to a lie/annoucement only if she doesn't know
	the value of the fluent annouced and she is partially observant otherwise (or oblivious if she, for example, is not in the room).
	It is possibile to see the domains encoding in the examples.
	- It is also possibile to manipulate the action executability conditions following the same principle.
	For example if you don't want an agent (agent a) to tell a lie if she knows that someone that is listening
	knows the truth (agent b) you can impose the following condition: (B(a, ((-B(b,secret)), (-B(b,-secret)))))
	that reads as "agent a knows that agent b doesn't know the value of secret".
	
	
- A little note on the semantic. Since our planner captures all the beliefs of all the agents in each state it is not straigthforward to add concept such as lies or non-consistent beliefs.
As already said we imposed some limitations in order to have a working version (that capture a somewhat simplified concept of lies). Creating the knowledge\belief graph impose us to consider
what are the effect if a lies such as:
	- The one who lies believes that who is listening will accept the lie?
	- The one who is listening and knows the truth believes that the one who lied belives the lie herself (faulty perception) or she knows the truth but she is liying intentionally?
	- If someone tells me "f" but I believe "-f", should I stick with my knowledge or should I listen to the other agent? 
	- And so on.....
We are currently working on a way to formalize all these concepts in a solid semantics but it is easier said than done:)

- One final remark on the doamains encodings. Our parser is not really flexible so for every doubt do not esistate in contacting us.
	
### MAKEFILE
- **make**: to compile the planner.
- **make doxygen**: to compile the planner and the documentation (to check it open doxygen/Docs/html/index.html).
- **make clean**: removes all compilation-generated files.
- **make clean_build**: removes all compilation-generated files.
- **make clean_out**: removes all the file in out/ (the pdf visualization of the states).
- **make clear**: executes both **clean_build** and **clean_out**.
- **make fresh**: executes **clear** and also removes doxygen documentation.
- **make old**: cleans and compile the old version (1.0) of EFP.
- **make all**: executes **make doxygen** and **make_old**.
	
	
### Example
Following some examples of execution (we assume to be in the main project folder):

- Print of all the possible parameters
	- bin/efp.out
- Execution of Grapevine_doxastic with 4 agents and plan of length 6 on Possibilities without check for already visited state
	- bin/efp.out exp/Doxastic/Grapevine/Grapevine_4/Dox_Grapevine_4__pl_6.txt -st POSS
- Execution of Grapevine_doxastic with 8 agents and plan of length 2 on Possibilities with check for already visited state (FASTER CONFIG.)
	- bin/efp.out exp/Doxastic/Grapevine/Grapevine_8/Dox_Grapevine_8__pl_2.txt -st POSS -check_visited
- Test of sequential action execution (-e parameter):
	- bin/efp.out exp/Doxastic/Grapevine/Grapevine_4/Dox_Grapevine_4__pl_2.txt -st POSS -e fib_a_1, fib_b_1, right_b, share_c_1, left_b, fib_c_1


- Execution of CC domain (example for any non-doxastic domain) on Kripke Structure with check for already visited state
	- bin/efp.out exp/ICAPS20/CC/CC_2_2_4/CC_2_2_4__pl_5.txt -check_visited
- Execution of Coin_In_The_Box domain (example for any non-doxastic domain) on Kripke Structure with check for already visited state and bisimulation reduction
	- bin/efp.out exp/ICAPS20/Coin_In_The_Box/Coin_in_the_Box__pl_6.txt -check_visited -bis PT


### Requirements
- flex
- bison