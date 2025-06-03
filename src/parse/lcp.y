/* Parser for AL language */

%defines

%{
#include "../src/parse/Reader.h"

int yyerror(const char *s);
int yylex(void);

std::string get_negation(const std::string*);
bool is_consistent(StringsSet,StringsSet);
//StringSetsSet get_negateFluentForm(StringSetsSet);
StringSetsSet negate_or(StringsSet);
StringSetsSet negate_form(StringSetsSet);
StringSetsSet join_SL2(StringSetsSet, StringSetsSet);
void print_string_set(StringsSet);
void print_string_set_set(StringSetsSet);

extern std::unique_ptr<Reader> domain_reader;

%}

%union{
  std::string*	str_val;
  StringsSet*  str_list; 
  StringSetsSet* str_list2;
  Proposition* prop;
  PropositionsList* prop_list;
  BeliefFormulaParsed* bf;
  ParsedFormulaeList* init_nodes;
}

%start	input 

/* precedence among operators */
%left OR
%left COMMA

%token SEMICOLON
%token COMMA
%token OR
%token LEFT_PAREN
%token RIGHT_PAREN
%token RIGHT_BRAC
%token LEFT_BRAC
%token <str_val> ID
%token <str_val> NEGATION
%token <str_val> NUMBER

%token FLUENT
%token ACTION
%token IF
%token CAUSES
%token EXECUTABLE
%token IMPOSSIBLE
%token DETERMINE
%token AWAREOF
%token OBSERVES
%token ANNOUNCES
%token AGEXEC
%token INIT
%token GOAL
%token AGENT
%token MB
%token MC
%token ME
%token MD
%token LIE

%token WRT
%token TRUSTY
%token MISTRUSTY
%token UNTRUSTY
%token STUBBORN
%token KEEPER
%token INSECURE


%type <str_val> id
%type <str_val> constant
%type <str_val> param
%type <str_val> param_list

%type <str_val> fluent
%type <str_list> fluent_set
%type <str_list> fluent_det_list

%type <str_val> literal
%type <str_list> literal_list
%type <str_list> fluent_decl
%type <str_list> fluent_decls

%type <str_val> agent
%type <str_list> agent_list
%type <str_list> agent_decl
%type <str_list> agent_decls

%type <str_val> action
%type <str_list> action_list
%type <str_list> action_decl
%type <str_list> action_decls

/*DEBUG_WARNING_REMOVAL %type <str_list> if_ DEBUG_WARNING_REMOVAL*/
%type <str_list2> if_part_fluent
%type <bf> if_part_bf
%type <bf> init
%type <init_nodes> init_spec
%type <bf> goal
%type <init_nodes> goal_spec

%type <str_list2> formula
%type <bf> BeliefFormulaParsed
/* DEBUG_WARNING_REMOVAL %type <str_list2> gd_formula DEBUG_WARNING_REMOVAL*/

//%type <prop> static_law
%type <prop> dynamic_law
%type <prop> executability
//%type <prop> impossibility
%type <prop> Proposition
%type <prop> determine
%type <prop> awareness
%type <prop> observance
%type <prop> announcement
%type <prop> executing
%type <prop_list> domain





%%
input:		
|
fluent_decls 
action_decls
agent_decls 
domain
init_spec 
goal_spec
 { 
  domain_reader->m_fluents = *$1;
  domain_reader->m_actions = *$2;
  domain_reader->m_agents = *$3;
  domain_reader->m_propositions = *$4;
  domain_reader->m_bf_initially = *$5;
  domain_reader->m_bf_goal = *$6;
}
;

id:
ID {
  $$ = $1;
};

/* constant */
constant:
NUMBER {
  $$ = $1;
}
|
id {
  $$ = $1;
};

/* parameter */
param:
constant
{
  $$ = $1;
}
/*|
  variable*/
;

/* param list */
param_list:
param {
  $$ = $1;
}
|
param_list COMMA param
{
  $$ = new std::string(*$1 + "," + *$3);
};

/* fluent & fluent list*/
fluent:
id
{
  $$ = $1;
}
|
id LEFT_PAREN param_list RIGHT_PAREN
{
  $$ = new std::string(*$1 + "(" + *$3 + ")");
};

fluent_det_list:
fluent {
  $$ = new StringsSet;
  $$->insert(*$1);
};

fluent_set:
fluent {
  $$ = new StringsSet;
  $$->insert(*$1);
}
|
fluent_set COMMA fluent {
  $$ = $1;
  $$->insert(*$3);
};

/* literal list */
literal:
fluent {
  $$ = $1;
}
|
NEGATION fluent
{
  $$ = new std::string(*$1 + *$2);
};

literal_list:
literal
{
  $$ = new StringsSet;
  $$->insert(*$1);
} 
| 
literal_list COMMA literal {
  $$ = $1;  
  $$->insert(*$3);
}; 

formula:
literal {
  StringsSet s1;

  $$ = new StringSetsSet;

  s1.insert(*$1);

  $$->insert(s1);
}
| formula COMMA formula
{
  StringSetsSet::iterator it1;
  StringSetsSet::iterator it2;
  StringsSet ns;

  $$ = new StringSetsSet;

  for (it2 = $1->begin(); it2 != $1->end(); it2++) {
    for (it1 = $3->begin(); it1 != $3->end(); it1++){
	  if (is_consistent(*it2,*it1)) {
		ns = *it2;
		ns.insert(it1->begin(),it1->end());
		$$->insert(ns);
	  }
    }
  }  
}
| formula OR formula {
  $$ = $1;
  $$->insert($3->begin(),$3->end());
}
| LEFT_PAREN formula RIGHT_PAREN
{
  $$ = $2;
};

/* fluent declaration */
fluent_decl: 
FLUENT fluent_set SEMICOLON {
  $$ = $2;
};

fluent_decls:
/* empty */
{
  $$ = new StringsSet;
}
|
fluent_decls fluent_decl
{
  $1->insert($2->begin(),$2->end());
  $$ = $1;
}
;

/* action declaration */
action:
id {
  $$ = new std::string(*$1);
}
|
id LEFT_PAREN param_list RIGHT_PAREN {
  $$ = new std::string(*$1 + "(" + *$3 + ")");
};

action_list:
action {
  $$ = new StringsSet;
  $$->insert(*$1);
}
|
action_list COMMA action {
  $$ = $1;
  $$->insert(*$3);
};

action_decl: 
ACTION action_list SEMICOLON {
  $$ = $2;
};

action_decls:
/* empty */
{
  $$ = new StringsSet;
}
|
action_decls action_decl
{
  $1->insert($2->begin(),$2->end());
  $$ = $1;
}
;
/* agent declaration */
agent:
id {
  $$ = new std::string(*$1);
}
|
id LEFT_PAREN param_list RIGHT_PAREN {
  $$ = new std::string(*$1 + "(" + *$3 + ")");
};

agent_list:
agent {
  $$ = new StringsSet;
  $$->insert(*$1);
}
|
agent_list COMMA agent {
  $$ = $1;
  $$->insert(*$3);
};

agent_decl: 
AGENT agent_list SEMICOLON {
  $$ = $2;
};

agent_decls:
/* empty */
{
  $$ = new StringsSet;
}
|
agent_decls agent_decl
{
  $1->insert($2->begin(),$2->end());
  $$ = $1;
}
;

/* domain description */
/*DEBUG_WARNING_REMOVAL if_part: DEBUG_WARNING_REMOVAL*/ 
/* empty */
/*DEBUG_WARNING_REMOVAL {
  $$ = new StringsSet;
}
|
IF literal_list {
  $$ = $2;
};DEBUG_WARNING_REMOVAL*/

/* if part for BeliefFormulaParsed */
if_part_bf:
/* fail */
{
  $$ = new BeliefFormulaParsed;
  $$->set_formula_type(BeliefFormulaType::BF_EMPTY);
}
|
IF BeliefFormulaParsed {
  $$ = $2;
};

BeliefFormulaParsed:
formula{  
    $$ = new BeliefFormulaParsed;
    $$->set_formula_type(BeliefFormulaType::FLUENT_FORMULA);
    $$->set_string_fluent_formula(*$1);
}
|
MB LEFT_PAREN agent COMMA BeliefFormulaParsed RIGHT_PAREN {
   $$ = new BeliefFormulaParsed;
   $$->set_formula_type(BeliefFormulaType::BELIEF_FORMULA);
   $$->set_string_agent(*$3);
   $$->set_bf1(*$5);
}
|
BeliefFormulaParsed COMMA BeliefFormulaParsed {
   $$ = new BeliefFormulaParsed;
   $$->set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
   $$->set_operator(BeliefFormulaOperator::BF_AND);
   $$->set_bf1(*$1);
   $$->set_bf2(*$3);
}
|
BeliefFormulaParsed OR BeliefFormulaParsed {
   $$ = new BeliefFormulaParsed;
   $$->set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
   $$->set_operator(BeliefFormulaOperator::BF_OR);
   $$->set_bf1(*$1);
   $$->set_bf2(*$3);
}
|
LEFT_PAREN NEGATION BeliefFormulaParsed RIGHT_PAREN{
   $$ = new BeliefFormulaParsed;
   $$->set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
   $$->set_operator(BeliefFormulaOperator::BF_NOT);
   $$->set_bf1(*$3);
}
|
LEFT_PAREN BeliefFormulaParsed RIGHT_PAREN{
    $$ = new BeliefFormulaParsed;
    $$->set_formula_type(BeliefFormulaType::PROPOSITIONAL_FORMULA);
    $$->set_operator(BeliefFormulaOperator::BF_INPAREN);
    $$->set_bf1(*$2);
}
|
ME LEFT_PAREN LEFT_BRAC agent_list RIGHT_BRAC COMMA BeliefFormulaParsed RIGHT_PAREN {
   $$ = new BeliefFormulaParsed;
   $$->set_formula_type(BeliefFormulaType::E_FORMULA);
   $$->set_string_group_agents(*$4);
   $$->set_bf1(*$7);
}
|
MC LEFT_PAREN LEFT_BRAC agent_list RIGHT_BRAC COMMA BeliefFormulaParsed RIGHT_PAREN {
   $$ = new BeliefFormulaParsed;
   $$->set_formula_type(BeliefFormulaType::C_FORMULA);
   $$->set_string_group_agents(*$4);
   $$->set_bf1(*$7);
}
;



/* if part for fluent_formula */
if_part_fluent: 
/* empty */
{
  $$ = new StringSetsSet;
}
|
IF formula {
  $$ = $2;
};







/* static law
static_law:
literal_list if_part SEMICOLON
{
  $$ = new Proposition;
  $$->set_type(STATIC);
  $$->set_action_precondition(*$2);
  $$->set_action_effect(*$1);
};*/

/* dynamic law */
dynamic_law:
action CAUSES literal_list if_part_bf SEMICOLON 
{  
  $$ = new Proposition;
  $$->set_type(PropositionType::ONTIC);
  $$->set_action_name(*$1);
  $$->set_executability_conditions(*$4);
  //@TODO:Effect_Conversion | previously   $$->m_action_effect = *$3;
  $$->add_action_effect(*$3);
};

/* executability condition */
executability:
EXECUTABLE action if_part_bf SEMICOLON
{
  $$ = new Proposition;
  $$->set_type(PropositionType::EXECUTABILITY);
  $$->set_action_name(*$2);
  $$->set_executability_conditions(*$3);
};

/* determines condition */
determine:
action DETERMINE literal_list if_part_bf SEMICOLON
{
  $$ = new Proposition;
  $$->set_type(PropositionType::SENSING);
  $$->set_action_name(*$1);
  //@TODO:Effect_Conversion | previously   $$->m_action_effect = *$3;
  $$->add_action_effect(*$3);
  $$->set_executability_conditions(*$4);
};

/* announcement condition */
announcement:
action ANNOUNCES literal_list if_part_bf SEMICOLON
{
  $$ = new Proposition;
  $$->set_type(PropositionType::ANNOUNCEMENT);
  $$->set_action_name(*$1);
  $$->add_action_effect(*$3);
  $$->set_executability_conditions(*$4);

};


/* awareness condition */
awareness:
agent AWAREOF action if_part_bf SEMICOLON
{
  $$ = new Proposition;
  $$->set_type(PropositionType::AWARENESS);
  $$->set_action_name(*$3);
  $$->set_agent(*$1);
  $$->set_observability_conditions(*$4);
};

/* observance condition */
observance:
agent OBSERVES action if_part_bf SEMICOLON
{
  $$ = new Proposition;
  $$->set_type(PropositionType::OBSERVANCE);
  $$->set_action_name(*$3);				
  $$->set_agent(*$1);
  $$->set_observability_conditions(*$4);
};


/* Proposition */
Proposition:
/*static_law {
  $$ = $1;
}
|*/
dynamic_law
{
  $$ = $1;
}
|
executability
{
  $$ = $1;
}
|
/*impossibility
{
  $$ = $1;
}
|*/
determine
{
  $$ = $1;
}
|
announcement
{
  $$ = $1;
}
|
observance
{
  $$ = $1;
}
|
awareness
{
  $$ = $1;
}
;

/* domain */
domain:
/* empty */
{
  $$ = new PropositionsList;
}
| domain Proposition
{
  $$ = $1;
  $1->push_back(*$2);
}
;


/* init */
init:
INIT BeliefFormulaParsed SEMICOLON
{
  $$ = $2;
};

init_spec:
/* empty */
{
  $$ = new ParsedFormulaeList;
  //$$->insert(bf());
  //$$ = new StringSetsSet;
  //$$->insert(StringsSet());
}
| init_spec init
{
  $$ = $1;
  $$->push_back(*$2);
};

/* goal */
/*DEBUG_WARNING_REMOVAL gd_formula:
literal {
  StringsSet s1;

  $$ = new StringSetsSet;

  s1.insert(*$1);
  $$->insert(s1);
}
| gd_formula COMMA gd_formula
{
  $$ = $1;
  $$->insert($3->begin(),$3->end());  
}
| 
gd_formula OR gd_formula {
  StringSetsSet::iterator it1;
  StringSetsSet::iterator it2;
  StringsSet ns;

  $$ = new StringSetsSet;

  for (it2 = $1->begin(); it2 != $1->end(); it2++) {
    for (it1 = $3->begin(); it1 != $3->end(); it1++){
      if (is_consistent(*it1,*it2)) {
	ns = *it2;
	ns.insert(it1->begin(),it1->end());
	$$->insert(ns);
      }
    }
  }  
}
| LEFT_PAREN gd_formula RIGHT_PAREN
{
  $$ = $2;
}; DEBUG_WARNING_REMOVAL*/

goal:
GOAL BeliefFormulaParsed SEMICOLON
{
  $$ = $2;
};

goal_spec:
/* empty */
{
  $$ = new ParsedFormulaeList;
}
| goal_spec goal
{
  $$ = $1;
  $$->push_back(*$2);
};
%%

int yyerror(std::string s)
{
  extern int yylineno;	// defined and maintained in lex.c
  extern char *yytext;	// defined and maintained in lex.c
  
  std::cerr << "ERROR: " << s << " at symbol \"" << yytext;
  std::cerr << "\" on line " << yylineno << std::endl;
  exit(1);
  return 0;
}

int yyerror(const char *s)
{
  return yyerror(std::string(s));
}

bool is_consistent(StringsSet sl1, StringsSet sl2)
{
  StringsSet::const_iterator it;
  std::string nl;

  for (it = sl2.begin(); it != sl2.end(); it++) {
	nl = get_negation(&(*it));
	if (sl1.find(nl) != sl1.end())
	  return false;
  }

  return true;
}

std::string get_negation(const std::string* s)
{
  std::string ns;

  if (s->substr(0,1) == NEGATION_SYMBOL) {
	return s->substr(1);
  }
  ns = NEGATION_SYMBOL;
  return ns.append(*s);
}

/*
StringSetsSet get_negateFluentForm(StringSetsSet input){
  
  StringSetsSet separate;
  StringSetsSet join;
  StringSetsSet::iterator it1;
  StringSetsSet::iterator it3;
  StringSetsSet negation;
  std::string temp;
  StringsSet::const_iterator it2;

  for(it1 = input.begin(); it1 != input.end(); it1++){
     if(it1->begin() == it1->end())
        join.insert(*it1);
     else
        separate.insert(*it1);
  }//for loop

  //Separate elements in separate
     for(it1 = separate.begin(); it1 != separate.end(); it1++){
        temp = get_negation(&(*(it1->begin())));    //possible pointer problem
        StringsSet tiep;
	tiep.insert(temp);
	negation.insert(tiep);
     }//for loop
  

  //Join elements in join with all elements in separate
  for(it3 = negation.begin(); it3 != negation.end(); it3++)
     for(it1 = join.begin(); it1 != join.end(); it1++)
        for(it2 = it1->begin(); it2 != it1->end(); it2++)
        {
           temp = get_negation(&(*it2));    //possible pointer problem
           StringsSet tiep;
           tiep.insert(temp);
           negation.insert(tiep);
	}
  
  return negation;
}
*/

//negate_or: input: String list = list of or. 
//             output: Stringlist 2 = list of and of negation

StringSetsSet negate_or(StringsSet input){
   
   StringsSet::iterator it;
   StringSetsSet output;
   std::string element;
   
   for(it = input.begin(); it != input.end(); it++){
      StringsSet temp;
      element = get_negation(&(*it));
      temp.insert(element);
      output.insert(temp);
   }
   //print_string_set_set(output);
   return output;
}


// or_2_stringlist2

//negate_and : input: std::stringlist2 = list of and of or
//		negate_or(each member of input) = a std::stringlist 2
//                -> n std::stringlist 2 -> std::stringlist 3
//                output = first member stirnglist 3 or second member of std::stringlist 3

StringSetsSet join_SL2(StringSetsSet input1, StringSetsSet input2){
  
  if(input2.size() == 0){
     return input1;
  }

  StringSetsSet::iterator it1;
  StringSetsSet::iterator it2;
  StringsSet ns;

  StringSetsSet output;

  for (it2 = input1.begin(); it2 != input1.end(); it2++) {
    for (it1 = input2.begin(); it1 != input2.end(); it1++){
      if (is_consistent(*it1,*it2)) {
	ns = *it2;
	ns.insert(it1->begin(),it1->end());
	output.insert(ns);
      }
    }
  }  
 
  return output;
   
}

StringSetsSet negate_form(StringSetsSet input){
   
  typedef std::set<StringSetsSet> string_set3;
  string_set3 list3;
  StringSetsSet::iterator it1;
  StringSetsSet::iterator it2;
  string_set3::iterator it3;
  StringsSet ns;
  StringSetsSet temp;

  StringSetsSet output;

  //turn all the otr statements to and statements
   for(it1 = input.begin(); it1 != input.end(); it1++){
      temp = negate_or(*it1);
      list3.insert(temp);
   }

   
   output = *list3.begin();
   for(it3 = ++list3.begin(); it3 != list3.end(); it3++){
      output = join_SL2(output, *it3); 
   }

   
   return output;
}

void print_string_set(StringsSet in){
	StringsSet::iterator it1;
	std::cout << "[ " ;
        for(it1 = in.begin();it1!=in.end();it1++){
		std::cout << *it1 << " , ";   
	}
	std::cout << "] " ;
}

void print_string_set_set(StringSetsSet in){
	StringSetsSet::iterator it1;
	std::cout << "[ "; 
        for(it1 = in.begin();it1!=in.end();it1++){
 		 
		print_string_set(*it1);
		std::cout << " , ";   
	}
	std::cout << " ] " ;
}

//Planning as Logic
