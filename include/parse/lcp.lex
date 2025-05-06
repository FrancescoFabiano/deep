/* Scanner for AL language */

%option nounput

%{
#include "../include/utilities/define.h"
#include "../include/actions/proposition.h"
#include "../include/formulae/belief_formula.h"
#include "../include/domain/attitude.h"

#include "tok.h"

int yyerror(const char* s);
//int yylineno = 1;
%}

digit		[0-9]
number	{digit}+
id [a-zA-Z][a-zA-Z_0-9]*
comment %.*$

%%

{number}	{ yylval.str_val = new std::string(yytext); return NUMBER; }
";"		{ return SEMICOLON; } 
"fluent" {return FLUENT;}
"action" {return ACTION;}
"agent" {return AGENT;}
"if" {return IF;}
"executes" {return AGEXEC;}
"causes" {return CAUSES;}
"executable" {return EXECUTABLE;}
"impossible" {return IMPOSSIBLE;}
"determines" {return DETERMINE;}
"aware_of" {return AWAREOF;}
"observes" {return OBSERVES;}
"announces" {return ANNOUNCES;}
"initially" {return INIT;}
"goal" {return GOAL;}
"B" {return MB;}
"C" {return MC;}
"E" {return ME;}
"D" {return MD;}
"dox_announces" {return LIE;}

"has_attitude" {return ATTITUDES;}
"wrt" {return WRT;}
"f_trusty" {return TRUSTY;}
"f_mistrusty" {return MISTRUSTY;}
"f_untrusty" {return UNTRUSTY;}
"f_stubborn" {return STUBBORN;}
"p_keeper" {return KEEPER;}
"p_insecure" {return INSECURE;}


"[" {return LEFT_BRAC;}
"]" {return RIGHT_BRAC;}
"," {return COMMA;}
"|" {return OR;}
"(" {return LEFT_PAREN;}
")" {return RIGHT_PAREN;}
"-" {yylval.str_val = new std::string(yytext); return NEGATION;}

{id} {yylval.str_val = new std::string(yytext); return ID;}

[ \t]*		{}
[\n]		{ yylineno++;	}
{comment} ;

.		{std::cerr << "SCANNER "; yyerror(""); exit(1);	}

