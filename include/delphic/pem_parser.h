/**
 * \class pem_parser
 * \brief Singleton class that parse the file that described the custom pems.
 * 
 * 
 * @see event, pem, pem_store.
 * 
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date April 28, 2021
 */


#pragma once
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
//#include <bits/stdc++.h>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <regex>
#include "../utilities/define.h"

class pem_parser
{
private:
    static void apply_spaces_regex(std::string & to_clean, const std::regex & pattern);

    static void parse_conditions(const std::string & line, event_metacond & cond, int line_count);

    static void parse_ag_list(const std::string & line);

    static void parse_edge(const std::string & edge, pem_edges & edges);

    static void parse_edges_list(const std::string & line, pem_edges & edges);

public:

    static void parse();

};

