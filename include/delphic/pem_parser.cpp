/*
 * \brief The main file.
 *
 * \copyright GNU Public License.
 *
 * \author Francesco Fabiano.
 * \date Aprile 1, 2019
 */

#include "pem_parser.h"
#include "event.h"
#include "pem_store.h"

void pem_parser::apply_spaces_regex(std::string & to_clean, const std::regex & pattern)
{

	std::string tmp_cleaned;

	while (std::regex_match(to_clean, pattern)) {
		tmp_cleaned = std::regex_replace(to_clean, pattern, "$1");
		tmp_cleaned += std::regex_replace(to_clean, pattern, "$3");
		to_clean = tmp_cleaned;
	}
}

void pem_parser::parse_conditions(const std::string & line, event_metacond & cond, int line_count)
{

	std::regex spaces_opened("(.*[\\w\\(\\$])(\\s+)([\\w\\(\\$].*)");
	std::regex spaces_closed("(.*[\\w\\)\\$])(\\s+)([\\w\\)\\$].*)");
	std::regex negation("not\\(([^\\s]+)\\)");

	//std::cout << std::endl << "\nOriginal string is " << line << std::endl;

	std::string cleaned = line;

	apply_spaces_regex(cleaned, spaces_opened);
	apply_spaces_regex(cleaned, spaces_closed);

	boost::replace_all(cleaned, ")(", ") (");


	std::istringstream args(cleaned);
	std::string sub_arg;

	bool is_negated;


	while (args >> sub_arg) {
		is_negated = false;
		sub_arg = std::regex_replace(sub_arg, std::regex("\\(([^\\s]+)\\)"), "$1");
		if (std::regex_match(sub_arg, negation)) {
			is_negated = true;
			sub_arg = std::regex_replace(sub_arg, negation, "$1");
		}

		if (sub_arg.compare("$act_pre$") == 0) {

			if (is_negated) {
				cond.insert(neg_act_pre);
			} else {
				cond.insert(act_pre);
			}


		} else if (sub_arg.compare("$act_eff$") == 0) {
			if (is_negated) {
				cond.insert(neg_act_eff);
			} else {
				cond.insert(act_eff);
			}
		} else if (sub_arg.compare("none") == 0) {
			cond.clear();
			cond.insert(none);
			return;
			//std::cout << "\n\t\t" << sub_arg;
		} else {
			//Here parse of normal fluent/Belief formulae to add as extra field in the event
			std::cerr << "\nParsing Error at Line " << line_count << ": The specified condition has no meaning. Please use \'act_eff\', \'act_pre\', or \'none\'.\n";
			exit(1);
		}
	}
}

void pem_parser::parse_ag_list(const std::string & line)
{

	std::string cleaned = line;

	boost::replace_all(cleaned, ",", ";");

	std::regex spaces_int("(.*\\;)(\\s+)(\\w.*)");
	std::regex spaces_ext("(.*\\w)(\\s+)(\\;.*)");

	//std::cout << std::endl << "\nOriginal string is " << line << std::endl;


	apply_spaces_regex(cleaned, spaces_int);
	apply_spaces_regex(cleaned, spaces_ext);

	boost::replace_all(cleaned, ";", " ");
	boost::replace_all(cleaned, ")(", ") (");



	std::istringstream args(cleaned);
	std::string sub_arg;


	while (args >> sub_arg) {
		pem_store::get_instance().add_agent_group(sub_arg);
	}

}

void pem_parser::parse_edge(const std::string & edge, pem_edges & edges)
{
	std::string cleaned = edge;

	boost::replace_all(cleaned, ";", " ");

	std::istringstream args(cleaned);
	std::string sub_arg;

	args >> sub_arg;
	event_ptr first = pem_store::get_instance().get_event(boost::lexical_cast<event_id>(sub_arg));

	args >> sub_arg;
    event_ptr second = pem_store::get_instance().get_event(boost::lexical_cast<event_id>(sub_arg));

	args >> sub_arg;
    agent_group e_to_add_ag = pem_store::get_instance().get_agent_group(sub_arg);

    auto it_eve = edges.find(first);

    if (it_eve != edges.end()) {
        event_map e_map = it_eve->second;
        auto it_evm = e_map.find(e_to_add_ag);

        if (it_evm != e_map.end()) {
            it_evm->second.insert(second);
        } else {
            e_map.insert(event_map::value_type(e_to_add_ag, {second}));
        }
    } else {
        event_map tmp;
        tmp.insert(event_map::value_type(e_to_add_ag, {second}));
        edges.insert(pem_edges ::value_type(first, tmp));
    }
//	if (edges.find(e_to_add_ag) != edges.end()) {
//		edges[e_to_add_ag].insert(e_to_add);
//	} else {
//		std::set<pem_edge> tmp_set;
//		tmp_set.insert(e_to_add);
//		edges.insert(std::pair<agent_group, std::set<pem_edge> >(e_to_add_ag, tmp_set));
//	}
}

void pem_parser::parse_edges_list(const std::string & line, pem_edges & edges)
{
	std::string cleaned = line;

	boost::replace_all(cleaned, ",", ";");

	std::regex spaces_opened("(.*[\\w\\(\\;])(\\s+)([\\w\\(\\;].*)");
	std::regex spaces_closed("(.*[\\w\\)\\;])(\\s+)([\\w\\)\\;].*)");

	//std::cout << std::endl << "\nOriginal string is " << line << std::endl;


	apply_spaces_regex(cleaned, spaces_opened);
	apply_spaces_regex(cleaned, spaces_closed);
	boost::replace_all(cleaned, ")(", ") (");


	//boost::replace_all(cleaned, ";", " ");


	std::istringstream args(cleaned);
	std::string sub_arg;

	while (args >> sub_arg) {
		sub_arg = std::regex_replace(sub_arg, std::regex("\\(([^\\s]+)\\)"), "$1");
		parse_edge(sub_arg, edges);
	}
}

void pem_parser::parse()
{
	std::string filename = "include/delphic/marho_pem.txt";
	std::cout << "\nTesting the generation of Event Models starting from the file: " << filename << std::endl;
	std::ifstream pem_file(filename);


	if (pem_file.is_open()) {

		bool in_event = false, in_model = false, not_assigned = true;
		std::string line;
		std::string type, field, internal_field;


		std::regex indentation("(\\s+)(.*)");
		std::regex comment("([^*]*)(\\*+)(.*)");

		std::regex type_regex("(\\()(\\s*)([^\\s]+)(\\s*)(\\:)(\\s*)(.*)");
		std::regex field_regex("(\\:)(\\s*)([^\\s]+)(\\s+)([\\(\\{])(.+)([\\)\\}])(.*)");


		event_id e_id;
		event_metacond e_meta_pre;
		event_metacond e_meta_post;
		event e_to_add;

		pem_id p_id;
		event_id p_pointed_id;
		pem_edges p_edges;
		pem p_to_add;


		int line_count = 0;

		while (getline(pem_file, line)) {

			++line_count;

			line = std::regex_replace(line, indentation, "$2");
			line = std::regex_replace(line, comment, "$1");

			if (std::regex_match(line, type_regex)) {

				type = std::regex_replace(line, type_regex, "$3");

				if (!not_assigned) {
					std::cerr << "\nParsing Error at Line " << line_count << ": Before declaring a new " << type << " you need to finish the previous type (Maybe you are missing a parenthesis?).\n";
					exit(1);
				}

				if (type.compare("event") == 0) {
					if (!not_assigned) {
						std::cerr << "\nParsing Error at Line " << line_count << ": Trying to define an \'event\' while already defining an \'event\' or a \'model\'.\n";
					}

					e_id = -1;
					e_meta_pre.clear();
					e_meta_post.clear();

					in_event = true;
					not_assigned = false;

				} else if (type.compare("model") == 0) {

					if (!not_assigned) {
						std::cerr << "\nParsing Error at Line " << line_count << ": Trying to define a \'model\' while already defining an \'event\' or a \'model\'.\n";

					}

					p_id = -1;
					p_pointed_id = -1;
					p_edges.clear();


					in_model = true;
					not_assigned = false;
				} else if (type.compare("obs_groups") == 0) {
					if (!not_assigned) {
						std::cerr << "\nParsing Error at Line " << line_count << ": Trying to define the \'observability groups\' while already defining an \'event\' or a \'model\'.\n";

					}
					std::string obs = std::regex_replace(line, type_regex, "$7");

					parse_ag_list(std::regex_replace(obs, std::regex("\\{(.*)\\}(.*)"), "$1"));

				} else {
					std::cerr << "\nParsing Error at Line " << line_count << ": The specified type -- " << type << " -- is not defined in our syntax. Please use either \'event\', \'model\', or \'obs_groups\'.\n";
					exit(1);
				}

			} else if (std::regex_match(line, std::regex("(\\()(.)+(\\:)(\\s*)(\\S)+"))) {
				std::cerr << "\nParsing Error at Line " << line_count << ": The syntax requires a newline before the definition of a new field.\n";
				exit(1);
			} else if (std::regex_match(line, field_regex)) {

				field = std::regex_replace(line, field_regex, "$3");
				internal_field = std::regex_replace(line, field_regex, "$6");

				if (not_assigned) {
					std::cerr << "\nParsing Error at Line " << line_count << ": Before declaring a new \'" << field << "\' you need to open a new \'event\' or \'model\'.\n";
					exit(1);
				}

				if (field.compare("id") == 0 && (e_id == -1 || p_id == -1)) {

					if (in_event) {
						e_id = boost::lexical_cast<event_id>(internal_field);
					} else if (in_model) {
						p_id = boost::lexical_cast<pem_id>(internal_field);
					} else {
						std::cerr << "\nParsing Error at Line " << line_count << ": Before declaring a new \'" << field << "\' you need to open a new \'event\' or \'model\'.\n";
						exit(1);
					}
				} else if (field.compare("pointed") == 0) {
					if (!in_model) {
						std::cerr << "\nParsing Error at Line " << line_count << ": Pointed event found outside of a model definition.";
						exit(1);
					}
					p_pointed_id = boost::lexical_cast<event_id>(internal_field);
				} else if (field.compare("precondition") == 0) {
					if (!in_event) {
						std::cerr << "\nParsing Error at Line " << line_count << ": Precondition found outside of an event definition.";
						exit(1);
					}
					parse_conditions(internal_field, e_meta_pre, line_count);
				} else if (field.compare("postcondition") == 0) {
					if (!in_event) {
						std::cerr << "\nParsing Error at Line " << line_count << ": Postcondition found outside of an event definition.";
						exit(1);
					}

					parse_conditions(internal_field, e_meta_post, line_count);
				} else if (field.compare("events") == 0) {
					if (!in_model) {
						std::cerr << "\nParsing Error at Line " << line_count << ": Events found outside of a model definition.";
						exit(1);
					}
					//	std::cout << "\n\tWith events: ";
					//	parse_list(internal_field);

					//Add here something as check

				} else if (field.compare("edges") == 0) {
					if (!in_model) {
						std::cerr << "\nParsing Error at Line " << line_count << ": Edges found outside of a model definition.";
						exit(1);
					}
					parse_edges_list(internal_field, p_edges);

					//std::cout << field << " -> " << internal_field << std::endl;
				} else {
					std::cerr << "\nParsing Error at Line " << line_count << ": The specified field -- " << field << " -- is not properly defined in our syntax (watch out for repetition). Please use \'id\', \'precondition\', \'postcondition\', \'events\', or \'edges\'.\n";
					exit(1);
				}
			} else if (std::regex_match(line, std::regex("(\\))(\\s*)"))) {

				if (in_event) {
					e_to_add.set_id(e_id);
					e_to_add.set_meta_precondition(e_meta_pre);
					e_to_add.set_meta_postconditions(e_meta_post);

					pem_store::get_instance().add_event(e_to_add);
				} else if (in_model) {
					p_to_add.set_id(p_id);
					p_to_add.set_pointed_id(p_pointed_id);
					p_to_add.set_edges(p_edges);

					pem_store::get_instance().add_pem(p_to_add);

				}
				in_event = false;
				in_model = false;
				not_assigned = true;

			} else if (line.size() > 0) {
				std::cerr << "\nParsing Error at Line " << line_count << ": the syntax in \'" << line << "\' is not correct.\n";
				exit(1);

			}
		}
	} else {
		std::cerr << "\nParsing Error: It is impossible to open the file \'" << filename << "\'.";
		exit(1);
	}
	pem_file.close();
}

