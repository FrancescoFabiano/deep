//
// Created by fraano on 30/05/2025.
//

#include "PortfolioSearch.h"
#include <fstream>
#include <sstream>
#include <thread>
#include <atomic>
#include <functional>
#include "../utilities/ExitHandler.h"
#include "search/SpaceSearcher.h"
#include "domain/Domain.h"


bool PortfolioSearch::run_portfolio_search(int user_threads) const {
    /*To fix*/
}

void PortfolioSearch::parse_configurations_from_file(const std::string& file_path) {
    m_search_configurations.clear();
    std::ifstream infile(file_path);
    if (!infile) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::PortfolioConfigFileError,
            "[PortfolioSearch] Could not open configuration file: " + file_path
        );
        // No return needed, exit_with_message will terminate.
    }
    std::string line;
    while (std::getline(infile, line)) {
        std::map<std::string, std::string> config;
        std::istringstream iss(line);
        std::string token;
        while (std::getline(iss, token, ',')) {
            if (auto pos = token.find('='); pos != std::string::npos) {
                std::string key = token.substr(0, pos);
                std::string value = token.substr(pos + 1);
                config[key] = value;
            }
        }
        if (!config.empty()) {
            m_search_configurations.push_back(config);
        }
    }
}

void PortfolioSearch::set_default_configurations() {
    m_search_configurations.clear();
    // Whatever is not set here will is kept from the user input.
    m_search_configurations.push_back({{"search", "BFS"}});
    m_search_configurations.push_back({{"search", "HFS"}, {"heuristic", "SUBGOALS"}});
    m_search_configurations.push_back({{"search", "HFS"}, {"heuristic", "L_PG"}});
    m_search_configurations.push_back({{"search", "HFS"}, {"heuristic", "S_PG"}});
    m_search_configurations.push_back({{"search", "HFS"}, {"heuristic", "C_PG"}});
    m_search_configurations.push_back({{"search", "HFS"}, {"heuristic", "GNN"}});
    m_search_configurations.push_back({{"strategy", "DFS"}});
}
