//
// Created by fraano on 30/05/2025.
//

#pragma once
#include <map>
#include <string>
#include <vector>


class PortfolioSearch {

public:
    /**
     * \brief Launches the portfolio-method search (multiple configurations).
     *
     * For now, uses a fixed configuration: BFS + ALL Heuristics + DFS (depending on the number of threads).
     * \return true if a plan was found, false otherwise.
     */
    [[nodiscard]]
    bool run_portfolio_search(int user_threads = 1) const;

    /**
     * \brief Parses configurations from a file.
     *
     * The file should contain configurations in the format:
     * key1=value1,key2=value2,...
     * Each line represents a different configuration.
     * \warning This does not check for parsing errors, so ensure the file is well-formed.
     *
     * \param file_path Path to the configuration file.
     */
    void parse_configurations_from_file(const std::string& file_path);

    void set_default_configurations();

private:
    std::vector<std::map<std::string, std::string>> m_search_configurations = {}; ///< List of search configurations to try.

};
