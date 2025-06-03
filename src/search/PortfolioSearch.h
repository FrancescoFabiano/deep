/**
 * \class PortfolioSearch
 * \brief Class that implements the portfolio-method search (multiple configurations).
 *
 * \details This class manages and executes multiple search configurations in parallel or sequentially.
 *          Each configuration can specify different search strategies and heuristics.
 *          The search stops as soon as a plan is found by any configuration.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano.
 * \date May 30, 2025
 */
#pragma once

#include <map>
#include <string>
#include <vector>
#include <ostream>

/**
 * \brief PortfolioSearch manages and executes multiple search configurations.
 */
class PortfolioSearch
{
public:
    /// \name Constructors & Destructor
    ///@{
    PortfolioSearch() = default;
    ~PortfolioSearch() = default;
    PortfolioSearch(const PortfolioSearch&) = default;
    PortfolioSearch(PortfolioSearch&&) noexcept = default;
    PortfolioSearch& operator=(const PortfolioSearch&) = default;
    PortfolioSearch& operator=(PortfolioSearch&&) noexcept = default;
    ///@}

    /// \name Main Methods
    ///@{
    /**
     * \brief Launches the portfolio-method search (multiple configurations).
     *
     * \param user_threads The number of threads to use for the search.
     * \param threads_per_search Number of threads to use in each search strategy (default: 1).
     * \return true if a plan was found, false otherwise.
     *
     */
    [[nodiscard]]
    bool run_portfolio_search(int user_threads = 1, int threads_per_search = 1) const;

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

    /**
     * \brief Sets a default set of search configurations.
     *
     * This will overwrite any previously set configurations.
     */
    void set_default_configurations();
    ///@}

private:
    /// \name Data Members
    ///@{
    /**
     * \brief List of search configurations to try.
     * Each configuration is a map from parameter name to value.
     */
    std::vector<std::map<std::string, std::string>> m_search_configurations{};
    ///@}
};
