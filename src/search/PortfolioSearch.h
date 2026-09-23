/**
 * \class PortfolioSearch
 * \brief Class that implements the portfolio-method search (multiple
 * configurations).
 *
 * \details This class manages and executes multiple search configurations in
 * parallel or sequentially. Each configuration can specify different search
 * strategies and heuristics. The search stops as soon as a plan is found by any
 * configuration.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano.
 * \date May 30, 2025
 */
#pragma once

#include <map>
#include <ostream>
#include <string>
#include <vector>

/**
 * \brief PortfolioSearch manages and executes multiple search configurations.
 */
class PortfolioSearch {
public:
  /// \name Constructors & Destructor
  ///@{
  PortfolioSearch();
  ///@}

  /// \name Main Methods
  ///@{
  /**
   * \brief Launches the portfolio-method search (multiple configurations).
   *
   * \return true if a plan was found, false otherwise.
   *
   */
  [[nodiscard]]
  bool run_portfolio_search() const;

  /**
   * \brief Parses configurations from a file.
   *
   * The file should contain configurations in the format:
   * key1=value1,key2=value2,...
   * Each non-empty, non-comment line represents a different configuration.
   * Lines starting with '#' are ignored.
   * \warning This performs only minimal validation, so configuration lines
   * should still be kept well-formed.
   *
   * \param file_path Path to the configuration file.
   */
  void parse_configurations_from_file(const std::string &file_path);

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
