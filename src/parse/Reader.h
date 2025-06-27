/**
 * \class Reader
 * \brief Stores the domain information read by the parsing process.
 *
 * \details This class is used during the parsing process to store all the read
 * information, i.e., the complete description of the domain (as std::string) is
 * found here. All the domain information stored are yet to be grounded (\ref
 * grounder).
 *
 * \todo The fields of the class are public but should be private and accessed
 * through getters and setters. \bug the () extra causes weird errors \bug the -
 * before B is not accepted \bug the - before fluent_formula is not accepted
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano.
 * \date May 17, 2025
 */

#pragma once

#include "actions/Proposition.h"
#include "utilities/Define.h"

class Reader {
public:
  /// \name Domain Data
  ///@{
  /** \brief Names of all the fluents (only positive) in the domain. */
  StringsSet m_fluents;
  /** \brief Names of all the actions in the domain. */
  StringsSet m_actions;
  /** \brief String descriptions of all the agents in the domain. */
  StringsSet m_agents;
  /** \brief String descriptions of all the initial conditions (initially in the
   * domain). */
  StringSetsSet m_initially;
  /** \brief String descriptions of all the goal conditions. */
  StringsSet m_goal;
  /** \brief \ref m_initially conditions described as \ref BeliefFormulaParsed
   * (yet to ground). */
  ParsedFormulaeList m_bf_initially;
  /** \brief \ref m_goal conditions described as \ref BeliefFormulaParsed (yet
   * to ground). */
  ParsedFormulaeList m_bf_goal;
  /** \brief String description of propositions, each one of these specifies an
   * action condition (yet to ground). */
  PropositionsList m_propositions;
  ///@}

  /**
   * \brief Default constructor.
   */
  Reader();

  /**
   * \brief Reads the info from the domain file.
   * \details Called to parse the file containing the domain and store the
   * information into the fields of the reader class.
   */
  void read();

  /**
   * \brief Print all the information stored inside the reader object.
   */
  void print() const;
};

// Global declaration so it accessed by the bison parser
/// \todo Change this when new parser is added
inline auto domain_reader = std::make_unique<Reader>();