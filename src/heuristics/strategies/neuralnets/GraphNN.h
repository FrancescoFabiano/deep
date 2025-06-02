#pragma once
#include <string>
#include "State.h"

/**
 * \class GraphNN
 * \brief Singleton class for Graph Neural Network-based heuristic evaluation.
 *
 * This class provides an interface for evaluating states using a neural network-based heuristic.
 * It is implemented as a singleton, ensuring only one instance exists during the application's lifetime.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date June 2, 2025
 */
template <StateRepresentation StateRepr>
class GraphNN
{
public:
    /**
     * \brief Get the singleton instance of GraphNN.
     * \return Reference to the singleton instance.
     */
    static GraphNN& get_instance();

    /**
     * \brief Create the singleton instance of GraphNN.
     */

    static void create_instance();


    /**
     * \brief Get the score for a given state using the neural network heuristic.
     * \tparam StateRepr The state representation type.
     * \param state The state to evaluate.
     * \return The heuristic score for the state.
     */
    [[nodiscard]] unsigned short get_score(const State<StateRepr>& state);

    /** \brief Deleted copy constructor (singleton pattern). */
    GraphNN(const GraphNN&) = delete;
    /** \brief Deleted copy assignment operator (singleton pattern). */
    GraphNN& operator=(const GraphNN&) = delete;
    /** \brief Deleted move constructor (singleton pattern). */
    GraphNN(GraphNN&&) = delete;
    /** \brief Deleted move assignment operator (singleton pattern). */
    GraphNN& operator=(GraphNN&&) = delete;

private:
    /**
     * \brief Private constructor for singleton pattern.
     */
    GraphNN();

    static GraphNN* instance; ///< Singleton instance pointer

    std::string m_checking_file_path; ///< Path to the file where the state is printed
};

#include "GraphNN.tpp"
