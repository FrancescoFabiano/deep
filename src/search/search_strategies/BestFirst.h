/**
 * \class BestFirst
 * \brief Implements the Best First Search strategy to explore the search space.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 29, 2025
 */

#pragma once
#include "states/State.h"
#include "utilities/ExitHandler.h"
#include <queue>
#include <string>


/**
 * \brief Compares two states based on their heuristic value.
 *
 * \param state1 The first state to compare.
 * \param state2 The second state to compare.
 * \return true if state1 has a higher heuristic value than state2, false otherwise.
 */
template <StateRepresentation State>
struct StateComparator
{
    bool operator()(const State& state1, const State& state2) const
    {
        return state1.get_heuristic_value() > state2.get_heuristic_value();
    }
};

/**
 * \brief BestFirst search strategy for use with SpaceSearcher.
 * \tparam State The state representation type (must satisfy StateRepresentation).
 */
template <StateRepresentation State>
class BestFirst {
public:

    /**
     * \brief Default constructor.
     */
    BestFirst() = default;

    /**
     * \brief Push a state into the search container.
     */
    void push(const State& s) {
        search_space.push(s);
    }

    /**
     * \brief Pop a state from the search container.
     */
    void pop() {
        search_space.pop();
    }

    /**
     * \brief Peek at the next state in the search container.
     */
    State peek() const {
        return search_space.top();
    }

    /**
     * \brief Get the name of the search strategy.
     */
    static std::string get_name()
    {
        return "Best First Search";
    }

    /**
     * \brief Reset the search container.
     */
    void reset() {
        search_space = std::priority_queue<State>();
    }

    /**
     * \brief Check if the search container is empty.
     */
    bool empty() const {
        return search_space.empty();
    }

private:
    std::priority_queue<State> search_space;
};
