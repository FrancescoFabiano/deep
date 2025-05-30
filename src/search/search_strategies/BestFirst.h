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

#include "heuristics_manager.h"


/**
 * \brief Compares two states based on their heuristic value.
 *
 * \param state1 The first state to compare.
 * \param state2 The second state to compare.
 * \return true if state1 has a higher heuristic value than state2, false otherwise.
 */
template <StateRepresentation StateRepr>
struct StateComparator
{
    bool operator()(const State<StateRepr>& state1, const State<StateRepr>& state2) const
    {
        return state1.get_heuristic_value() > state2.get_heuristic_value();
    }
};

/**
 * \brief BestFirst search strategy for use with SpaceSearcher.
 * \tparam StateRepr The state representation type (must satisfy StateRepresentation).
 */
template <StateRepresentation StateRepr>
class BestFirst {
public:

    /**
     * \brief Default constructor.
     */
    BestFirst() = default;

    /**
     * \brief Push a state into the search container.
     */
    void push(const State<StateRepr>& s) {
        heuristics_manager.set_heuristic_value(s);
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
    State<StateRepr> peek() const {
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
        search_space = StatePriorityQueue();
    }

    /**
     * \brief Check if the search container is empty.
     */
    [[nodiscard]] bool empty() const {
        return search_space.empty();
    }

private:
    using StatePriorityQueue = std::priority_queue<State<StateRepr>, std::vector<State<StateRepr>>, StateComparator<State<StateRepr>>>;
    StatePriorityQueue search_space;
};
