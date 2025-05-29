/**
 * \class BreadthFirst
 * \brief Implements the Breadth First Search strategy to explore the search space.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 29, 2025
 */

#pragma once
#include <queue>
#include <string>
#include <chrono>
#include "states/State.h"

/**
 * \brief BreadthFirst search strategy for use with SpaceSearcher.
 * \tparam State The state representation type (must satisfy StateRepresentation).
 */
template <StateRepresentation State>
class BreadthFirst {
public:

    /**
     * \brief Default constructor.
     */
    BreadthFirst() = default;

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
        return search_space.front();
    }

    /**
     * \brief Get the name of the search strategy.
     */
    static std::string get_name()
    {
        return "Breadth First Search";
    }

    /**
     * \brief Reset the search container.
     */
    void reset() {
        search_space = std::queue<State>();
    }

    /**
     * \brief Check if the search container is empty.
     */
    [[nodiscard]] bool empty() const {
        return search_space.empty();
    }

private:
    std::queue<State> search_space;
};

/*
Example usage with SpaceSearcher:

#include "search/SpaceSearcher.h"
#include "search/search_strategies/BreadthFirst.h"

// Create the strategy and searcher
BreadthFirst<MyState> bfs_strategy;
SpaceSearcher<MyState, BreadthFirst<MyState>> searcher("BFS", bfs_strategy);/*
