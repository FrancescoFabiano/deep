/**
* \class DepthFirst
 * \brief Implements the Depth First Search strategy to explore the search space.
 *
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 29, 2025
 */

#pragma once
#include "states/State.h"
#include <stack>
#include <string>


/**
 * \brief DepthFirst search strategy for use with SpaceSearcher.
 * \tparam StateRepr The state representation type (must satisfy StateRepresentation).
 */
template <StateRepresentation StateRepr>
class DepthFirst
{
public:
    /**
     * \brief Default constructor.
     */
    DepthFirst() = default;

    /**
     * \brief Push a state into the search container.
     */
    void push(const State<StateRepr>& s)
    {
        search_space.push(s);
    }

    /**
     * \brief Pop a state from the search container.
     */
    void pop()
    {
        search_space.pop();
    }

    /**
     * \brief Peek at the next state in the search container.
     */
    State<StateRepr> peek() const
    {
        return search_space.top();
    }

    /**
     * \brief Get the name of the search strategy.
     */
    std::string get_name() const
    {
        return m_name;
    }

    /**
     * \brief Reset the search container.
     */
    void reset()
    {
        search_space = std::stack<State<StateRepr>>();
    }

    /**
     * \brief Check if the search container is empty.
     */
    [[nodiscard]] bool empty() const
    {
        return search_space.empty();
    }

private:
    std::stack<State<StateRepr>> search_space;
    std::string m_name = "Depth First Search"; ///< Name of the search strategy.
};


/*
// For queue (BFS)
std::queue<State> q;
auto push_q = [](auto& c, const State& s) { c.push(s); };
auto pop_q  = [](auto& c) { c.pop(); };
auto peek_q = [](auto& c) -> State { return c.front(); };
search_sequential_generic(initial, actions, check_visited, bisimulation_reduction, q, push_q, pop_q, peek_q);

// For stack (DFS)
std::stack<State> s;
auto push_s = [](auto& c, const State& s) { c.push(s); };
auto pop_s  = [](auto& c) { c.pop(); };
auto peek_s = [](auto& c) -> State { return c.top(); };
search_sequential_generic(initial, actions, check_visited, bisimulation_reduction, s, push_s, pop_s, peek_s);

// For priority_queue (Best-First)
std::priority_queue<State, std::vector<State>, StateComparator<State>> pq;
auto push_pq = [](auto& c, const State& s) { c.push(s); };
auto pop_pq  = [](auto& c) { c.pop(); };
auto peek_pq = [](auto& c) -> State { return c.top(); };
search_sequential_generic(initial, actions, check_visited, bisimulation_reduction, pq, push_pq, pop_pq, peek_pq);*/
