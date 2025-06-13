/**
* \class IterativeDepthFirst
 * \brief Implements the Iterative Depth First Search strategy to explore the search space.
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
 * \brief IterativeDepthFirst search strategy for use with SpaceSearcher.
 * \tparam StateRepr The state representation type (must satisfy StateRepresentation).
 */
template <StateRepresentation StateRepr>
class IterativeDepthFirst
{
public:
    /**
     * \brief Default constructor.
     */
    explicit IterativeDepthFirst(const State<StateRepr>& initial_state)
    {
        m_initial_state = initial_state;
    }

    /**
     * \brief Push a state into the search container.
     */
    void push(const State<StateRepr>& s)
    {
        if (s.get_plan_length() <= max_depth)
        {
            search_space.push(s);
        }
        else
        {
            m_reached_max_depth = true;
        }
    }

    /**
     * \brief Pop a state from the search container.
     */
    void pop()
    {

        search_space.pop();
        if (search_space.empty() && m_reached_max_depth) //The first state pushed is the initial state, we just need to make sure it is always there extending the depth
        {
            search_space.push(m_initial_state);
            m_reached_max_depth = false; // Reset the flag when we pop the last state
            max_depth += iterative_step; // Increase the depth for the next iteration
        }
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
    std::string m_name = "Iterative Depth First Search"; ///< Name of the search strategy.
    State<StateRepr> m_initial_state; ///< Initial state of the search used when we reset the search space.
    short iterative_step = 1; ///< Iterative step for the search strategy, used to control the increase in depth of the search.
    short max_depth = 2; ///< Maximum depth of the search, used to control the maximum depth of the search. This will be increased at the beginning.
    bool m_reached_max_depth = true; ///< Flag to indicate if the maximum depth has been reached. Set to true to account for the initial state being popped first.
};