#include "KripkeReachabilityHelper.h"
#include "BeliefFormula.h"
#include "KripkeState.h"
#include "KripkeWorld.h"
#include "utilities/SetHelper.h"
#include <utility>

#include "Domain.h"

/**
 * \file KripkeReachabilityHelper.cpp
 * \brief Implementation of KripkeReachabilityHelper.h
 * \copyright GNU Public License.
 * \author Francesco Fabiano
 * \date May 2025
 */

KripkeWorldPointersSet KripkeReachabilityHelper::get_B_reachable_worlds(
    const Agent& ag, const KripkeWorldPointer& world, const KripkeState& kstate)
{
    KripkeWorldPointersSet ret;
    const auto& beliefs = kstate.get_beliefs();
    if (const auto pw_map = beliefs.find(world); pw_map != beliefs.end())
    {
        if (const auto pw_set = pw_map->second.find(ag); pw_set != pw_map->second.end())
        {
            SetHelper::sum_set<KripkeWorldPointer>(ret, pw_set->second);
        }
    }
    return ret;
}

bool KripkeReachabilityHelper::get_B_reachable_worlds_recursive(const Agent& ag, const KripkeWorldPointer& world,
                                                                KripkeWorldPointersSet& reached,
                                                                const KripkeState& kstate)
{
    const auto& beliefs = kstate.get_beliefs();
    if (const auto pw_map = beliefs.find(world); pw_map != beliefs.end())
    {
        if (const auto pw_set = pw_map->second.find(ag); pw_set != pw_map->second.end())
        {
            auto previous_size = reached.size();
            SetHelper::sum_set<KripkeWorldPointer>(reached, pw_set->second);
            return previous_size == reached.size();
        }
    }
    return true;
}

KripkeWorldPointersSet KripkeReachabilityHelper::get_E_reachable_worlds(
    const AgentsSet& ags, const KripkeWorldPointer& world, const KripkeState& kstate)
{
    KripkeWorldPointersSet ret;
    for (const auto& agent : ags)
    {
        SetHelper::sum_set<KripkeWorldPointer>(ret, get_B_reachable_worlds(agent, world, kstate));
    }
    return ret;
}

bool KripkeReachabilityHelper::get_E_reachable_worlds_recursive(const AgentsSet& ags,
                                                                const KripkeWorldPointersSet& worlds,
                                                                KripkeWorldPointersSet& reached,
                                                                const KripkeState& kstate)
{
    bool is_fixed_point = true;
    for (const auto& world : worlds)
    {
        for (const auto& agent : ags)
        {
            if (!get_B_reachable_worlds_recursive(agent, world, reached, kstate))
            {
                is_fixed_point = false;
            }
        }
    }
    return is_fixed_point;
}

KripkeWorldPointersSet KripkeReachabilityHelper::get_C_reachable_worlds(
    const AgentsSet& ags, const KripkeWorldPointer& world, const KripkeState& kstate)
{
    bool is_fixed_point = false;
    KripkeWorldPointersSet newly_reached = get_E_reachable_worlds(ags, world, kstate);
    KripkeWorldPointersSet already_reached;
    KripkeWorldPointersSet ret;
    while (!is_fixed_point)
    {
        SetHelper::sum_set<KripkeWorldPointer>(newly_reached, ret);
        SetHelper::minus_set<KripkeWorldPointer>(newly_reached, already_reached);
        is_fixed_point = get_E_reachable_worlds_recursive(ags, newly_reached, ret, kstate);
        already_reached = newly_reached;
    }
    return ret;
}


void KripkeReachabilityHelper::get_all_reachable_worlds(const KripkeWorldPointer& world,
                                                        KripkeWorldPointersSet& reached_worlds,
                                                        KripkeWorldPointersTransitiveMap& reached_edges,
                                                        const KripkeState& kstate)
{
    const auto& agents = Domain::get_instance().get_agents();

    for (const auto& agent : agents)
    {
        KripkeWorldPointersSet pw_list;

        if (auto world_beliefs_it = kstate.get_beliefs().find(world); world_beliefs_it != kstate.get_beliefs().end())
        {
            const auto& agent_beliefs_map = world_beliefs_it->second;
            if (auto agent_beliefs_it = agent_beliefs_map.find(agent); agent_beliefs_it != agent_beliefs_map.end())
            {
                pw_list = agent_beliefs_it->second;
            }
        }

        for (const auto& reachable_world : pw_list)
        {
            if (reached_worlds.insert(reachable_world).second)
            {
                get_all_reachable_worlds(reachable_world, reached_worlds, reached_edges, kstate);

                if (auto beliefs_it = kstate.get_beliefs().find(reachable_world); beliefs_it != kstate.get_beliefs().
                    end())
                {
                    reached_edges.emplace(reachable_world, beliefs_it->second);
                }
            }
        }
    }
}

void KripkeReachabilityHelper::clean_unreachable_worlds(KripkeState& kstate)
{
    KripkeWorldPointersSet reached_worlds;
    KripkeWorldPointersTransitiveMap reached_edges;

    const auto& pointed_world = kstate.get_pointed();
    reached_worlds.insert(pointed_world);

    const auto& beliefs = kstate.get_beliefs();
    if (const auto it = beliefs.find(pointed_world); it != beliefs.end())
    {
        reached_edges.emplace(pointed_world, it->second);
    }

    get_all_reachable_worlds(pointed_world, reached_worlds, reached_edges, kstate);

    kstate.set_worlds(reached_worlds);
    kstate.set_beliefs(reached_edges);
}
