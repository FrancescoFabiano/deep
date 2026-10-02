#include "SignatureBisimulation.h"

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

void SignatureBisimulation::contract(KripkeState &kstate) {
  const auto &worlds = kstate.get_worlds();
  const std::size_t n = worlds.size();
  if (n <= 1) {
    return;
  }

  const std::vector<KripkeWorldPointer> world_of(worlds.begin(), worlds.end());
  std::map<KripkeWorldPointer, int> index;
  for (std::size_t i = 0; i < n; ++i) {
    index.emplace(world_of[i], static_cast<int>(i));
  }

  // Successors of each world as (agent id, world index) pairs.
  std::map<Agent, int> agent_id;
  std::vector<Agent> agent_of;
  std::vector<std::vector<std::pair<int, int>>> successors(n);
  for (const auto &[source, by_agent] : kstate.get_beliefs()) {
    const auto source_it = index.find(source);
    if (source_it == index.end()) {
      continue;
    }
    for (const auto &[agent, targets] : by_agent) {
      const auto [agent_it, added] =
          agent_id.emplace(agent, static_cast<int>(agent_of.size()));
      if (added) {
        agent_of.push_back(agent);
      }
      for (const auto &target : targets) {
        if (const auto target_it = index.find(target);
            target_it != index.end()) {
          successors[source_it->second].emplace_back(agent_it->second,
                                                     target_it->second);
        }
      }
    }
  }

  // Initial partition: same valuation (the canonical world object) and same
  // designatedness.
  const auto &designated = kstate.get_designated_worlds();
  std::vector<int> block(n);
  std::size_t blocks = 0;
  {
    std::map<std::pair<const KripkeWorld *, bool>, int> ids;
    for (std::size_t i = 0; i < n; ++i) {
      const auto key = std::make_pair(world_of[i].get_ptr().get(),
                                      designated.contains(world_of[i]));
      block[i] = ids.emplace(key, static_cast<int>(ids.size())).first->second;
    }
    blocks = ids.size();
  }

  // Refine by (block, set of (agent, successor block)) until stable. Each
  // round refines the previous partition, so an unchanged block count means
  // an unchanged partition.
  std::vector<int> next(n);
  std::vector<std::pair<int, int>> signature;
  while (blocks < n) {
    std::map<std::pair<int, std::vector<std::pair<int, int>>>, int> ids;
    for (std::size_t i = 0; i < n; ++i) {
      signature.clear();
      for (const auto &[agent, target] : successors[i]) {
        signature.emplace_back(agent, block[target]);
      }
      std::ranges::sort(signature);
      signature.erase(std::ranges::unique(signature).begin(), signature.end());
      next[i] = ids.emplace(std::make_pair(block[i], signature),
                            static_cast<int>(ids.size()))
                    .first->second;
    }
    block.swap(next);
    if (ids.size() == blocks) {
      break;
    }
    blocks = ids.size();
  }

  if (blocks == n) {
    return; // already minimal
  }

  // Quotient: the first world of each block represents it; bisimilar worlds
  // have the same successor blocks, so the representative's edges suffice.
  std::vector<int> representative(blocks, -1);
  for (std::size_t i = 0; i < n; ++i) {
    if (representative[block[i]] < 0) {
      representative[block[i]] = static_cast<int>(i);
    }
  }

  KripkeWorldPointersSet new_worlds;
  KripkeWorldPointersSet new_designated;
  KripkeWorldPointersTransitiveMap new_beliefs;
  for (const int r : representative) {
    const auto &world = world_of[r];
    new_worlds.insert(world);
    if (designated.contains(world)) {
      new_designated.insert(world);
    }
    for (const auto &[agent, target] : successors[r]) {
      new_beliefs[world][agent_of[agent]].insert(
          world_of[representative[block[target]]]);
    }
  }

  kstate.set_beliefs(new_beliefs);
  kstate.set_worlds(new_worlds);
  kstate.set_designated_worlds(new_designated);
}
