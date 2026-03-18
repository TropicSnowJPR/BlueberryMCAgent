"""trainer/reward.py – shaped reward function for early survival progression.

Rewards are computed from consecutive observation dictionaries so that the
trainer can assign a scalar reward to each transition without requiring the
game to provide an explicit reward signal.
"""

from __future__ import annotations

# Items that matter for each progression milestone.
# Format: item_id (as returned by MineScript) → milestone name
WOOD_ITEMS = {
    "oak_log", "birch_log", "spruce_log", "jungle_log",
    "acacia_log", "dark_oak_log", "mangrove_log", "cherry_log",
}
PLANK_ITEMS = {
    "oak_planks", "birch_planks", "spruce_planks", "jungle_planks",
    "acacia_planks", "dark_oak_planks", "mangrove_planks", "cherry_planks",
}
CRAFTING_TABLE_ITEMS = {"crafting_table"}
PICKAXE_ITEMS = {
    "wooden_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "golden_pickaxe", "diamond_pickaxe", "netherite_pickaxe",
}
COBBLESTONE_ITEMS = {"cobblestone", "cobbled_deepslate"}
FOOD_ITEMS = {
    "apple", "bread", "cooked_beef", "cooked_porkchop",
    "cooked_chicken", "cooked_mutton", "cooked_rabbit",
    "cooked_salmon", "cooked_cod", "baked_potato",
    "carrot", "potato", "golden_apple", "golden_carrot",
}

# Reward magnitudes
R_LOG_PER_ITEM = 0.5
R_PLANK_FIRST = 1.0          # one-time bonus for crafting first plank
R_CRAFTING_TABLE_FIRST = 2.0
R_PICKAXE_FIRST = 3.0
R_COBBLESTONE_PER_ITEM = 0.3
R_FOOD_PER_ITEM = 0.4
R_LIVING_BONUS = 0.0001      # tiny reward just for surviving each step
R_DAMAGE_PENALTY = -1.0      # per health point lost
R_DEATH_PENALTY = -10.0


def _count_items(inventory: dict, item_set: set) -> int:
    return sum(inventory.get(item, 0) for item in item_set)


def compute_reward(
    obs_prev: dict,
    obs_curr: dict,
    milestones: dict,
) -> tuple[float, dict]:
    """Compute a scalar reward from consecutive observations.

    Parameters
    ----------
    obs_prev:   observation at time t-1
    obs_curr:   observation at time t
    milestones: mutable dict persisted across steps, tracks one-time bonuses

    Returns
    -------
    (reward, updated_milestones)
    """
    reward = R_LIVING_BONUS

    inv_prev = obs_prev.get("inventory", {})
    inv_curr = obs_curr.get("inventory", {})

    # ── Wood logs ──────────────────────────────────────────────────────────────
    wood_prev = _count_items(inv_prev, WOOD_ITEMS)
    wood_curr = _count_items(inv_curr, WOOD_ITEMS)
    if wood_curr > wood_prev:
        reward += (wood_curr - wood_prev) * R_LOG_PER_ITEM

    # ── Planks (first time any plank appears) ──────────────────────────────────
    planks_curr = _count_items(inv_curr, PLANK_ITEMS)
    if not milestones.get("planks_unlocked") and planks_curr > 0:
        reward += R_PLANK_FIRST
        milestones["planks_unlocked"] = True

    # ── Crafting table ─────────────────────────────────────────────────────────
    ct_curr = _count_items(inv_curr, CRAFTING_TABLE_ITEMS)
    if not milestones.get("crafting_table_unlocked") and ct_curr > 0:
        reward += R_CRAFTING_TABLE_FIRST
        milestones["crafting_table_unlocked"] = True

    # ── Pickaxe ────────────────────────────────────────────────────────────────
    pk_curr = _count_items(inv_curr, PICKAXE_ITEMS)
    if not milestones.get("pickaxe_unlocked") and pk_curr > 0:
        reward += R_PICKAXE_FIRST
        milestones["pickaxe_unlocked"] = True

    # ── Cobblestone ────────────────────────────────────────────────────────────
    cob_prev = _count_items(inv_prev, COBBLESTONE_ITEMS)
    cob_curr = _count_items(inv_curr, COBBLESTONE_ITEMS)
    if cob_curr > cob_prev:
        reward += (cob_curr - cob_prev) * R_COBBLESTONE_PER_ITEM

    # ── Food items ─────────────────────────────────────────────────────────────
    food_prev = _count_items(inv_prev, FOOD_ITEMS)
    food_curr = _count_items(inv_curr, FOOD_ITEMS)
    if food_curr > food_prev:
        reward += (food_curr - food_prev) * R_FOOD_PER_ITEM

    # ── Damage / death ─────────────────────────────────────────────────────────
    hp_prev = float(obs_prev.get("health", 20.0))
    hp_curr = float(obs_curr.get("health", 20.0))
    if hp_curr < hp_prev:
        reward -= (hp_prev - hp_curr) * abs(R_DAMAGE_PENALTY)
    if hp_curr <= 0.0 and hp_prev > 0.0:
        reward += R_DEATH_PENALTY

    return float(reward), milestones
