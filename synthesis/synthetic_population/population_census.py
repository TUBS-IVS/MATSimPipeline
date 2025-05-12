import numpy as np
import math
import pandas as pd
from collections import defaultdict


# Redefine get_likelihood and perturbation_probs after reset

perturbation_probs = {
    0: {0: 1.00},
    1: {-1: 0.65, 2: 0.30, 3: 0.05},
    2: {-2: 0.40, 1: 0.50, 2: 0.10},
    3: {0: 0.40, 1: 0.35, 2: 0.15, -3: 0.05, 3: 0.05},
    4: {0: 0.35, -1: 0.20, 1: 0.20, 2: 0.10, 3: 0.05, -4: 0.05, 4: 0.05},
    5: {0: 0.35, -1: 0.175, 1: 0.175, -2: 0.10, 2: 0.10, 3: 0.05, 4: 0.05},
    6: {0: 0.35, -1: 0.175, 1: 0.175, -2: 0.075, 2: 0.075, -3: 0.05, 3: 0.05, 4: 0.05},
    7: {0: 0.35, -1: 0.15, 1: 0.15, -2: 0.075, 2: 0.075, -3: 0.05, 3: 0.05, -4: 0.05, 4: 0.05},
}

def get_likelihood(observed, true):
    base_true = true if true in perturbation_probs else 7
    distribution = perturbation_probs.get(base_true, perturbation_probs[7])
    delta = observed - true
    return distribution.get(delta, 0.0)

def greedy_bayesian_adjust_multistep_exact(observed_vals, target_sum, max_step=4):
    observed_vals = [int(o) for o in observed_vals]
    n = len(observed_vals)
    best_estimates = observed_vals.copy()

    delta = target_sum - sum(best_estimates)
    if delta == 0:
        return best_estimates
    if abs(delta) < max_step:
        max_step = abs(delta)

    # Collect all potential steps from original observed values
    adjustments = []
    for i, obs in enumerate(observed_vals):
        for step in range(1, max_step + 1):
            p_old = get_likelihood(obs, obs)
            # positive adjustments
            if delta > 0:
                p_new = get_likelihood(obs, obs + step)

                if p_new > 0 and p_old > 0:
                    loss = -np.log(p_new) + np.log(p_old)
                    adjustments.append((loss, i, step))
            # negative adjustments
            if delta < 0:
                if obs - step >= 0:
                    p_new = get_likelihood(obs, obs - step)
                    if p_new > 0 and p_old > 0:
                        loss = -np.log(p_new) + np.log(p_old)
                        adjustments.append((loss, i, -step))

    # Sort by smallest likelihood loss
    adjustments.sort()

    # Apply adjustments relative to original observed values
    applied = defaultdict(int)
    current_total = sum(best_estimates)

    average_needed_adjustment = delta / n
    adjusted_indices = set()
    i = 0
    while i < len(adjustments) and current_total != target_sum:
        _, idx, step = adjustments[i]

        if idx in adjusted_indices:
            i += 1
            continue
        # Skip steps too large in the current delta context
        if abs(step) > (abs(current_total - target_sum)):
            i += 1
            continue
        # Skip steps that are just too small
        if abs(step) < abs(average_needed_adjustment) * 0.9:
            i += 1
            continue
        candidate = observed_vals[idx] + step
        if candidate >= 0:
            applied[idx] = step
            current_total += step
            adjusted_indices.add(idx)
        i += 1

    i = 0

    # Try to fill the remainder
    while i < len(adjustments) and current_total != target_sum:
        _, idx, step = adjustments[i]

        if idx in adjusted_indices:
            i += 1
            continue
        # Skip steps too large in the current delta context
        if abs(step) > (abs(current_total - target_sum)):
            i += 1
            continue
        # Skip steps that are just too small (looser)
        if abs(step) < abs(average_needed_adjustment) * 0.5:
            i += 1
            continue
        candidate = observed_vals[idx] + step
        if candidate >= 0:
            applied[idx] = step
            current_total += step
            adjusted_indices.add(idx)
        i += 1

    while i < len(adjustments) and current_total != target_sum:
        _, idx, step = adjustments[i]

        if idx in adjusted_indices:
            i += 1
            continue
        # Skip steps too large in the current delta context
        if abs(step) > (abs(current_total - target_sum)):
            i += 1
            continue
        # Don't skip steps that are just too small this round

        candidate = observed_vals[idx] + step
        if candidate >= 0:
            applied[idx] = step
            current_total += step
            adjusted_indices.add(idx)
        i += 1

    while current_total != target_sum:
        remainder = target_sum - current_total
        step = 1 if remainder > 0 else -1
        # Sort indices by value descending to minimize relative impact
        sorted_indices = sorted(range(n), key=lambda i: observed_vals[i] + applied[i], reverse=True)
        for idx in sorted_indices[:abs(remainder)]:
            applied[idx] += step
        current_total = sum(observed_vals[i] + applied[i] for i in range(n))

    assert current_total == target_sum, f"Adjustment failed: final={current_total}, target={target_sum}"

    # Construct final estimate
    corrected = [observed_vals[i] + applied[i] for i in range(n)]
    assert sum(corrected) == target_sum
    return corrected


def run():

    # Load data
    merged_100 = pd.read_pickle(r"C:\Users\petre\Documents\GitHub\MATSimPipeline\data\syn_pop\merged_100m_gitter.pkl")
    merged_1km = pd.read_pickle(r"C:\Users\petre\Documents\GitHub\MATSimPipeline\data\syn_pop\merged_1km_gitter.pkl")
    merged_10km = pd.read_pickle(r"C:\Users\petre\Documents\GitHub\MATSimPipeline\data\syn_pop\merged_10km_gitter.pkl")

    # --- Align total 10km population to national total ---
    national_target = 82_719_540
    current_total_10km = merged_10km["Insgesamt_Bevoelkerung_Zensus2022_Alter_in_10er-Jahresgruppen_10km-Gitter"].fillna(0).astype(np.int32).sum()
    print(f"Original 10km total: {current_total_10km} → Target: {national_target}")
    # Extract and correct
    original_10km = merged_10km["Insgesamt_Bevoelkerung_Zensus2022_Alter_in_10er-Jahresgruppen_10km-Gitter"].fillna(0).astype(np.int32).values
    corrected_10km = greedy_bayesian_adjust_multistep_exact(original_10km, national_target)
    merged_10km["Insgesamt_Bevoelkerung"] = corrected_10km
    merged_10km["Bevoelkerung_adjust_diff"] = merged_10km["Insgesamt_Bevoelkerung"] - merged_10km["Insgesamt_Bevoelkerung_Zensus2022_Alter_in_10er-Jahresgruppen_10km-Gitter"]
    print("Saving 10km")
    merged_10km.to_csv(r"C:\Users\petre\Documents\GitHub\MATSimPipeline\data\syn_pop\merged_10km_poptotaladjusted.csv")
    merged_10km.to_pickle(r"C:\Users\petre\Documents\GitHub\MATSimPipeline\data\syn_pop\merged_10km_poptotaladjusted.pkl")
    print("Saved")

    # 10km -> 1km adjustment
    runs=0
    limiter = 200
    for gid_10km in merged_10km["GITTER_ID_10km"].unique():
        runs+=1
        if runs%1000 == 1:
            print(runs)
        # if runs > limiter:
        #     print("Limit reached")
        #     break
        target = merged_10km.loc[merged_10km["GITTER_ID_10km"] == gid_10km, "Insgesamt_Bevoelkerung"].values
        if len(target) == 0: continue
        if pd.isnull(target):
            target = [0]
        target = int(round(target[0]))
        idx_1km = merged_1km["Linked_10km"] == gid_10km
        observed = merged_1km.loc[idx_1km, "Insgesamt_Bevoelkerung_Zensus2022_Alter_in_10er-Jahresgruppen_1km-Gitter"]
        observed = observed.fillna(0).astype(np.int32).values
        if len(observed) == 0: continue
        corrected = greedy_bayesian_adjust_multistep_exact(observed, target)
        corrected = np.asarray(corrected, dtype=np.int32)
        merged_1km.loc[idx_1km, f"Insgesamt_Bevoelkerung"] = corrected

    # 1km -> 100m adjustment
    runs=0
    limiter = 200
    grouped_100m = merged_100.groupby("Linked_1km", sort=False)

    for gid_1km in merged_1km["GITTER_ID_1km"].unique():
        runs+=1
        if runs%1000 == 1:
            print(runs)
        # if runs > limiter:
        #     print("Limit reached")
        #     break
        target = merged_1km.loc[merged_1km["GITTER_ID_1km"] == gid_1km, "Insgesamt_Bevoelkerung"].values
        if len(target) == 0: continue
        if pd.isnull(target):
            target = [0]
        target = int(round(target[0]))
        try:
            df_100m = grouped_100m.get_group(gid_1km)
            observed = df_100m["Insgesamt_Bevoelkerung_Zensus2022_Alter_in_10er-Jahresgruppen_100m-Gitter"].fillna(
                0).astype(np.int32).values
            corrected = greedy_bayesian_adjust_multistep_exact(observed, target)
            merged_100.loc[df_100m.index, "Insgesamt_Bevoelkerung"] = corrected
        except KeyError:
            continue
    merged_100["Bevoelkerung_adjust_diff"] = merged_100["Insgesamt_Bevoelkerung"] - merged_100["Insgesamt_Bevoelkerung_Zensus2022_Alter_in_10er-Jahresgruppen_100m-Gitter"]
    merged_1km["Bevoelkerung_adjust_diff"] = merged_1km["Insgesamt_Bevoelkerung"] - merged_1km["Insgesamt_Bevoelkerung_Zensus2022_Alter_in_10er-Jahresgruppen_1km-Gitter"]

    print (f"Summe 100m:{merged_100["Insgesamt_Bevoelkerung"].sum()}")
    print (f"Summe 1km:{merged_1km["Insgesamt_Bevoelkerung"].sum()}")
    print (f"Summe 10km:{merged_10km["Insgesamt_Bevoelkerung"].sum()}")
    #
    # # === Check 10km sums ===
    # for gid in merged_10km["GITTER_ID_10km"].unique()[:limiter]:
    #     target = merged_10km.loc[merged_10km["GITTER_ID_10km"] == gid,
    #                              "Insgesamt_Bevoelkerung"].values
    #     if len(target) == 0: continue
    #     if pd.isnull(target):
    #         target = [0]
    #     target = int(round(target[0]))
    #     actual = merged_1km.loc[merged_1km["Linked_10km"] == gid,
    #                             "Insgesamt_Bevoelkerung"].sum()
    #     # assert target == int(round(actual)), f"Mismatch in 10km cell {gid}: target={target}, actual={actual}"
    #     if target != int(round(actual)):
    #         print(f"Mismatch in 10km cell {gid}: target={target}, actual={actual}")
    # # === Check 1km sums ===
    # for gid in merged_1km["GITTER_ID_1km"].unique()[:limiter]:
    #     target = merged_1km.loc[merged_1km["GITTER_ID_1km"] == gid,
    #                             "Insgesamt_Bevoelkerung"].values
    #     if len(target) == 0: continue
    #     if pd.isnull(target):
    #         target = [0]
    #     target = int(round(target[0]))
    #     actual = merged_100.loc[merged_100["Linked_1km"] == gid,
    #                             "Insgesamt_Bevoelkerung"].sum()
    #     # assert target == int(round(actual)), f"Mismatch in 1km cell {gid}: target={target}, actual={actual}"
    #     if target != int(round(actual)):
    #         print(f"Mismatch in 1km cell {gid}: target={target}, actual={actual}")
    # print("✅ All sums match at both 10km→1km and 1km→100m levels.")

    merged_1km.to_csv(r"C:\Users\petre\Documents\GitHub\MATSimPipeline\data\syn_pop\merged_1km_poptotaladjusted.csv")
    merged_100.to_csv(r"C:\Users\petre\Documents\GitHub\MATSimPipeline\data\syn_pop\merged_100m_poptotaladjusted.csv")
    merged_1km.to_pickle(r"C:\Users\petre\Documents\GitHub\MATSimPipeline\data\syn_pop\merged_1km_poptotaladjusted.pkl")
    merged_100.to_pickle(r"C:\Users\petre\Documents\GitHub\MATSimPipeline\data\syn_pop\merged_100m_poptotaladjusted.pkl")


import cProfile
import subprocess
import sys

if __name__ == "__main__":
    profile_file = "profile_output.prof"
    cProfile.run("run()", filename=profile_file)

    print("📊 Launching SnakeViz...")
    subprocess.run([sys.executable, "-m", "snakeviz", profile_file])
