#!/usr/bin/env bash
# Overnight v4 development run (2026-11-05 -> 06; docs/v4_plan.md sections 9-10, plan items 1-3).
#   lanes 1-4: stacked v4 variants on all 13 benchmark datasets, splits 20-29 (development; not held-out)
#     v4k   stacked GLS + reallocated contrast, GLS-se noise ratio            (section 9)
#     v4kn  stacked, noise-matched simulated cohorts                         (plan item 2)
#     v4kr  stacked + ridge-GLS third estimator, 60 cohorts                    (plan item 1)
#     v4knr both
#   lane 5: CyTOF-anchored fractions on COMBAT real bulk, splits 20-24        (plan item 3)
# then v4_overnight_summary.py -> benchmarks/dev/overnight_2026-11-05.txt
E=/c/Users/duy.pham/AppData/Local/miniforge3/envs/program2cell
export PATH="$E:$E/Library/mingw-w64/bin:$E/Library/usr/bin:$E/Library/bin:$E/Scripts:$E/bin:$PATH"
cd /c/Users/duy.pham/Documents/dev/program2cell/scripts/experiments || exit 1
L=/c/Users/duy.pham/Documents/dev/program2cell/benchmarks/dev/causal_loop
V="v4k,v4kn,v4kr,v4knr"
date
python causal_loop.py run kang,zhang 20-29 v4k_pair,v4kn_pair,v4kr_pair,v4knr_pair > $L/overnight_1.log 2>&1 &
python causal_loop.py run mcf_ida,mcf_tram,tahoe_rmc 20-29 $V > $L/overnight_2.log 2>&1 &
python causal_loop.py run tahoe_gem,tahoe_enc,tahoe_tram 20-29 $V > $L/overnight_3.log 2>&1 &
python causal_loop.py run parse_ifnb,parse_ifng,parse_il4,parse_tnf,parse_gmcsf 20-29 $V > $L/overnight_4.log 2>&1 &
mkdir -p ../../benchmarks/dev/combat_cytof
python combat_cytof.py 20-24 > ../../benchmarks/dev/combat_cytof/run.log 2>&1 &
wait
date
python v4_overnight_summary.py > ../../benchmarks/dev/overnight_2026-11-05.txt 2>&1
date
echo OVERNIGHT DONE
