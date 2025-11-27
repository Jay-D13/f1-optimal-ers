# F1 ERS Baseline Strategies - Quick Reference

## You Now Have 3 Baseline Strategies

### 1. **Pure Greedy** (Classic Literature Baseline) ⭐
**File:** [controllers/greedy_baseline.py](controllers/greedy_baseline.py) - `PureGreedyStrategy`

**Strategy:**
- Deploy max power on straights (radius > 500m)
- Harvest max power when braking
- No lookahead or optimization

**When to use:**
- Comparing against literature (standard benchmark)
- Showing improvement in papers/reports
- Expected: 2-7% slower than optimal

**Typical quote:**
> "The offline optimizer achieves 3.4% improvement over the pure greedy baseline, consistent with literature expectations."

---

### 2. **Smart Heuristic** (Your Original Baseline) ⭐⭐
**File:** [controllers/simple_rule.py](controllers/simple_rule.py) - `SimpleRuleBasedStrategy`

**Strategy:**
- 150m lookahead for corners
- Deploy at full throttle (avoiding drag-limited speeds)
- Harvest when braking
- P-controller with safety margins

**When to use:**
- Showing "smart but non-optimal" comparison
- Demonstrating value beyond intelligent heuristics
- Expected: 0.5-2% slower than optimal

**Typical quote:**
> "Even compared to a lookahead-based heuristic, the optimizer achieves 0.9s improvement, demonstrating value beyond simple rules."

---

### 3. **Always Deploy** (Worst Case) ⭐⭐⭐
**File:** [controllers/greedy_baseline.py](controllers/greedy_baseline.py) - `AlwaysDeployGreedy`

**Strategy:**
- Deploy whenever not braking (very wasteful)
- Harvest when braking
- No intelligence

**When to use:**
- Showing full improvement spectrum
- Demonstrating why strategy matters
- Expected: 3-10% slower than optimal

**Typical quote:**
> "The optimization reduces lap time by 4.8s compared to naive always-deploy, validating the importance of strategic energy management."

---

## Where These Are Used

### In `validate_project.py`:
```python
# Runs Pure Greedy and Smart Heuristic
# Compares:
# - Offline Optimal (fastest)
# - Smart Heuristic (intelligent)
# - Pure Greedy (literature standard)
#
# Tests 15 validations including:
# - Smart beats Greedy (validates lookahead)
# - Optimal improvement 1-10% (literature range)
```

**Run it:**
```bash
python validate_project.py
```

**Output:**
- 15 automated tests
- Validation report
- `validation_analysis.png` with comparison plots

---

### In `compare_baselines.py`:
```python
# Runs ALL 4 strategies:
# 1. Offline Optimal
# 2. Smart Heuristic
# 3. Pure Greedy
# 4. Always Deploy
#
# Creates full comparison hierarchy
```

**Run it:**
```bash
python compare_baselines.py
```

**Output:**
- Complete lap time hierarchy
- Energy usage comparison
- `baseline_comparison.png` with 6 plots

---

## Quick Decision Guide

**For academic papers:**
→ Use **Pure Greedy** as primary baseline
→ Cite literature expectation: 2-7% improvement

**For demonstrating intelligence:**
→ Use **Smart Heuristic** to show optimization beats smart rules
→ Shows value beyond engineering best practices

**For showing full spectrum:**
→ Include **Always Deploy** as worst case
→ Demonstrates importance of strategy

**For comprehensive validation:**
→ Run `validate_project.py` (Greedy + Smart)
→ 15 tests covering physics, performance, intelligence

**For full comparison:**
→ Run `compare_baselines.py` (all 4 strategies)
→ Complete hierarchy from worst to best

---

## Expected Results (Monaco)

| Strategy | Lap Time | vs Optimal | Use Case |
|----------|----------|------------|----------|
| Offline Optimal | 67.2s | - | Your optimizer |
| Smart Heuristic | 68.1s | +0.9s | Intelligent baseline |
| Pure Greedy | 69.6s | +2.4s | Literature baseline |
| Always Deploy | 71.8s | +4.6s | Worst case |

**Key Validations:**
✓ Smart beats Greedy (validates lookahead value)
✓ Optimal beats Smart (validates optimization value)
✓ Improvement 2-7% vs Greedy (matches literature)

---

## Answer to Your Question

> "Is this one of the typical known baseline comparison strategies?"

**Your Smart Heuristic:** Not a classic greedy, but **better** because:
- More realistic (what engineers might actually implement)
- Shows your optimizer beats intelligent strategies
- Validates lookahead and drag-limit logic

**Pure Greedy:** Classic literature baseline ✓
- Standard in papers (Borhan 2012, Serrao 2011)
- Simple: deploy on straights, harvest when braking
- Expected 2-7% improvement for optimal vs greedy

**Recommendation:**
1. Keep **Smart Heuristic** as your main baseline (more realistic)
2. Include **Pure Greedy** for literature comparison
3. *Optional:* Add **Always Deploy** for worst case

This gives you the full story: naive → greedy → smart → optimal
