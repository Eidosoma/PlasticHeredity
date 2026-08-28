# Questions for Fable — ECA Clean-Room Reconciliation

The exhaustive clean-room sensitivity round did not meet the frozen strong-match
bar. No sibling source, tests, scripts, or seeds were inspected.

## Best surviving settings

- `322e82cd2d0e0d80`: strict rho 0.749, break rho 0.763, strict MAE 0.026, break MAE 0.142; settings `{"activity_count": "realized", "launch_anchor": "first_completed_generation", "launch_preparation": "noiseless_generation", "monochrome_death": "deterministic_immediate", "observed_daughter": "post_copy_offspring", "process_noise": "pre_rule_each_sweep", "seed_mode": "exact_half"}`.
- `b35342e5d6a7e996`: strict rho 0.743, break rho 0.748, strict MAE 0.026, break MAE 0.149; settings `{"activity_count": "deterministic", "launch_anchor": "first_completed_generation", "launch_preparation": "noiseless_generation", "monochrome_death": "terminal_only", "observed_daughter": "post_copy_offspring", "process_noise": "pre_rule_each_sweep", "seed_mode": "exact_half"}`.

## Minimal executable-contract questions

1. Are the 16 launch rows observable parents, or is composition zero the first
   completed activity-gated generation?
2. Is there deterministic launch preparation or burn-in? If so, how many sweeps
   or what stopping rule is used?
3. How are the 16 frozen launch rows constructed (exact density, PRNG/hash
   family, and whether rows are shared across rules)? Providing the rows or
   their hex digests would resolve the largest remaining ensemble ambiguity.
4. Within one sweep, is process noise applied before or after the ECA rule?
5. Does the 256-change activity counter include noise flips, deterministic rule
   changes only, or the realized stored-row difference?
6. Is monochrome death checked during a generation, only when its activity
   clock stops, before process noise, or after process noise?
7. Is the observed daughter the terminal row before copy error or the offspring
   after copy error?
8. For the Life round-5 ensemble form, which generations/futures are pooled,
   and is each generation normalized before the 0.75 mass support is taken?

## Most useful code-free golden traces

For rules 8, 13, 35, 110, and 172 on one frozen launch row, please provide the
launch row, terminal row, sweep count, death flag, and observed final4 vector
for the first two generations at `(eta, epsilon)=(0,0)` and at
`(0.01,0.015)`. A supplied sequence of process/copy masks would let us replay
the stochastic trace without seeing implementation code.

## Automated adjudication snapshot

```json
{
  "best_setting": {
    "activity_count": "realized",
    "launch_anchor": "first_completed_generation",
    "launch_preparation": "noiseless_generation",
    "monochrome_death": "deterministic_immediate",
    "observed_daughter": "post_copy_offspring",
    "process_noise": "pre_rule_each_sweep",
    "seed_mode": "exact_half"
  },
  "best_setting_id": "322e82cd2d0e0d80",
  "confirmation_details": {
    "322e82cd2d0e0d80": {
      "champions_in_top10": 1,
      "class_medians_ok": false,
      "false_champions_ok": true,
      "passed": false,
      "reference_class_medians": {
        "1": 0.025146484375,
        "2": 0.36279296875,
        "3": 0.99755859375,
        "4": 0.99853515625
      },
      "top10": [
        1,
        3,
        19,
        33,
        35,
        50,
        56,
        138,
        178,
        232
      ]
    },
    "b35342e5d6a7e996": {
      "champions_in_top10": 1,
      "class_medians_ok": false,
      "false_champions_ok": true,
      "passed": false,
      "reference_class_medians": {
        "1": 0.025146484375,
        "2": 0.36279296875,
        "3": 0.99755859375,
        "4": 0.99853515625
      },
      "top10": [
        1,
        3,
        19,
        33,
        35,
        50,
        51,
        56,
        178,
        232
      ]
    }
  },
  "design_digest": "e8d57cd895af6089c1d0c9b7d53031db190d2020e5a3ae82f8484dee01ed1eda",
  "downstream": {
    "all_downstream_gates": false,
    "atlas_gates": {
      "champion_strict": {
        "11": 0.048828125,
        "184": 0.0146484375,
        "35": 0.05908203125,
        "43": 0.0029296875,
        "57": 0.00537109375
      },
      "class3_min_break_by_8": 0.96044921875,
      "class_break_medians": {
        "1": 0.0,
        "2": 0.56494140625,
        "3": 1.0,
        "4": 0.99951171875
      },
      "clean_class12_median_break_by_8": 0.360107421875,
      "edge_mean_jaccard": 0.12239583333333333,
      "gate_class3_separation": true,
      "gate_heavy_tail": true,
      "gate_metric": true,
      "gate_rule110_top_decile": false,
      "gate_smoothness": true,
      "heavy_tail_share": 0.6266233766233766,
      "metric_rho_descriptors": 0.11724143540733696,
      "metric_rho_hamming": 0.0932268925134048,
      "n_forms": 182,
      "random_mean_jaccard": 0.02092308266703428,
      "raw_class4_strict": {
        "106": 0.0,
        "110": 0.0,
        "41": 0.0,
        "54": 0.0
      },
      "rule110_strict": 0.0,
      "rule110_strict_rank": 74.0,
      "smoothness_ratio": 5.849799251913112
    },
    "gates": {
      "evolution_parity": false,
      "life_fidelity_persistence_forms": true,
      "particle_redemption_and_coverage": false,
      "phase_parity": true
    },
    "reference_comparison": {
      "break_by_8_mean_absolute_error": 0.14129083806818182,
      "break_by_8_spearman": 0.7637538292461242,
      "directional_checks": {
        "at_least_three_raw_champions_nonzero": true,
        "class3_separation": true,
        "raw_class4_strict_below_0.005": true,
        "rule110_strict_below_0.005": true
      },
      "n_common_rules": 88,
      "selected_rules": {
        "106": {
          "ours": {
            "break_by_8": 1.0,
            "strict": 0.0,
            "wolfram_class": 4.0
          },
          "reference": {
            "break_by_8": 0.9990234375,
            "strict": 0.0,
            "wolfram_class": 4.0
          }
        },
        "11": {
          "ours": {
            "break_by_8": 0.525390625,
            "strict": 0.048828125,
            "wolfram_class": 2.0
          },
          "reference": {
            "break_by_8": 0.8466796875,
            "strict": 0.33984375,
            "wolfram_class": 2.0
          }
        },
        "110": {
          "ours": {
            "break_by_8": 0.99951171875,
            "strict": 0.0,
            "wolfram_class": 4.0
          },
          "reference": {
            "break_by_8": 0.998046875,
            "strict": 0.0,
            "wolfram_class": 4.0
          }
        },
        "150": {
          "ours": {
            "break_by_8": 1.0,
            "strict": 0.0,
            "wolfram_class": 3.0
          },
          "reference": {
            "break_by_8": 1.0,
            "strict": 0.0,
            "wolfram_class": 3.0
          }
        },
        "184": {
          "ours": {
            "break_by_8": 0.044921875,
            "strict": 0.0146484375,
            "wolfram_class": 2.0
          },
          "reference": {
            "break_by_8": 0.36279296875,
            "strict": 0.16796875,
            "wolfram_class": 2.0
          }
        },
        "30": {
          "ours": {
            "break_by_8": 1.0,
            "strict": 0.0,
            "wolfram_class": 3.0
          },
          "reference": {
            "break_by_8": 1.0,
            "strict": 0.0,
            "wolfram_class": 3.0
          }
        },
        "35": {
          "ours": {
            "break_by_8": 0.255859375,
            "strict": 0.05908203125,
            "wolfram_class": 2.0
          },
          "reference": {
            "break_by_8": 0.7578125,
            "strict": 0.4345703125,
            "wolfram_class": 2.0
          }
        },
        "41": {
          "ours": {
            "break_by_8": 0.99951171875,
            "strict": 0.0,
            "wolfram_class": 4.0
          },
          "reference": {
            "break_by_8": 1.0,
            "strict": 0.0,
            "wolfram_class": 4.0
          }
        },
        "43": {
          "ours": {
            "break_by_8": 0.09033203125,
            "strict": 0.0029296875,
            "wolfram_class": 2.0
          },
          "reference": {
            "break_by_8": 0.728515625,
            "strict": 0.3486328125,
            "wolfram_class": 2.0
          }
        },
        "54": {
          "ours": {
            "break_by_8": 0.9990234375,
            "strict": 0.0,
            "wolfram_class": 4.0
          },
          "reference": {
            "break_by_8": 0.99755859375,
            "strict": 0.0,
            "wolfram_class": 4.0
          }
        },
        "57": {
          "ours": {
            "break_by_8": 0.11962890625,
            "strict": 0.00537109375,
            "wolfram_class": 2.0
          },
          "reference": {
            "break_by_8": 0.51611328125,
            "strict": 0.29296875,
            "wolfram_class": 2.0
          }
        },
        "90": {
          "ours": {
            "break_by_8": 1.0,
            "strict": 0.0,
            "wolfram_class": 3.0
          },
          "reference": {
            "break_by_8": 1.0,
            "strict": 0.0,
            "wolfram_class": 3.0
          }
        }
      },
      "strict_mean_absolute_error": 0.02617853338068182,
      "strict_spearman": 0.7992842203659892
    },
    "setting": {
      "activity_count": "realized",
      "launch_anchor": "first_completed_generation",
      "launch_preparation": "noiseless_generation",
      "monochrome_death": "deterministic_immediate",
      "observed_daughter": "post_copy_offspring",
      "process_noise": "pre_rule_each_sweep",
      "seed_mode": "exact_half"
    }
  },
  "elapsed_seconds": 9793.942387580872,
  "holdout_passes": [],
  "overall_success": false,
  "raw_pass_settings": [],
  "raw_strong_match": false
}
```
