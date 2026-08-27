# Remaining code-free questions for Fable

The golden trace pack resolved the raw ECA simulator. The clean-room engine now
matches all 88 retained atlas rows and all 88 form libraries exactly. No sibling
source, tests, scripts, or executable seeds were inspected.

Only three downstream provenance questions remain:

1. **ECA phase RNG.** What exact seed tag/byte recipe initializes each
   `(eta, rule, seed_index)` phase stream? The disclosed `eca-traj-v1` tag
   exactly reproduces the atlas, while the phase grid is evidently a fresh
   stream. Current phase agreement is already strong (strict Spearman 0.9636,
   break Spearman 0.9988), and all registered phase gates match.
2. **Particle dictionary launches.** Could you provide the four 64-bit hash
   rows—or just their hexadecimal digests—used for each rule's noiseless domain
   dictionary? The corrected lifecycle gives rule 110 strict 37/2048 versus the
   retained 38/2048, but exact particle comparison is not justified without
   those rows.
3. **Life trajectory RNG and boundary trace.** For one named glider future,
   could you provide the launch, first two generation sweep counts, terminal
   boards, process/copy masks, and stream seed recipe? Round-5 pooling is now
   implemented as disclosed, yet supports remain glider 159, blinker 667, toad
   2715 versus retained 2715, 667, 671.

Optional analysis-only clarification: with the atlas rows and libraries now
identical, what random-pair seed and exact library-distance calculation produced
the retained descriptor/Hamming correlations 0.13497 and 0.15339? The
clean-room re-adjudication gets 0.13649 and 0.11215, so this no longer concerns
the simulator.
