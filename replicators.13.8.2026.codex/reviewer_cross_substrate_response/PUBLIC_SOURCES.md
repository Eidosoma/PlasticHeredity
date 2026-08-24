# Public model sources

Only the following public materials define the two substrate contracts.  The
unpublished motivating exercise is excluded by `HYPOTHESIS_PROVENANCE.md`.

1. Atsushi Kamimura and Kunihiko Kaneko (2014), "Compartmentalization and
   Cell Division through Molecular Discreteness and Crowding in a Catalytic
   Reaction Network," *Life* 4(4), 586--597.
   <https://doi.org/10.3390/life4040586>
2. Hiroki Sayama (1999), "A New Structurally Dissolvable Self-Reproducing
   Loop Evolving in a Simple Cellular Automata Space," *Artificial Life*
   5(4), 343--365.  Author-hosted PDF:
   <https://bingdev.binghamton.edu/sayama/papers/ALIFE5-4-343.pdf>
3. The public Golly `Evoloop.table` transition table distributed through the
   Cellular Automata project:
   <https://sourceforge.net/projects/cellularauto/files/Golly/Evoloop.table/download>

The local transition table is frozen by the source manifest emitted before
simulation.  The code independently validates its 132 rule rows, 55,139
covered rotate-four neighbourhoods, and a 150-tick canonical-seed fixture.
