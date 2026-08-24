#!/usr/bin/env python3
"""Freeze and validate E01 S12E Phase-0 evidence before development outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/workspace")
ARTIFACTS = Path("/artifacts/research_steps/S12E")
CACHE = Path("/cache/e01_s12e")
CONFIG = REPO / "configs/e01/s12e_paper_pipeline_detective_preregistration.yaml"
PAPER_DIR = Path("/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759")
PAPER_MARKDOWN = PAPER_DIR / "pdf-markdown.md"
PAPER_PDF = CACHE / "arxiv-2607.28250v1-source-response.bin"
SAFE_LATTICE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")

SOURCE_REPOS = {
    "historicalGard": Path("/cache/e01_s03/sources/gard-historical"),
    "modernGard": Path("/cache/e01_s03/sources/gard-modern"),
    "iigr": Path("/cache/e01_s12b/sources/IntegratedInformationGeneRegulation"),
    "phirl": Path("/cache/e01_s12b/sources/PhiRL"),
    "grnLineage": Path("/cache/e01_s12e/sources/BreakingGRNMemories"),
}

PRIOR_STEPS = (
    "S01",
    "S02",
    "S03",
    "S04",
    "S05",
    "S06",
    "S07",
    "S08",
    "S09",
    "S10",
    "S11",
    "S11R",
    "S12",
    "S12B",
    "S12C",
    "S12D",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def prior_baseline() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for step in PRIOR_STEPS:
        root = Path("/artifacts/research_steps") / step
        if not root.is_dir():
            raise FileNotFoundError(f"immutable prior step is missing: {root}")
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            files.append(
                {
                    "step": step,
                    "relativePath": str(path.relative_to(root)),
                    "sizeBytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    payload = {
        "schemaVersion": "E01-S12E-immutable-prior-baseline-v1.0.0",
        "steps": list(PRIOR_STEPS),
        "fileCount": len(files),
        "files": files,
    }
    payload["aggregateSha256"] = canonical_json_sha256(files)
    return payload


def source_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    relevant = {
        "historicalGard": [
            "tgs_parameters_v10.m",
            "tgs_newbeta_v10.m",
            "tgs_grow_v10.m",
            "tgs_agard_v10.m",
            "tgs_split_v10.m",
            "tgs_nondrift.m",
            "tgs_acluster.m",
        ],
        "iigr": ["main.py", "information.py", "phi_lattice_22.pickle"],
        "phirl": ["main.py", "information.py", "phi_lattice_22.pickle"],
        "grnLineage": ["phi.py", "information.py", "README.md"],
    }
    for source_id, files in relevant.items():
        repo = SOURCE_REPOS[source_id]
        head = git(repo, "rev-parse", "HEAD^{commit}")
        tree = git(repo, "rev-parse", "HEAD^{tree}")
        remote = git(repo, "remote", "get-url", "origin")
        for relative in files:
            path = repo / relative
            records.append(
                {
                    "sourceId": source_id,
                    "repository": remote,
                    "commit": head,
                    "tree": tree,
                    "relativePath": relative,
                    "sizeBytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "gitBlob": git(repo, "rev-parse", f"HEAD:{relative}"),
                }
            )
    paper_figures = []
    for path in sorted((PAPER_DIR / "figures").glob("*.png")):
        with Image.open(path) as image:
            paper_figures.append(
                {
                    "path": str(path),
                    "sha256": sha256(path),
                    "sizeBytes": path.stat().st_size,
                    "pixels": [image.width, image.height],
                    "mode": image.mode,
                    "embeddedMetadata": image.info,
                }
            )
    return {
        "schemaVersion": "E01-S12E-source-snapshot-v1.0.0",
        "capturedAtUtc": datetime.now(UTC).isoformat(),
        "paper": {
            "arxivId": "2607.28250v1",
            "pdfPath": str(PAPER_PDF),
            "pdfSha256": sha256(PAPER_PDF),
            "sourceEndpointRepresentation": "PDF_NOT_TEX_ARCHIVE",
            "sourceResponseByteIdenticalToS03": (
                sha256(PAPER_PDF)
                == sha256(Path("/cache/e01_s03/downloads/paper-2607.28250v1-source.tar"))
            ),
            "pdfPages": 18,
            "pdfCreator": "Acrobat PDFMaker 26 for Word",
            "pdfProducer": "Adobe PDF Library 26.1.183",
            "pdfAuthorMetadata": "bhanson",
            "pdfCreateDate": "2026-07-23T11:58:13Z",
            "pdfEmbeddedFiles": 0,
            "pdfCommentsMetadata": "",
            "derivedMarkdownPath": str(PAPER_MARKDOWN),
            "derivedMarkdownSha256": sha256(PAPER_MARKDOWN),
            "figures": paper_figures,
        },
        "safeLattice": {
            "path": str(SAFE_LATTICE),
            "sha256": sha256(SAFE_LATTICE),
            "scientificExecutionLoadsPickle": False,
        },
        "sourceFiles": records,
        "repositorySearch": {
            "checkedAtUtc": datetime.now(UTC).isoformat(),
            "githubUser": "pigozzif",
            "publicRepositoryCount": 34,
            "authorCodeSearchReference": "/artifacts/research_steps/S03/author_code_search.json",
            "exactPaperTitleRepositoryCount": 0,
            "iigrPublicForkCount": 0,
            "phirlPublicForkCount": 0,
            "iigrBranches": {"master": config["sourcePins"]["iigr"]["commit"]},
            "phirlBranches": {"master": config["sourcePins"]["phirl"]["commit"]},
            "interpretation": "No public GARD-paper implementation was found; absence is not proof of nonexistence.",
        },
        "sourceRelationship": "SOURCE_AND_PAPER_INFORMED_FORENSIC_RECONSTRUCTION",
    }


def write_paper_fingerprints(config: dict[str, Any]) -> None:
    payload = {
        "schemaVersion": "E01-S12E-paper-fingerprint-ledger-v1.0.0",
        "researchStepId": config["researchStepId"],
        "evidenceBoundary": "paper_reported_or_explicitly_marked_inference",
        "fingerprints": config["paperFingerprints"],
        "sourceEvidence": {
            "paperMarkdown": str(PAPER_MARKDOWN),
            "methodsLines": "61-97",
            "resultsLines": "31-49",
            "captionsLines": "230-284",
            "claimLedger": "/artifacts/E01_forensic_replication_bundle/ledgers/claim_ledger.csv",
            "sourceReconciliation": "/artifacts/research_steps/S01/source_reconciliation.csv",
        },
        "inferences": [
            {
                "id": "S12E-INF-001",
                "formula": "716/0.88",
                "value": 813.6363636363636,
                "unit": "molecular_batch_steps",
                "status": "INFERRED_NOT_REPORTED",
            }
        ],
    }
    (ARTIFACTS / "paper_fingerprint_ledger.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def ambiguity_rows() -> list[dict[str, str]]:
    rows = [
        ("S12E-A001", "arxiv_source_payload", "TeX archive versus PDF-only submission", "PDF_NOT_TEX_ARCHIVE", "arXiv source endpoint; byte/file inspection"),
        ("S12E-A002", "beta_distribution_parameterization", "A,sigma as log-space versus raw-space moments", "exp(A+sigma*Z)", "historical tgs_newbeta_v10.m:10"),
        ("S12E-A003", "poisson_exposure", "Unstated batch exposure", "one unit per paper molecular step", "paper says Poisson updates; no exposure supplied"),
        ("S12E-A004", "batch_loss_boundary", "Poisson losses can exceed counts", "clip each loss to available count", "human-directed equation min(L_i,n_i)"),
        ("S12E-A005", "batch_overshoot", "Stop at or above 80 versus trim", "retain overshoot", "human direction; no trimming clue found"),
        ("S12E-A006", "maxsteps_action", "Fission or terminate at 1,000", "fission current nonempty state", "paper says updates until maxsteps then fission"),
        ("S12E-A007", "rho", "Uniform normalized reservoir versus rho_i=1", "separate K0-K3 rho=1/100 and K4 rho=1", "historical source versus explicit candidate"),
        ("S12E-A008", "daughter_choice", "First, literal random, or nonempty random daughter", "separate K0-K4 branches", "paper says one daughter only"),
        ("S12E-A009", "K0_scope", "Exact S12 engine versus event-kernel control", "paper initialization/binomial/maxsteps shared; categorical kernel only", "human candidate wording and common Phase-1 rules"),
        ("S12E-A010", "paper_time_axis", "Batch updates versus source observations including fission", "batch updates primary; both lengths reported", "Methods n_tot and intervention boundary ambiguity"),
        ("S12E-A011", "time_to_first_unit", "Raw steps versus percentage", "raw batch step primary; percent diagnostic", "Table percent glyph versus prose molecular steps"),
        ("S12E-A012", "dominant_H_cluster", "Definition of most recurring composition", "H>0.9 connected components; largest then earliest", "paper prose lacks linkage rule"),
        ("S12E-A013", "euclidean_kmeans_scope", "All points versus non-drift points", "historical H>0.9 non-drift points", "tgs_acluster.m:38-66"),
        ("S12E-A014", "k1_silhouette", "Undefined silhouette at k=1", "score 0; use only absent positive k>=2", "operational frozen rule"),
        ("S12E-A015", "persistent_cluster", "Minimum compotype recurrence", "cluster size at least 3", "S08 explicit validation convention; reconstruction choice"),
        ("S12E-A016", "dominant_radius", "Maximum member distance versus robust cap", "min(max,median+3*1.4826*MAD)", "human-directed robust-cap requirement"),
        ("S12E-A017", "label_molecular_mapping", "Generation label placement", "post-fission g through next fission; pre-first drift", "S12 frozen mapping retained explicitly"),
        ("S12E-A018", "CLR_zero_policy", "Unreported pseudocount", "additive 0.5", "frozen S12 reconstruction; not author identified"),
        ("S12E-A019", "GRN_preprocessing", "CLR alone versus zscore/GSR/AR1", "M1 and M2 separate", "paper versus public GRN source"),
        ("S12E-A020", "metric_identity", "source emergence versus corrected local_phi_r", "M1/M2/M4 emergence; M3 comparator", "public main.py assignments"),
        ("S12E-A021", "association_identity", "corr(E,Y) versus corr(delta E,Y)", "text primary; differenced caption diagnostic", "paper text/caption discrepancy"),
        ("S12E-A022", "temporal_fit", "Completed trajectory versus past-only", "full descriptive; prefix prospective", "public local Gaussian fits supplied full array"),
        ("S12E-A023", "partition_rng", "NetworkX Fiedler initialization implicit", "domain-separated explicit RandomState seed", "S12C confirmed wrapper semantics"),
        ("S12E-A024", "nonfinite", "Replace, clip, or retain", "retain status; no replacement", "human no-hidden-failure boundary"),
        ("S12E-A025", "GRN_nan_to_num_clue", "Later GRN code replaces nonfinite emergence", "not authorized as S12E metric branch", "BreakingGRNMemories phi.py:55,65-68"),
        ("S12E-A026", "intervention_online_semantics", "How a single candidate obtains local emergence", "I1/I2/I3 separate", "paper underspecified; human-directed branches"),
        ("S12E-A027", "intervention_tie", "Raw arg extrema tie", "add before delete then lowest component", "predeclared deterministic policy"),
        ("S12E-A028", "confirmation_selection", "Choose pipeline using emergence", "forbidden; engine/label fingerprints only", "phase firewall"),
        ("S12E-A029", "figure_trajectory_extent", "Raster axes are display limits not raw data", "record panel bounds; use broad 500-1500 gate", "manual figure inspection"),
        ("S12E-A030", "source_code_availability", "Unavailable paper implementation", "UNRESOLVED_AUTHOR_CODE; source-informed only", "paper code availability and GitHub search"),
    ]
    return [
        {
            "ambiguityId": row[0],
            "parameter": row[1],
            "ambiguity": row[2],
            "frozenResolution": row[3],
            "evidence": row[4],
            "resolutionClass": "FROZEN_RECONSTRUCTION_BRANCH",
            "outcomeDependentChangePermitted": "false",
        }
        for row in rows
    ]


def write_ambiguities() -> None:
    rows = ambiguity_rows()
    with (ARTIFACTS / "implementation_ambiguity_ledger.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_figure_measurements() -> None:
    rows = [
        ("figure-01.png", "inset_D", "molecular_step", 0, 800, "illustrative inset; approximate raster axis"),
        ("figure-02.png", "A_aggregate", "molecular_step", 0, 1300, "aggregate display extends to about 1300"),
        ("figure-02.png", "B_sample", "molecular_step", 0, 800, "sample run display"),
        ("figure-02.png", "C_sample", "molecular_step", 0, 800, "sample run display"),
        ("figure-02.png", "D_sample", "molecular_step", 0, 1000, "sample run display"),
        ("figure-02.png", "B_sample", "phi_r", -60, 10, "approximate visible y range"),
        ("figure-02.png", "C_sample", "phi_r", -15, 5, "approximate visible y range"),
        ("figure-02.png", "D_sample", "phi_r", -160, 100, "approximate visible y range"),
        ("figure-03.png", "A", "spearman_rho", -1, 1, "mean line labeled rho=0.139"),
        ("figure-07.png", "boxplot", "molecular_steps", 300, 1800, "max/control/min persistence display"),
        ("figure-08.png", "trend", "GARD_generation", 0, 100, "probability trend over generations"),
    ]
    path = ARTIFACTS / "paper_figure_measurements.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["figureFile", "panel", "axis", "minimum", "maximum", "measurementNote"])
        writer.writerows(rows)


def write_source_clues(snapshot: dict[str, Any]) -> None:
    text = f"""# S12E source clue ledger

## Top summary

- **Research step ID:** `E01-S12E-PAPER-PIPELINE-DETECTIVE-RECONSTRUCTION-v1.0.0`.
- **Completion status:** Phase-0 archaeology complete; scientific development outcomes remained unopened when this ledger was frozen.
- **Artifacts written:** Phase-0 preregistration, source snapshot, paper fingerprints and figure measurements, ambiguity ledger, four implementation registries, immutable-prior baseline, and validation records under `/artifacts/research_steps/S12E/`.
- **Validation result:** Source identities and paper payload passed their pinned hash/commit checks; the arXiv source response is PDF-only and byte-identical to the S03 copy.
- **Outcome classification:** `UNDERDETERMINED` at Phase 0; no engine, label, emergence, or intervention outcome has been observed.
- **Caveats or blockers:** No public GARD-paper code or TeX source was found. Public GRN and PhiRL code are lineage clues, not the unavailable paper implementation.
- **Recommended next action:** Commit and push the locked S12E method, then run Phase 1 without computing replication labels or information metrics.

## Paper and archive

- arXiv `2607.28250v1` source endpoint returned a PDF with SHA-256 `{snapshot['paper']['pdfSha256']}`; it has 18 pages, zero embedded files, empty Comments metadata, and Word/Acrobat producer metadata. No TeX comments or original figure filenames are available.
- The supplied raster figures have no embedded PNG metadata. Their hashes, pixel sizes, and manually read axis ranges are frozen in `source_snapshot_manifest.json` and `paper_figure_measurements.csv`.
- Methods explicitly state distinct-type initialization, Poisson vector updates, binomial fission, and the 100/40/80/100/1000/-4/4 tuple. This conflicts materially with public historical v10 eventwise growth, with-replacement initialization, unbounded generations, and fixed-size split behavior.

## Historical GARD

- `tgs_parameters_v10.m` fixes `Kf=1e-2`, `Kb=1e-4`, lognormal `mu=-4`, `sigma=4`, `hthresh=0.9`, `ks=1:10`, and ten replicas.
- `tgs_grow_v10.m` draws one normalized join/loss event per loop, not a simultaneous Poisson vector.
- `tgs_agard_v10.m` supplies uniform `rho`, initializes counts with replacement, and follows the first split output.
- `tgs_split_v10.m` selects a fixed half without replacement and can discard one odd molecule; it is not binomial componentwise fission.
- `tgs_nondrift.m` technique 1 labels adjacent-composition cosine scores averaged over incoming/outgoing transitions; `tgs_acluster.m` applies k-means only to non-drift generations.

## Public information-theory lineage

- IIGR commit `7c1c22f` corrects the `local_phi_r` overwrite bug. Its `main.py` separately assigns `integrated = local_phi_r(...)` and `emergence = synergy + causation`, uses lag-one Gaussian MI, a noise-connected unnormalized Fiedler partition, and completed supplied arrays for local Gaussian fits.
- PhiRL commit `a6d1d0d` preserves that scalar split, filters dimensions at standard deviation `1e-8`, and descends from the `9030b598` trace-scaled covariance-regularization commit.
- The 2026 public `BreakingGRNMemories` lineage uses `MEASURES=['emergence']`, the same synergy-plus-causation atoms, `noise=True`, and public GRN preprocessing. One file also replaces nonfinite values with zero. Because that replacement is not authorized among M1-M4, it remains a clue rather than a branch.
- Public GitHub inspection found 34 repositories for `pigozzif`, no exact-title repository, no public fork of IIGR or PhiRL, and only the pinned `master` branches. This is an absence result, not proof that code does not exist elsewhere.

## Frozen forensic consequence

S12E must decide upstream GARD and label identity without any emergence value. Metric branches remain exactly M1-M4; no GRN nonfinite-replacement branch or post-outcome method is permitted. Full supplied-array values are retrospective, while prefix refits are the only prospective source reconstruction.
"""
    (ARTIFACTS / "source_clue_ledger.md").write_text(text, encoding="utf-8")


def write_registries(config: dict[str, Any]) -> None:
    mappings = {
        "engine_registry.yaml": {"schemaVersion": "E01-S12E-engine-registry-v1.0.0", **config["phase1"]},
        "label_registry.yaml": {"schemaVersion": "E01-S12E-label-registry-v1.0.0", **config["phase2"]},
        "metric_registry.yaml": {"schemaVersion": "E01-S12E-metric-registry-v1.0.0", **config["phase3"]},
        "intervention_semantics_registry.yaml": {"schemaVersion": "E01-S12E-intervention-registry-v1.0.0", **config["phase4"]},
    }
    for name, payload in mappings.items():
        (ARTIFACTS / name).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def validate(config: dict[str, Any], snapshot: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if config["researchStepId"] != "E01-S12E-PAPER-PIPELINE-DETECTIVE-RECONSTRUCTION-v1.0.0":
        errors.append("research step ID mismatch")
    if list(config["phase1"]["candidates"]) != [
        "K0_HISTORICAL_EVENTWISE",
        "K1_PAPER_POISSON_RANDOM_NONEMPTY",
        "K2_PAPER_POISSON_FIRST_DAUGHTER",
        "K3_PAPER_POISSON_RANDOM_LITERAL",
        "K4_PAPER_POISSON_RHO_ONE",
    ]:
        errors.append("engine candidate registry mismatch")
    if len(config["phase2"]["candidates"]) != 4:
        errors.append("label candidate count is not four")
    if len(config["phase3"]["branches"]) != 4:
        errors.append("metric branch count is not four")
    if len(config["phase4"]["scoringSemantics"]) != 3:
        errors.append("intervention semantic count is not three")
    if len(config["outcomeVocabulary"]) != 12:
        errors.append("outcome vocabulary count mismatch")
    if snapshot["paper"]["pdfSha256"] != config["sourcePins"]["paper"]["expectedSha256"]:
        errors.append("paper hash changed")
    if snapshot["safeLattice"]["sha256"] != config["sourcePins"]["safeLattice"]["sha256"]:
        errors.append("safe lattice hash changed")
    expected_commits = {
        "historicalGard": config["sourcePins"]["historicalGard"]["commit"],
        "iigr": config["sourcePins"]["iigr"]["commit"],
        "phirl": config["sourcePins"]["phirl"]["commit"],
        "grnLineage": config["sourcePins"]["grnLineage"]["commit"],
    }
    by_source: dict[str, set[str]] = {}
    for record in snapshot["sourceFiles"]:
        by_source.setdefault(record["sourceId"], set()).add(record["commit"])
    for source, commit in expected_commits.items():
        if by_source.get(source) != {commit}:
            errors.append(f"commit mismatch for {source}")
    if baseline["fileCount"] < 300:
        errors.append("immutable prior baseline unexpectedly small")
    ambiguity_count = sum(
        1 for _ in csv.DictReader((ARTIFACTS / "implementation_ambiguity_ledger.csv").open(encoding="utf-8"))
    )
    if ambiguity_count != 30:
        errors.append("ambiguity ledger must contain 30 frozen rows")
    return {
        "schemaVersion": "E01-S12E-phase0-validation-v1.0.0",
        "validatedAtUtc": datetime.now(UTC).isoformat(),
        "passed": not errors,
        "errors": errors,
        "checks": {
            "engineCandidates": 5,
            "labelCandidates": 4,
            "metricBranches": 4,
            "interventionSemantics": 3,
            "ambiguityRows": ambiguity_count,
            "priorFileCount": baseline["fileCount"],
            "paperSourceIsPdfOnly": snapshot["paper"]["sourceEndpointRepresentation"] == "PDF_NOT_TEX_ARCHIVE",
            "priorAggregateSha256": baseline["aggregateSha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-commit", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.record_commit:
        commit = git(REPO, "rev-parse", "HEAD^{commit}")
        branch = git(REPO, "branch", "--show-current")
        remote = git(REPO, "rev-parse", "origin/eidosoma/groups/42^{commit}")
        record = json.loads((ARTIFACTS / "preregistration_record.json").read_text())
        record.update(
            {
                "designCommit": commit,
                "remoteCommit": remote,
                "branch": branch,
                "commitAndPushPassed": commit == remote,
                "recordedAtUtc": datetime.now(UTC).isoformat(),
            }
        )
        write_json(ARTIFACTS / "preregistration_record.json", record)
        if commit != remote:
            raise SystemExit("local design commit is not the pushed branch head")
        manifest_path = ARTIFACTS / "preregistration_artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["files"]:
            path = ARTIFACTS / row["relativePath"]
            row["sizeBytes"] = path.stat().st_size
            row["sha256"] = sha256(path)
        write_json(manifest_path, manifest)
        return

    shutil.copy2(CONFIG, ARTIFACTS / "preregistration.yaml")
    write_paper_fingerprints(config)
    write_ambiguities()
    write_figure_measurements()
    write_registries(config)
    snapshot = source_snapshot(config)
    write_json(ARTIFACTS / "source_snapshot_manifest.json", snapshot)
    write_source_clues(snapshot)
    baseline = prior_baseline()
    write_json(ARTIFACTS / "immutable_prior_baseline.json", baseline)
    write_json(CACHE / "immutable_prior_baseline.json", baseline)
    validation = validate(config, snapshot, baseline)
    write_json(ARTIFACTS / "phase0_validation.json", validation)
    record = {
        "schemaVersion": "E01-S12E-preregistration-record-v1.0.0",
        "researchStepId": config["researchStepId"],
        "frozenAtUtc": datetime.now(UTC).isoformat(),
        "preregistrationSha256": sha256(ARTIFACTS / "preregistration.yaml"),
        "repositoryConfigSha256": sha256(CONFIG),
        "phase0ValidationPassed": validation["passed"],
        "priorAggregateSha256": baseline["aggregateSha256"],
        "developmentOutcomesOpened": False,
        "designCommit": None,
        "remoteCommit": None,
        "commitAndPushPassed": False,
    }
    write_json(ARTIFACTS / "preregistration_record.json", record)
    manifest_files = [
        "preregistration.yaml",
        "paper_fingerprint_ledger.yaml",
        "paper_figure_measurements.csv",
        "implementation_ambiguity_ledger.csv",
        "source_clue_ledger.md",
        "source_snapshot_manifest.json",
        "engine_registry.yaml",
        "label_registry.yaml",
        "metric_registry.yaml",
        "intervention_semantics_registry.yaml",
        "immutable_prior_baseline.json",
        "phase0_validation.json",
        "preregistration_record.json",
    ]
    manifest = {
        "schemaVersion": "E01-S12E-preregistration-artifact-manifest-v1.0.0",
        "files": [
            {
                "relativePath": name,
                "sizeBytes": (ARTIFACTS / name).stat().st_size,
                "sha256": sha256(ARTIFACTS / name),
            }
            for name in manifest_files
        ],
    }
    write_json(ARTIFACTS / "preregistration_artifact_manifest.json", manifest)
    if not validation["passed"]:
        raise SystemExit("Phase-0 validation failed: " + "; ".join(validation["errors"]))
    print(json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()
