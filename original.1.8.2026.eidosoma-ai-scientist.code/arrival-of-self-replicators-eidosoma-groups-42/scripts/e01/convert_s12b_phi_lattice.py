#!/usr/bin/env python3
"""Audit and convert the pinned PhiID lattice in a disposable process.

The raw pickle is never consumed by S12B scientific code.  This converter must
be invoked with ``python -I``.  It first inspects every opcode, then uses a
restricted unpickler with a three-global allowlist, validates the graph, and
writes inert JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import pickletools
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx

ALLOWED_GLOBALS = {
    ("builtins", "dict"): dict,
    ("networkx.classes.digraph", "DiGraph"): nx.DiGraph,
    ("networkx.classes.reportviews", "NodeView"): nx.classes.reportviews.NodeView,
}
EXPECTED_STRINGS = {
    "networkx.classes.digraph",
    "DiGraph",
    "builtins",
    "dict",
    "networkx.classes.reportviews",
    "NodeView",
    "graph_attr_dict_factory",
    "node_dict_factory",
    "node_attr_dict_factory",
    "adjlist_outer_dict_factory",
    "adjlist_inner_dict_factory",
    "edge_attr_dict_factory",
    "graph",
    "_node",
    "_adj",
    "_pred",
    "_succ",
    "nodes",
    "_nodes",
    "descendants",
}
BOTTOM = (((0,), (1,)), ((0,), (1,)))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RestrictedLatticeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return ALLOWED_GLOBALS[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(f"forbidden global {module}.{name}") from exc


def jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def inspect_pickle(data: bytes) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    strings: set[str] = set()
    sensitive: list[dict[str, Any]] = []
    protocols: list[int] = []
    for opcode, argument, position in pickletools.genops(data):
        counts[opcode.name] += 1
        if opcode.name == "PROTO":
            protocols.append(int(argument))
        if opcode.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}:
            strings.add(str(argument))
        if opcode.name in {
            "GLOBAL",
            "STACK_GLOBAL",
            "REDUCE",
            "BUILD",
            "OBJ",
            "INST",
            "NEWOBJ",
            "NEWOBJ_EX",
            "EXT1",
            "EXT2",
            "EXT4",
            "PERSID",
            "BINPERSID",
        }:
            sensitive.append(
                {"position": position, "opcode": opcode.name, "argument": repr(argument)}
            )
    unexpected_strings = sorted(strings - EXPECTED_STRINGS)
    if unexpected_strings:
        raise ValueError(f"unexpected pickle strings: {unexpected_strings}")
    if counts["REDUCE"] or counts["OBJ"] or counts["INST"] or counts["NEWOBJ_EX"]:
        raise ValueError("disallowed executable pickle opcode present")
    return {
        "protocols": protocols,
        "opcodeCounts": dict(sorted(counts.items())),
        "sensitiveOpcodes": sensitive,
        "unicodeStrings": sorted(strings),
        "unexpectedStrings": unexpected_strings,
        "restrictedGlobalAllowlist": [".".join(item) for item in sorted(ALLOWED_GLOBALS)],
    }


def convert(inputs: list[Path], output: Path) -> dict[str, Any]:
    if not getattr(sys.flags, "isolated", 0):
        raise RuntimeError("converter must be run with python -I")
    raw = [path.read_bytes() for path in inputs]
    hashes = [sha256_bytes(item) for item in raw]
    if len(set(hashes)) != 1:
        raise ValueError("pinned lattice files are not byte-identical")
    audit = inspect_pickle(raw[0])
    graph = RestrictedLatticeUnpickler(io.BytesIO(raw[0])).load()
    if not isinstance(graph, nx.DiGraph):
        raise TypeError("lattice is not a networkx.DiGraph")
    if len(graph) != 16 or BOTTOM not in graph:
        raise ValueError("unexpected PhiID lattice cardinality or bottom atom")
    for atom, attributes in graph.nodes(data=True):
        if set(attributes) != {"descendants"}:
            raise ValueError(f"unexpected node attributes for {atom!r}")
        if not isinstance(attributes["descendants"], set):
            raise TypeError("descendants attribute must be a set")
        if not attributes["descendants"].issubset(set(graph.nodes)):
            raise ValueError("descendant references an unknown atom")
    distances = dict(nx.shortest_path_length(graph, target=BOTTOM))
    order: list[Any] = []
    for distance in range(max(distances.values()) + 1):
        order.extend(atom for atom in distances if distances[atom] == distance)
    if order[0] != BOTTOM or set(order) != set(graph.nodes):
        raise ValueError("invalid source Mobius order")
    payload = {
        "schema": "eidosoma.e01.s12b.safe_phi_lattice.v1",
        "researchStepId": "S12B",
        "sourceRelationship": "SOURCE_INFORMED_RECONSTRUCTION",
        "rawPickleSha256": hashes[0],
        "rawSources": [
            {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
            for path, digest in zip(inputs, hashes, strict=True)
        ],
        "pickleAudit": audit,
        "conversionIsolation": {
            "pythonIsolatedFlag": True,
            "restrictedUnpickler": True,
            "rawPicklePermittedInScientificRunner": False,
        },
        "directed": True,
        "nodeCount": len(graph),
        "edgeCount": graph.number_of_edges(),
        "order": jsonable(order),
        "nodes": [
            {
                "atom": jsonable(atom),
                "descendants": jsonable(graph.nodes[atom]["descendants"]),
            }
            for atom in order
        ],
        "edges": [
            {"source": jsonable(source), "target": jsonable(target)}
            for source, target in sorted(graph.edges, key=lambda edge: (repr(edge[0]), repr(edge[1])))
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = convert(args.input, args.output)
    print(json.dumps({"success": True, "nodeCount": payload["nodeCount"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
