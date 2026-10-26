"""Structural validation for k3s/*.yaml -- no kubectl/k3s/API server here (week 12, day 4).

This sandbox has no `kubectl`, no k3s installation, and no physical
Jetson (see `k3s/README.md`'s own honesty note), so `kubectl apply
--dry-run=server`/`kubectl kustomize` aren't options. This script is the
real check that IS possible without any of that: every file parses as
valid YAML, every Kubernetes object has the required
`apiVersion`/`kind`/`metadata.name` triple, and a handful of
cross-manifest consistency checks that catch the exact class of bug a
hand-authored set of Deployment/Service/PVC files is prone to --
a Service whose `selector` doesn't match any Deployment's Pod labels
(traffic would go nowhere), or a Deployment's `volumes[].persistentVolumeClaim.claimName`
that doesn't name a PVC actually defined anywhere in the directory
(the Pod would never leave `Pending`). Neither of those is a YAML-syntax
error -- both are exactly the kind of mistake `yaml.safe_load` alone
can't catch, and exactly the kind `kubectl apply --dry-run=server` WOULD
catch if it were available here.

Usage:
    PYTHONPATH=. python scripts/validate_k3s_manifests.py [--dir k3s]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REQUIRED_TOP_LEVEL_FIELDS = ("apiVersion", "kind")


class ManifestValidationError(Exception):
    """Raised for a structural problem found in one or more manifests."""


def load_manifests(directory: str | Path) -> dict[str, list[dict]]:
    """Parse every `*.yaml` file in `directory` as one or more YAML documents.

    Args:
        directory: Path to a directory of Kubernetes manifest files.

    Returns:
        `{filename: [doc, ...]}` -- a file with multiple `---`-separated
        documents maps to a list with more than one entry.

    Raises:
        FileNotFoundError: If `directory` does not exist.
        yaml.YAMLError: If any file isn't valid YAML.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Manifest directory not found: {dir_path}")

    manifests: dict[str, list[dict]] = {}
    for path in sorted(dir_path.glob("*.yaml")):
        docs = [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc]
        manifests[path.name] = docs
    return manifests


def check_required_fields(manifests: dict[str, list[dict]]) -> list[str]:
    """Every object has apiVersion/kind/metadata.name (Kustomization is exempt from the last one).

    Args:
        manifests: From `load_manifests`.

    Returns:
        Human-readable error strings, one per missing field found (empty if none).
    """
    errors = []
    for filename, docs in manifests.items():
        for doc in docs:
            for field in REQUIRED_TOP_LEVEL_FIELDS:
                if field not in doc:
                    errors.append(f"{filename}: missing required field {field!r}.")
            if doc.get("kind") != "Kustomization" and "name" not in doc.get("metadata", {}):
                errors.append(f"{filename}: missing metadata.name.")
    return errors


def _all_objects(manifests: dict[str, list[dict]]) -> list[dict]:
    return [doc for docs in manifests.values() for doc in docs]


def check_deployment_resources(manifests: dict[str, list[dict]]) -> list[str]:
    """Every container in every Deployment sets `resources.requests`/`limits.memory` and an image.

    On a single, memory-constrained edge device, an unset memory request/
    limit is a real deploy-time risk (the scheduler has no idea how much
    RAM a Pod needs, and nothing stops it from consuming the whole
    device) -- worth catching here, not just at "the Pod OOMed in
    production."
    """
    errors = []
    for doc in _all_objects(manifests):
        if doc.get("kind") != "Deployment":
            continue
        name = doc["metadata"]["name"]
        containers = doc["spec"]["template"]["spec"].get("containers", [])
        if not containers:
            errors.append(f"Deployment {name!r}: no containers defined.")
        for container in containers:
            if not container.get("image"):
                errors.append(f"Deployment {name!r}/{container.get('name')}: no image set.")
            resources = container.get("resources", {})
            if "memory" not in resources.get("requests", {}):
                errors.append(
                    f"Deployment {name!r}/{container.get('name')}: no resources.requests.memory."
                )
            if "memory" not in resources.get("limits", {}):
                errors.append(
                    f"Deployment {name!r}/{container.get('name')}: no resources.limits.memory."
                )
    return errors


def check_recreate_strategy(manifests: dict[str, list[dict]]) -> list[str]:
    """Every Deployment uses `strategy.type: Recreate` (see k3s/README.md's rationale)."""
    errors = []
    for doc in _all_objects(manifests):
        if doc.get("kind") != "Deployment":
            continue
        name = doc["metadata"]["name"]
        strategy_type = doc["spec"].get("strategy", {}).get("type")
        if strategy_type != "Recreate":
            errors.append(
                f"Deployment {name!r}: strategy.type is {strategy_type!r}, expected "
                "'Recreate' (RollingUpdate can double-book memory on a single-node "
                "device -- see k3s/README.md)."
            )
    return errors


def check_health_probes(manifests: dict[str, list[dict]]) -> list[str]:
    """Every container in every Deployment has readinessProbe/livenessProbe httpGet paths set."""
    errors = []
    for doc in _all_objects(manifests):
        if doc.get("kind") != "Deployment":
            continue
        name = doc["metadata"]["name"]
        for container in doc["spec"]["template"]["spec"].get("containers", []):
            for probe_name in ("readinessProbe", "livenessProbe"):
                probe = container.get(probe_name)
                if probe is None or "httpGet" not in probe or not probe["httpGet"].get("path"):
                    errors.append(
                        f"Deployment {name!r}/{container.get('name')}: missing "
                        f"{probe_name}.httpGet.path."
                    )
    return errors


def check_service_selectors_match_a_deployment(manifests: dict[str, list[dict]]) -> list[str]:
    """Every Service's `selector` matches at least one Deployment's Pod labels.

    A Service whose selector matches nothing routes traffic to zero
    endpoints -- a silent, hard-to-debug misconfiguration that
    `yaml.safe_load` alone would never catch.
    """
    deployments = [doc for doc in _all_objects(manifests) if doc.get("kind") == "Deployment"]
    pod_label_sets = [
        set(doc["spec"]["template"]["metadata"].get("labels", {}).items()) for doc in deployments
    ]

    errors = []
    for doc in _all_objects(manifests):
        if doc.get("kind") != "Service":
            continue
        name = doc["metadata"]["name"]
        selector = set(doc["spec"].get("selector", {}).items())
        if not selector:
            errors.append(f"Service {name!r}: no selector set.")
            continue
        if not any(selector <= labels for labels in pod_label_sets):
            errors.append(
                f"Service {name!r}: selector {dict(selector)} matches no Deployment's Pod labels."
            )
    return errors


def check_pvc_claims_are_defined(manifests: dict[str, list[dict]]) -> list[str]:
    """Every Deployment volume's `persistentVolumeClaim.claimName` names a PVC defined somewhere.

    A Deployment referencing a PVC that doesn't exist schedules a Pod
    that sits in `Pending` forever, waiting on a volume that will never
    appear -- a real, common hand-authoring mistake this catches without
    needing a real cluster.
    """
    pvc_names = {
        doc["metadata"]["name"]
        for doc in _all_objects(manifests)
        if doc.get("kind") == "PersistentVolumeClaim"
    }

    errors = []
    for doc in _all_objects(manifests):
        if doc.get("kind") != "Deployment":
            continue
        name = doc["metadata"]["name"]
        for volume in doc["spec"]["template"]["spec"].get("volumes", []):
            claim = volume.get("persistentVolumeClaim", {}).get("claimName")
            if claim and claim not in pvc_names:
                errors.append(
                    f"Deployment {name!r}: volume {volume.get('name')!r} references "
                    f"PVC {claim!r}, which isn't defined in this directory."
                )
    return errors


ALL_CHECKS = (
    check_required_fields,
    check_deployment_resources,
    check_recreate_strategy,
    check_health_probes,
    check_service_selectors_match_a_deployment,
    check_pvc_claims_are_defined,
)


def validate(directory: str | Path) -> list[str]:
    """Run every check in `ALL_CHECKS` against `directory`'s manifests.

    Args:
        directory: Path to a directory of Kubernetes manifest files.

    Returns:
        Every error string found, across all checks (empty if none).
    """
    manifests = load_manifests(directory)
    errors: list[str] = []
    for check in ALL_CHECKS:
        errors.extend(check(manifests))
    return errors


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for `validate_k3s_manifests.py`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="k3s")
    return parser.parse_args()


def main() -> None:
    """Validate `args.dir`'s manifests, print results, exit non-zero on any error."""
    args = parse_args()
    errors = validate(args.dir)

    if errors:
        print(f"{len(errors)} problem(s) found in '{args.dir}':")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    manifest_count = sum(len(docs) for docs in load_manifests(args.dir).values())
    print(f"All checks passed: {manifest_count} Kubernetes object(s) in '{args.dir}'.")


if __name__ == "__main__":
    main()
