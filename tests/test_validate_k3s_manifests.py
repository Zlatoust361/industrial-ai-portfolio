"""Tests for `scripts.validate_k3s_manifests` (week 12, day 4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validate_k3s_manifests import (
    check_deployment_resources,
    check_health_probes,
    check_pvc_claims_are_defined,
    check_recreate_strategy,
    check_required_fields,
    check_service_selectors_match_a_deployment,
    load_manifests,
    validate,
)


def test_validate_passes_against_the_real_k3s_directory() -> None:
    assert validate("k3s") == []


def test_load_manifests_finds_every_yaml_file_in_the_real_directory() -> None:
    manifests = load_manifests("k3s")
    assert "namespace.yaml" in manifests
    assert "vlm-inference-deployment.yaml" in manifests
    assert "chat-backend-deployment.yaml" in manifests


def test_load_manifests_missing_directory_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_manifests("k3s/does-not-exist")


def _write(tmp_path: Path, filename: str, content: str) -> None:
    (tmp_path / filename).write_text(content)


_DEPLOYMENT = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
spec:
  strategy:
    type: {strategy}
  template:
    metadata:
      labels:
        app: {name}
    spec:
      containers:
        - name: {name}
          image: {image}
          resources:
            requests:
              memory: "128Mi"
            limits:
              memory: "256Mi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
"""

_SERVICE = """
apiVersion: v1
kind: Service
metadata:
  name: {name}
spec:
  selector:
    app: {selector_app}
"""


def test_check_required_fields_catches_missing_kind(tmp_path: Path) -> None:
    _write(tmp_path, "bad.yaml", "apiVersion: v1\nmetadata:\n  name: x\n")
    manifests = load_manifests(tmp_path)
    errors = check_required_fields(manifests)
    assert any("kind" in e for e in errors)


def test_check_required_fields_catches_missing_metadata_name(tmp_path: Path) -> None:
    _write(tmp_path, "bad.yaml", "apiVersion: v1\nkind: Service\nmetadata: {}\n")
    manifests = load_manifests(tmp_path)
    errors = check_required_fields(manifests)
    assert any("metadata.name" in e for e in errors)


def test_check_required_fields_kustomization_is_exempt_from_metadata_name(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "kustomization.yaml",
        "apiVersion: kustomize.config.k8s.io/v1beta1\nkind: Kustomization\nresources: []\n",
    )
    manifests = load_manifests(tmp_path)
    assert check_required_fields(manifests) == []


def test_check_deployment_resources_catches_missing_memory_limit(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "d.yaml",
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: x\n"
        "spec:\n  template:\n    metadata: {}\n    spec:\n      containers:\n"
        "        - name: x\n          image: x:latest\n          resources:\n"
        "            requests:\n              memory: 1Mi\n",
    )
    manifests = load_manifests(tmp_path)
    errors = check_deployment_resources(manifests)
    assert any("resources.limits.memory" in e for e in errors)


def test_check_deployment_resources_catches_missing_image(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "d.yaml",
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: x\n"
        "spec:\n  template:\n    metadata: {}\n    spec:\n      containers:\n"
        "        - name: x\n          resources:\n"
        "            requests: {memory: 1Mi}\n            limits: {memory: 1Mi}\n",
    )
    manifests = load_manifests(tmp_path)
    errors = check_deployment_resources(manifests)
    assert any("no image set" in e for e in errors)


def test_check_recreate_strategy_catches_rolling_update(tmp_path: Path) -> None:
    _write(tmp_path, "d.yaml", _DEPLOYMENT.format(name="x", strategy="RollingUpdate", image="x:1"))
    manifests = load_manifests(tmp_path)
    errors = check_recreate_strategy(manifests)
    assert any("RollingUpdate" in e for e in errors)


def test_check_recreate_strategy_accepts_recreate(tmp_path: Path) -> None:
    _write(tmp_path, "d.yaml", _DEPLOYMENT.format(name="x", strategy="Recreate", image="x:1"))
    manifests = load_manifests(tmp_path)
    assert check_recreate_strategy(manifests) == []


def test_check_health_probes_catches_missing_readiness_probe(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "d.yaml",
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: x\n"
        "spec:\n  template:\n    metadata: {}\n    spec:\n      containers:\n"
        "        - name: x\n          image: x:1\n          livenessProbe:\n"
        "            httpGet:\n              path: /health\n",
    )
    manifests = load_manifests(tmp_path)
    errors = check_health_probes(manifests)
    assert any("readinessProbe" in e for e in errors)


def test_check_service_selectors_match_a_deployment_catches_mismatch(tmp_path: Path) -> None:
    _write(tmp_path, "d.yaml", _DEPLOYMENT.format(name="app-a", strategy="Recreate", image="x:1"))
    _write(tmp_path, "s.yaml", _SERVICE.format(name="app-a-svc", selector_app="app-b-typo"))
    manifests = load_manifests(tmp_path)
    errors = check_service_selectors_match_a_deployment(manifests)
    assert any("matches no Deployment" in e for e in errors)


def test_check_service_selectors_match_a_deployment_accepts_a_real_match(tmp_path: Path) -> None:
    _write(tmp_path, "d.yaml", _DEPLOYMENT.format(name="app-a", strategy="Recreate", image="x:1"))
    _write(tmp_path, "s.yaml", _SERVICE.format(name="app-a-svc", selector_app="app-a"))
    manifests = load_manifests(tmp_path)
    assert check_service_selectors_match_a_deployment(manifests) == []


def test_check_service_selectors_catches_missing_selector(tmp_path: Path) -> None:
    _write(tmp_path, "s.yaml", "apiVersion: v1\nkind: Service\nmetadata:\n  name: x\nspec: {}\n")
    manifests = load_manifests(tmp_path)
    errors = check_service_selectors_match_a_deployment(manifests)
    assert any("no selector" in e for e in errors)


def test_check_pvc_claims_are_defined_catches_dangling_reference(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "d.yaml",
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: x\n"
        "spec:\n  template:\n    metadata: {}\n    spec:\n      containers: []\n"
        "      volumes:\n        - name: data\n          persistentVolumeClaim:\n"
        "              claimName: nonexistent-pvc\n",
    )
    manifests = load_manifests(tmp_path)
    errors = check_pvc_claims_are_defined(manifests)
    assert any("nonexistent-pvc" in e for e in errors)


def test_check_pvc_claims_are_defined_accepts_a_real_reference(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "pvc.yaml",
        "apiVersion: v1\nkind: PersistentVolumeClaim\nmetadata:\n  name: my-pvc\nspec: {}\n",
    )
    _write(
        tmp_path,
        "d.yaml",
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: x\n"
        "spec:\n  template:\n    metadata: {}\n    spec:\n      containers: []\n"
        "      volumes:\n        - name: data\n          persistentVolumeClaim:\n"
        "              claimName: my-pvc\n",
    )
    manifests = load_manifests(tmp_path)
    assert check_pvc_claims_are_defined(manifests) == []


def test_validate_aggregates_errors_from_every_check(tmp_path: Path) -> None:
    _write(tmp_path, "bad.yaml", "apiVersion: v1\nmetadata:\n  name: x\n")  # missing kind
    errors = validate(tmp_path)
    assert len(errors) >= 1
