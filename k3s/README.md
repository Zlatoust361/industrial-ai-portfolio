# k3s/ — single-node edge deployment (week 12, day 4)

k3s, not full k8s: k3s is Rancher's single-binary, ARM64-native
Kubernetes distribution built specifically for edge/IoT devices — one
process for both control plane and node agent, a bundled lightweight
SQLite datastore instead of etcd, and every add-on (Traefik, ServiceLB,
local-path-provisioner, metrics-server) toggleable at install time. Full
k8s assumes a multi-node cluster with its own etcd quorum, a real load
balancer, and cluster-autoscaling concerns that don't apply to — and
would actively waste memory on — a single Jetson Orin Nano.

**Every manifest here assumes exactly one node, on purpose** — this
pilot's own stated constraint ("если это единственный Jetson в
пилоте"). No `nodeSelector`/pod anti-affinity/topology spread
constraints anywhere: with one node, Kubernetes' default scheduler
already has exactly one place to put every Pod, and adding node-affinity
rules for a cluster that doesn't have a second node to affinitize
against would be complexity with nothing to defend against.

## Install (once, on the device)

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable traefik --disable servicelb --disable metrics-server" sh -
```

`--disable traefik/servicelb/metrics-server`: none of the three do
anything useful for this pilot (no ingress routing beyond two NodePort
Services, no multi-node load balancing, no autoscaling decisions being
made) and each is a real standing memory cost on a device that's already
close to its budget — see "Memory: does k3s itself fit?" below.
`local-path-provisioner` stays enabled (default) — both PVCs here need
it.

**GPU access**: `vlm-inference-deployment.yaml` sets
`runtimeClassName: nvidia`, which requires containerd's NVIDIA runtime
configured as a k3s `RuntimeClass` on the device (the k3s/Jetson
equivalent of `docker-compose.jetson.yml`'s `runtime: nvidia`) — set up
once, separately, outside these manifests. **Not verified in this
sandbox** (no k3s, no physical device, no `kubectl` — see "Honesty
note" below); confirm the exact containerd config against the device's
actual JetPack/k3s version before a first real deploy.

## Apply

```bash
kubectl apply -k k3s/
kubectl -n vlm-edge rollout status deployment/vlm-inference --timeout=180s
kubectl -n vlm-edge rollout status deployment/chat-backend --timeout=60s
```

CI (`.github/workflows/jetson-cicd.yml`) does the same, after pointing
`k3s/kustomization.yaml`'s `images:` at the git-SHA-tagged images it
just built.

## Why `strategy: Recreate` on both Deployments, not the k8s default

The Deployment default (`RollingUpdate`) briefly runs the OLD and NEW
Pod side by side (`maxSurge: 25%` by default) so traffic never drops —
correct and desirable on a real multi-node cluster, where the surge Pod
lands on a *different* node with its own spare capacity. On this
single-node pilot there is no different node: a surge Pod schedules
onto the exact same 8 GiB machine already running the old one.
`vlm-inference` alone requests/limits 6144Mi; surging it would ask this
device to fit ~12 GiB of inference containers into 8 GiB at once, which
does not degrade gracefully — it OOMs. `Recreate` terminates the old Pod
completely before starting the new one: a few seconds of real downtime
on every deploy, traded deliberately for never double-booking memory
this device doesn't have. Applied to `chat-backend` too, even though its
own footprint is small — the underlying reason (no second node to
surge onto) applies to every workload here, not just the memory-heavy
one.

## Health checks

Both Deployments use `startupProbe` (not a long `initialDelaySeconds` on
`livenessProbe` — the correct k8s primitive for a slow-starting
container) followed by `readinessProbe`/`livenessProbe`, all hitting the
same `GET /health` each service already exposes
(`src/api/app.py`/`assistant/chat_app.py`). `vlm-inference` gets a
120s startup budget (24 × 5s) — its first real request loads and runs
the CLIP checkpoint, the same reason `docker-compose.jetson.yml`'s own
healthcheck gives it a 90s `start_period`. `chat-backend` gets 20s (10 ×
2s) — no model to load.

`kubectl rollout status` blocks until the new Pod's `readinessProbe`
succeeds (or the given `--timeout` elapses) — this is what CI's deploy
job waits on before declaring a deploy successful, and what fails the CI
job (triggering rollback, below) if the new image never becomes healthy.

## Rollback

```bash
kubectl -n vlm-edge rollout undo deployment/vlm-inference
kubectl -n vlm-edge rollout undo deployment/chat-backend
```

Built into every Deployment — no extra tooling (no Argo Rollouts, no
Flagger) needed for a single-node pilot. `revisionHistoryLimit: 5` on
both keeps the last 5 ReplicaSets (and their Pod template, including the
image tag) around for `rollout undo` to revert to. This only works
because every deploy references an **immutable, git-SHA-tagged** image
(`k3s/kustomization.yaml`'s `images:` list, set by CI per-deploy) — if
every deploy instead pushed to a mutable `:latest` tag, `rollout undo`
would just re-apply the same tag pointing at the same (new, broken)
image, reverting nothing. CI (below) calls `rollout undo` automatically
when `rollout status` times out, so a bad deploy self-heals back to the
last-known-good image without a human needing to intervene at 2am for a
single-device pilot with no on-call rotation watching it.

## Memory: does k3s itself fit?

`docs/architecture.md`'s combined-footprint table (week 12 day 2) found
**6591.8 MB used / 8192 MB total**, leaving **~1600 MB** for the OS and
everything else — that number did NOT include k3s's own control-plane
process (k3s combines server + agent into one process even on a
single-node install, so there's no separate "control plane node" to
absorb this cost elsewhere). k3s is real and lighter than a full
kube-apiserver/etcd/controller-manager/scheduler stack, but it is not
free — widely-cited community figures put a minimal single-node k3s
server (traefik/servicelb/metrics-server disabled, as installed above)
in the **low-to-mid hundreds of MB** RSS range. This repo has no k3s
installation to measure that number against directly (no k3s, no
Jetson, no `kubectl` in this sandbox — see "Honesty note" below), so it
is stated here as a directional, sourced-from-community-reports figure,
not a number this repo measured itself the way every other row in
`docs/architecture.md`'s table was. **Open risk, explicitly carried
forward to day 5**: verify k3s's real RSS on the actual device before
trusting the ~1600 MB headroom still holds once k3s itself is added to
the sum.

## Honesty note: what's real here, what isn't verified

Every manifest below is written to match the real Kubernetes/k3s API
schema (`apiVersion`/`kind` combinations, field names, structure) by
careful hand-authoring against the same schema `kubectl`/the k3s API
server itself validates against — but this sandbox has **no `kubectl`,
no k3s installation, and no physical Jetson**, so nothing here has been
validated against a real API server (no `kubectl apply --dry-run=server`,
no `kubectl kustomize`, no real rollout ever executed). What WAS
verified for real: every file parses as valid YAML
(`scripts/validate_k3s_manifests.py`, `make validate-k3s-manifests` —
real `yaml.safe_load` plus structural checks: required
`apiVersion`/`kind`/`metadata.name` fields present, container images
reference the two real Dockerfiles' image names, probe paths match the
two services' real `/health` endpoints). This is the same
"verified-to-a-real-boundary, not beyond" honesty this repo has applied
to every hardware/network-gated artifact all session — `docker buildx
build --check` for both Dockerfiles reaches the exact same
network-blocked `FROM` boundary week 08's `Dockerfile.jetson` already
established (this sandbox's proxy blocks layer downloads from every
registry, `docker.io` included, not just `nvcr.io`).
