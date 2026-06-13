# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Substrate benchmark orchestrator.

Clones a branch of the substrate repo, builds and deploys it to the test
cluster, then submits one Kubernetes Job per entry in tests.yaml that runs
benchmarking/locust/runner.py. Tears down substrate + workloads between
tests so they don't pollute each other.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

import yaml


SUBSTRATE_DIR = "/workspace/substrate"
ENV_FILE = "/opt/automation/ate-dev-env.sh"
RUNNER_JOB_TMPL = "/opt/automation/manifests/runner-job.yaml.tmpl"
NAMESPACE = "monitoring"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="Git URL of substrate repo to clone")
    p.add_argument("--branch", default="main", help="Branch to benchmark")
    p.add_argument(
        "--dest",
        required=True,
        help="Root destination for results (passed through to runner.py --dest)",
    )
    p.add_argument(
        "--tests",
        default="/etc/orchestrator/tests.yaml",
        help="Path to the tests YAML file (mounted from a ConfigMap)",
    )
    return p.parse_args()


def run(cmd, **kwargs):
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def run_no_check(cmd, **kwargs):
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=False, **kwargs)


def source_env(path):
    result = subprocess.run(
        ["bash", "-c", f'set -a; source "{path}"; env'],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ[k] = v


def wait_for_docker(timeout=120):
    print("Waiting for DIND sidecar...", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        r = subprocess.run(["docker", "info"], capture_output=True)
        if r.returncode == 0:
            print("DIND ready.", flush=True)
            return
        time.sleep(2)
    raise RuntimeError("DIND sidecar did not become ready within timeout")


def registry_host(ko_docker_repo):
    return ko_docker_repo.split("/", 1)[0]


def sanitize(name):
    return re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")


def render_template(path, subs, extra_args=()):
    text = Path(path).read_text()
    for k, v in subs.items():
        text = text.replace("${" + k + "}", str(v))
    if not extra_args:
        return text
    docs = list(yaml.safe_load_all(text))
    for doc in docs:
        if doc and doc.get("kind") == "Job":
            doc["spec"]["template"]["spec"]["containers"][0]["args"].extend(
                str(a) for a in extra_args
            )
    return yaml.safe_dump_all(docs)


def parse_duration_seconds(s):
    m = re.fullmatch(r"(\d+)\s*([smh]?)", s.strip())
    if not m:
        raise ValueError(f"unrecognized duration: {s}")
    n = int(m.group(1))
    unit = m.group(2) or "s"
    return n * {"s": 1, "m": 60, "h": 3600}[unit]


def wait_for_no_active_runners(timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        r = subprocess.run(
            [
                "kubectl",
                "get",
                "jobs",
                "-n",
                NAMESPACE,
                "-l",
                "app=substrate-benchmark-runner",
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            print(f"kubectl get jobs failed: {r.stderr}", flush=True)
            time.sleep(5)
            continue
        items = json.loads(r.stdout).get("items", [])
        active = [
            j["metadata"]["name"]
            for j in items
            if j.get("status", {}).get("succeeded", 0) == 0
            and j.get("status", {}).get("failed", 0) == 0
        ]
        if not active:
            return
        print(f"Waiting for in-progress runner jobs: {active}", flush=True)
        time.sleep(10)
    raise RuntimeError(
        f"Existing runner jobs still active after {timeout}s; aborting"
    )


def wait_for_job(name, timeout_seconds):
    start = time.time()
    while time.time() - start < timeout_seconds:
        r = subprocess.run(
            ["kubectl", "get", "job", name, "-n", NAMESPACE, "-o", "json"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            print(f"kubectl get job failed: {r.stderr}", flush=True)
            time.sleep(5)
            continue
        status = json.loads(r.stdout).get("status", {})
        if status.get("succeeded", 0) >= 1:
            return "complete"
        if status.get("failed", 0) >= 1:
            return "failed"
        time.sleep(10)
    return "timeout"


def deploy_substrate():
    run(["hack/install-ate.sh", "--deploy-ate-system"])


def teardown_substrate():
    run_no_check(["hack/install-ate.sh", "--delete-ate-system"])


def deploy_workloads(worker_count=1):
    run(
        [
            "benchmarking/workloads/deploy.sh",
            "--deploy",
            "--worker-count",
            str(worker_count),
        ]
    )
    # Block until ActorTemplates are Ready
    run(
        [
            "kubectl",
            "wait",
            "--for=condition=Ready",
            "--all",
            "actortemplates",
            "-n",
            "benchmark-workloads",
            "--timeout=300s",
        ]
    )


def teardown_workloads():
    run_no_check(["benchmarking/workloads/deploy.sh", "--delete"])


def run_test(test, image, dest, commit):
    name = test["name"]
    job_name = f"runner-{sanitize(name)}-{commit[:7]}-{uuid.uuid4().hex[:6]}"
    subs = {
        "JOB_NAME": job_name,
        "IMAGE": image,
        "TEST_FILE": test["file"],
        "DURATION": test["duration"],
        "USERS": test["users"],
        "TAG": commit,
        "NAME": name,
        "DEST": dest,
    }
    manifest = render_template(RUNNER_JOB_TMPL, subs, test.get("flags", []))
    wait_for_no_active_runners()
    print(f"Submitting Job {job_name}", flush=True)
    subprocess.run(
        ["kubectl", "apply", "-f", "-"], input=manifest, text=True, check=True
    )
    timeout = parse_duration_seconds(test["duration"]) + 1800
    result = wait_for_job(job_name, timeout)
    print(f"Job {job_name} result: {result}", flush=True)
    run_no_check(
        ["kubectl", "logs", f"job/{job_name}", "-n", NAMESPACE, "--tail=500"]
    )
    run_no_check(["kubectl", "delete", "job", job_name, "-n", NAMESPACE])
    return result


def main():
    args = parse_args()
    source_env(ENV_FILE)
    for k in ("PROJECT_ID", "CLUSTER_NAME", "CLUSTER_LOCATION", "KO_DOCKER_REPO"):
        if not os.environ.get(k):
            sys.exit(f"{k} not set after sourcing {ENV_FILE}")

    wait_for_docker()
    run(
        [
            "gcloud",
            "auth",
            "configure-docker",
            registry_host(os.environ["KO_DOCKER_REPO"]),
            "--quiet",
        ]
    )
    run(
        [
            "gcloud",
            "container",
            "clusters",
            "get-credentials",
            os.environ["CLUSTER_NAME"],
            "--location",
            os.environ["CLUSTER_LOCATION"],
            "--project",
            os.environ["PROJECT_ID"],
        ]
    )

    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            args.branch,
            args.repo,
            SUBSTRATE_DIR,
        ]
    )
    os.chdir(SUBSTRATE_DIR)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"Building commit {commit}", flush=True)

    # Substrate images are built and pushed by hack/install-ate.sh via its
    # `run_ko apply` / `run_ko resolve` calls on the ko:// references in
    # manifests/ate-install. No separate `make build-images` step needed.

    locust_image = f"{os.environ['KO_DOCKER_REPO']}/locust-test:{commit}"
    run(
        [
            "docker",
            "build",
            "-t",
            locust_image,
            "-f",
            "benchmarking/locust/Dockerfile",
            "benchmarking/locust/",
        ]
    )
    run(["docker", "push", locust_image])

    tests = yaml.safe_load(Path(args.tests).read_text())["tests"]
    print(f"Running {len(tests)} test(s)", flush=True)

    deploy_substrate()
    deploy_workloads(tests[0].get("workerCount", 1))

    results = []
    for i, test in enumerate(tests):
        is_last = i == len(tests) - 1
        print(f"\n=== test {i + 1}/{len(tests)}: {test['name']} ===", flush=True)
        try:
            status = run_test(test, locust_image, args.dest, commit)
        except Exception as e:
            print(f"Test {test['name']} crashed: {e}", flush=True)
            status = "error"
        results.append((test["name"], status))

        teardown_workloads()
        teardown_substrate()
        if not is_last:
            deploy_substrate()
            deploy_workloads(tests[i + 1].get("workerCount", 1))

    print("\n=== summary ===", flush=True)
    failed = 0
    for name, status in results:
        print(f"  {name}: {status}", flush=True)
        if status != "complete":
            failed += 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
