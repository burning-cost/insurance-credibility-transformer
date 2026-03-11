"""Upload project to Databricks and run tests via a notebook job."""

import os
import time
import base64
import pathlib
import uuid

# Load credentials
with open(os.path.expanduser("~/.config/burning-cost/databricks.env")) as f:
    for line in f:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()
host = os.environ["DATABRICKS_HOST"].rstrip("/")
print(f"Connected to: {host}")

project_dir = pathlib.Path("/home/ralph/repos/insurance-credibility-transformer")
src_pkg = project_dir / "src" / "insurance_credibility_transformer"
test_dir_path = project_dir / "tests"


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def b64_encode(content):
    return base64.b64encode(content.encode("utf-8")).decode("ascii")


src_files = {}
for f in sorted(src_pkg.glob("*.py")):
    src_files[f.name] = b64_encode(read_file(f))

test_files = {}
for f in sorted(test_dir_path.glob("*.py")):
    test_files[f.name] = b64_encode(read_file(f))

print(f"Source files: {len(src_files)}, test files: {len(test_files)}")

run_uid = uuid.uuid4().hex[:8]

write_lines = [
    "import os, base64, sys, subprocess",
    f"BASE = '/tmp/ict_{run_uid}'",
    "pkg_dir = BASE + '/src/insurance_credibility_transformer'",
    "tst_dir = BASE + '/tests'",
    "os.makedirs(pkg_dir, exist_ok=True)",
    "os.makedirs(tst_dir, exist_ok=True)",
]
for fname, b64 in src_files.items():
    write_lines.append(
        f"open(os.path.join(pkg_dir, '{fname}'), 'w').write(base64.b64decode('{b64}').decode())"
    )
for fname, b64 in test_files.items():
    write_lines.append(
        f"open(os.path.join(tst_dir, '{fname}'), 'w').write(base64.b64decode('{b64}').decode())"
    )

write_block = "\n".join(write_lines)

notebook_source = f"""# Databricks notebook source
# MAGIC %md # insurance-credibility-transformer v0.1.0 - Test Suite v6

# COMMAND ----------

# MAGIC %pip install --quiet torch numpy polars pytest

# COMMAND ----------

# Write source and test files (embedded as base64)
{write_block}

sys.path.insert(0, BASE + '/src')
import torch
print(f"PyTorch: {{torch.__version__}}, CUDA: {{torch.cuda.is_available()}}")

# COMMAND ----------

all_output = []

# Unit tests
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     *[BASE + '/tests/' + f for f in [
         "test_tokenizer.py", "test_attention.py", "test_loss.py",
         "test_datasets.py", "test_retrieval.py", "test_transformer.py",
         "test_icl.py",
     ]],
     "-v", "--tb=short", "--no-header",
    ],
    capture_output=True, text=True, timeout=300,
    env={{**os.environ, "PYTHONPATH": BASE + '/src'}},
)
all_output.append("=== UNIT TESTS ===")
all_output.append(result.stdout)
if result.stderr:
    all_output.append("STDERR: " + result.stderr[-500:])
unit_passed = result.returncode == 0
all_output.append(f"Unit tests: {{'PASSED' if unit_passed else 'FAILED'}} (rc={{result.returncode}})")

# Integration tests
result2 = subprocess.run(
    [sys.executable, "-m", "pytest",
     BASE + '/tests/test_integration.py',
     "-v", "--tb=short", "--no-header",
    ],
    capture_output=True, text=True, timeout=600,
    env={{**os.environ, "PYTHONPATH": BASE + '/src'}},
)
all_output.append("=== INTEGRATION TESTS ===")
all_output.append(result2.stdout)
if result2.stderr:
    all_output.append("STDERR: " + result2.stderr[-500:])
integ_passed = result2.returncode == 0
all_output.append(f"Integration tests: {{'PASSED' if integ_passed else 'FAILED'}} (rc={{result2.returncode}})")

full_output = "\\n".join(all_output)
if len(full_output) > 10000:
    full_output = full_output[-10000:]

dbutils.notebook.exit(full_output)
"""

notebook_path = "/Workspace/Users/pricing.frontier@gmail.com/ict-test-runner-v6"
print(f"Uploading notebook to {notebook_path} ...")
notebook_b64 = base64.b64encode(notebook_source.encode()).decode()

w.workspace.import_(
    path=notebook_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=notebook_b64,
    overwrite=True,
)
print("Notebook uploaded")

print("Submitting serverless job...")
run = w.jobs.submit(
    run_name="ict-test-suite-v6",
    tasks=[
        jobs.SubmitTask(
            task_key="run-tests",
            notebook_task=jobs.NotebookTask(
                notebook_path=notebook_path,
            ),
        )
    ],
)
run_id = run.run_id
print(f"Run ID: {run_id}")
print(f"View at: {host}/jobs/runs/{run_id}")

print("Waiting for completion...")
for i in range(90):
    time.sleep(10)
    run_state = w.jobs.get_run(run_id=run_id)
    lc = str(run_state.state.life_cycle_state) if run_state.state else "?"
    rs = str(run_state.state.result_state) if run_state.state else ""
    print(f"  [{i*10:3d}s] {lc} {rs}")

    if "TERMINATED" in lc or "INTERNAL_ERROR" in lc:
        for task in run_state.tasks or []:
            if task.run_id:
                try:
                    out = w.jobs.get_run_output(run_id=task.run_id)
                    if out.notebook_output and out.notebook_output.result:
                        print("\n--- Test Output ---")
                        print(out.notebook_output.result)
                    if out.error:
                        print(f"\n--- Error ---\n{out.error[:1000]}")
                    if out.error_trace:
                        print(f"\n--- Trace ---\n{out.error_trace[-2000:]}")
                except Exception as e:
                    print(f"Could not get output: {e}")

        if "SUCCESS" in rs:
            print("\nALL TESTS PASSED on Databricks!")
        else:
            print(f"\nFAILED: {rs}")
        break
else:
    print("Timed out after 15 minutes")
