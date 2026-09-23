from ..types import JobHandler
from ..rethink import run_rethink_job


def _run(job_id: str, worker_id: str):
    return run_rethink_job(job_id)


JOB_HANDLER = JobHandler("rethink", _run)
