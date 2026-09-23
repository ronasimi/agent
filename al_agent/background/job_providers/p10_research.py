from ..types import JobHandler
from ..research import run_research_job

def _run(job_id: str, worker_id: str):
    return run_research_job(job_id, worker_id)

JOB_HANDLER = JobHandler("research", _run)
