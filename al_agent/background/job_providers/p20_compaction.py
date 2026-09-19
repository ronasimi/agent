from ..handlers import JobHandler
from ..maintenance import run_context_compaction_job

def _run(job_id: str, worker_id: str):
    return run_context_compaction_job(job_id)

JOB_HANDLER = JobHandler("context_compaction", _run)
