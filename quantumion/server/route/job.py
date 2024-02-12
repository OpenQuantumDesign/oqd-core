from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi import status as http_status

from rq import Queue
from rq.job import Callback
from rq.job import Job as RQJob

from sqlalchemy import select

########################################################################################

from quantumion.backend.analog.python.qutip import QutipBackend
from quantumion.backend.digital.python.tc import TensorCircuitBackend
from quantumion.backend.task import Task

from quantumion.server.route.auth import user_dependency
from quantumion.server.database import db_dependency, JobInDB
from quantumion.server.jobqueue import (
    redis_client_dependency,
    report_success,
    report_failure,
    report_stopped,
)
from quantumion.server.model import Job

########################################################################################

job_router = APIRouter(tags=["Job"])


@job_router.post("/submit/{backend}", tags=["Job"])
async def submit_job(
    task: Task,
    backend: Literal["qutip", "tensorcircuit"],
    user: user_dependency,
    db: db_dependency,
    redis_client: redis_client_dependency,
):
    backends = {"qutip": QutipBackend(), "tensorcircuit": TensorCircuitBackend()}
    job = Queue(connection=redis_client).enqueue(
        backends[backend].run,
        task,
        on_success=Callback(report_success),
        on_failure=Callback(report_failure),
        on_stopped=Callback(report_stopped),
    )

    job_in_db = JobInDB(
        job_id=job.id,
        task=task.model_dump_json(),
        backend=backend,
        status=job.get_status(),
        result=None,
        user_id=user.user_id,
    )
    db.add(job_in_db)
    await db.commit()
    await db.refresh(job_in_db)

    return Job.model_validate(job_in_db)


@job_router.get("/retrieve/{job_id}", tags=["Job"])
async def retrieve_job(job_id: str, user: user_dependency, db: db_dependency):
    query = await db.execute(
        select(JobInDB).filter(
            JobInDB.job_id == job_id,
            JobInDB.user_id == user.user_id,
        )
    )
    job_in_db = query.scalars().first()
    if job_in_db:
        return Job.model_validate(job_in_db)

    raise HTTPException(status_code=http_status.HTTP_401_UNAUTHORIZED)


@job_router.delete("/cancel/{job_id}", tags=["Job"])
async def cancel_job(
    job_id: str,
    user: user_dependency,
    db: db_dependency,
    redis_client: redis_client_dependency,
):
    query = await db.execute(
        select(JobInDB).filter(
            JobInDB.job_id == job_id,
            JobInDB.user_id == user.user_id,
        )
    )
    job_in_db = query.scalars().first()
    if job_in_db:
        job = RQJob.fetch(id=job_id, connection=redis_client)
        job.cancel()
        setattr(job_in_db, "status", "canceled")
        await db.commit()
        await db.refresh(job_in_db)
        return Job.model_validate(job_in_db)

    raise HTTPException(status_code=http_status.HTTP_401_UNAUTHORIZED)
