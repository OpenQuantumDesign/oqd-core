import os

from fastapi import FastAPI

from redis import Redis
from rq import Queue
from rq.job import Job

########################################################################################

from backends.qsim.qutip import QutipBackend
from backends.base import Submission, Specification, Result, run

########################################################################################

redis_client = Redis(host=os.environ["REDIS_HOST"], port=6379, decode_responses=False)
queue = Queue(connection=redis_client)

########################################################################################

app = FastAPI()


@app.post("/qsim_simulator/")
async def submit(submission: Submission):
    print(f"Queueing {submission} on server backend. {len(queue)} jobs in queue.")
    job = queue.enqueue(run, submission)
    return {"id": job.id, "status": job.get_status()}


@app.post("/check_status/")
async def check_status(request: dict):
    print(f"Requesting status of job {request['id']}")
    job = Job.fetch(id=request["id"], connection=redis_client)
    return {"id": job.id, "status": job.get_status()}


@app.post("/get_result/")
async def get_result(request: dict):
    print(f"Requesting result for job {request['id']}")
    job = Job.fetch(id=request["id"], connection=redis_client)
    return job.return_value()
