import os

from typing import Annotated

from redis import Redis, ConnectionPool

from sqlalchemy import select

from fastapi import Depends

import asyncio

from contextlib import asynccontextmanager

########################################################################################

from quantumion.server.database import get_db, JobInDB

########################################################################################

REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PASSWORD = os.environ["REDIS_PASSWORD"]

redis_pool = ConnectionPool(
    host=REDIS_HOST, password=REDIS_PASSWORD, port=6379, decode_responses=False
)


def get_redis_client():
    redis_client = Redis.from_pool(redis_pool)
    try:
        yield redis_client
    finally:
        redis_client.close()


redis_client_dependency = Annotated[Redis, Depends(get_redis_client)]

########################################################################################


async def _report_success(job, connection, result, *args, **kwargs):
    async with asynccontextmanager(get_db)() as db:
        status_update = dict(status="finished", result=result.model_dump_json())
        query = await db.execute(select(JobInDB).filter(JobInDB.job_id == job.id))
        job_in_db = query.scalars().first()
        for k, v in status_update.items():
            setattr(job_in_db, k, v)
        await db.commit()


def report_success(job, connection, result, *args, **kwargs):
    return asyncio.get_event_loop().run_until_complete(
        _report_success(job, connection, result, *args, **kwargs)
    )


async def _report_failure(job, connection, result, *args, **kwargs):
    async with asynccontextmanager(get_db)() as db:
        status_update = dict(status="failed")
        query = await db.execute(select(JobInDB).filter(JobInDB.job_id == job.id))
        job_in_db = query.scalars().first()
        for k, v in status_update.items():
            setattr(job_in_db, k, v)
        await db.commit()


def report_failure(job, connection, result, *args, **kwargs):
    return asyncio.get_event_loop().run_until_complete(
        _report_failure(job, connection, result, *args, **kwargs)
    )


async def _report_stopped(job, connection, result, *args, **kwargs):
    async with asynccontextmanager(get_db)() as db:
        status_update = dict(status="stopped")
        query = await db.execute(select(JobInDB).filter(JobInDB.job_id == job.id))
        job_in_db = query.scalars().first()
        for k, v in status_update.items():
            setattr(job_in_db, k, v)
        await db.commit()


def report_stopped(job, connection, result, *args, **kwargs):
    return asyncio.get_event_loop().run_until_complete(
        _report_stopped(job, connection, result, *args, **kwargs)
    )
