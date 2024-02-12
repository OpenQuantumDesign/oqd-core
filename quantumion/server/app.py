from fastapi import FastAPI

from contextlib import asynccontextmanager

########################################################################################

from quantumion.server.database import engine, Base
from quantumion.server.route import user_router, auth_router, job_router

########################################################################################


@asynccontextmanager
async def create_db(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(lifespan=create_db)
app.include_router(user_router)
app.include_router(auth_router)
app.include_router(job_router)


async def create_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
