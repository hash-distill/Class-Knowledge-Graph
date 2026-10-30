"""Compose local services and manage the subprocess controller lifecycle."""
from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import threading
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from ..common import ROOT
from ..courseware_service import CoursewareService
from ..evidence_index import EvidenceIndex
from ..jobs import JobQueue
from ..lesson_store import LessonStore
from ..provider import LLMProvider,ReviewError
from ..workspace import WorkspaceStore
from ..vision_review import VisionReviewStore
from .http import error_response
from . import chat,courseware,lessons,media,tasks,workspace,vision_review


@dataclass
class Services:
    store:LessonStore
    provider:LLMProvider
    workspace:WorkspaceStore
    index:EvidenceIndex
    jobs:JobQueue
    courseware:CoursewareService
    chat_lock:threading.BoundedSemaphore
    vision_review:VisionReviewStore

    def effective(self,lesson_id):
        return self.workspace.overlay(self.store.load(lesson_id))


def create_app(*,store=None,provider=None,courseware_service=None,data_dir=None,frontend_dir=None):
    default_store=store is None
    store=store if store is not None else LessonStore(os.getenv('REVIEW_LESSON_REGISTRY') or None)
    data=Path(data_dir or os.getenv('REVIEW_DATA_DIR') or
              (ROOT/'artifacts/runtime' if default_store else store.registry_path.parent/'.review-state'))
    jobs=JobQueue(data/'jobs.sqlite3',store.registry_path)
    services=Services(store,provider if provider is not None else LLMProvider(),WorkspaceStore(data/'workspace.sqlite3'),
        EvidenceIndex(data/'evidence.sqlite3'),jobs,
        courseware_service if courseware_service is not None else CoursewareService(store,job_queue=jobs),
        threading.BoundedSemaphore(1),VisionReviewStore(data/'workspace.sqlite3'))

    @asynccontextmanager
    async def lifespan(app):
        jobs.start()
        try:yield
        finally:jobs.stop()

    app=FastAPI(title='Jingxi Insight',version='4.0',redirect_slashes=False,lifespan=lifespan)
    app.state.services=services

    @app.exception_handler(HTTPException)
    async def http_error(request,exc):
        code='not_found' if exc.status_code==404 else 'method_not_allowed'
        return error_response(ReviewError(code,'Endpoint not found.' if exc.status_code==404 else 'This endpoint does not support this request method.',exc.status_code))

    for module in (lessons,media,chat,courseware,workspace,tasks,vision_review):app.include_router(module.router(services))
    frontend=Path(frontend_dir or os.getenv('REVIEW_FRONTEND_DIR') or ROOT.parent/'frontend/dist')
    if (frontend/'index.html').is_file():app.mount('/',StaticFiles(directory=frontend,html=True),name='workbench')
    return app
