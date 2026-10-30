"""Persistent worker status, cancellation and explicit retry."""
import re
from fastapi import APIRouter
from ..provider import ReviewError
from .http import ReviewRoute


def router(services):
    routes=APIRouter(route_class=ReviewRoute)

    @routes.get('/api/review/jobs/{job_id}')
    def status(job_id:str):
        if not re.fullmatch(r'[a-f0-9]{32}',job_id):raise ReviewError('not_found','接口不存在。',404)
        return services.courseware.status(job_id)

    @routes.get('/api/review/tasks')
    def tasks(lesson_id:str|None=None):
        if lesson_id is not None:services.store.entry(lesson_id)
        return {'tasks':services.jobs.list(lesson_id)}

    @routes.post('/api/review/tasks/{job_id}/cancel')
    def cancel(job_id:str):return services.jobs.cancel(job_id)

    @routes.post('/api/review/tasks/{job_id}/retry',status_code=202)
    def retry(job_id:str):return services.jobs.retry(job_id)
    return routes
