"""Keep existing binary upload and transactional courseware contracts."""
from typing import Annotated
from urllib.parse import unquote
from fastapi import APIRouter,Header,Request
from starlette.concurrency import run_in_threadpool
from .http import ReviewRoute


def router(services):
    routes=APIRouter(route_class=ReviewRoute)

    @routes.post('/api/review/lessons/{lesson_id}/courseware',status_code=202,
        openapi_extra={'requestBody':{'required':True,'content':{
            'application/octet-stream':{'schema':{'type':'string','format':'binary'}}}}})
    async def upload(lesson_id:str,request:Request,filename:Annotated[str,Header(alias='X-Filename')]=''):
        return await run_in_threadpool(services.courseware.submit,lesson_id,unquote(filename),await request.body())

    @routes.post('/api/review/lessons/{lesson_id}/courseware/matches')
    def confirm(lesson_id:str,payload:dict):return services.workspace.overlay(services.courseware.confirm(lesson_id,payload))

    @routes.delete('/api/review/lessons/{lesson_id}/courseware')
    def detach(lesson_id:str):return services.workspace.overlay(services.courseware.remove(lesson_id))
    return routes
