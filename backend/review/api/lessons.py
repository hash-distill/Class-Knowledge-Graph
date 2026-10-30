"""Lesson catalog and source-bound evidence queries."""
from fastapi import APIRouter,Query
from .http import ReviewRoute


def router(services):
    routes=APIRouter(route_class=ReviewRoute)

    @routes.get('/api/review/health')
    def health():
        return {'service':'classroom-review','status':'ready','framework':'fastapi','storage':'sqlite',
                'worker':'subprocess','model':services.provider.status(),'lessons':len(services.store.list_lessons())}

    @routes.get('/api/review/lessons')
    def lessons():return {'lessons':services.store.list_lessons()}

    @routes.get('/api/review/runtime')
    def runtime():
        from ..runtime import inspect_runtime
        return inspect_runtime()

    @routes.get('/api/review/lessons/{lesson_id}')
    def lesson(lesson_id:str):return services.effective(lesson_id)

    @routes.get('/api/review/lessons/{lesson_id}/evidence/search')
    def search(lesson_id:str,q:str=Query(min_length=1,max_length=1200)):
        return {'evidence':services.index.search(services.effective(lesson_id),q)}
    return routes
