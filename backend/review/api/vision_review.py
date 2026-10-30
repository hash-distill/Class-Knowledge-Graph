"""Human visual review endpoints; machine scores are never overwritten."""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StrictInt, StrictStr, ConfigDict
from .http import ReviewRoute
from ..vision_review import frame_context


class ReviewSave(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expected_version: StrictInt = Field(ge=0)
    basis_fingerprint: StrictStr = Field(pattern='^[0-9a-f]{64}$')
    annotations: list[dict] = Field(max_length=200)


class Restore(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expected_version: StrictInt = Field(ge=0)
    basis_fingerprint: StrictStr = Field(pattern='^[0-9a-f]{64}$')
    restore_version: StrictInt = Field(ge=1)


def router(services):
    routes = APIRouter(prefix='/api/review/lessons/{lesson_id}/vision-review', route_class=ReviewRoute)
    def context(lesson_id, frame_id):
        return frame_context(services.store, services.store.load(lesson_id), frame_id)

    @routes.get('/frames/{frame_id}')
    def get(lesson_id: str, frame_id: str):
        return services.vision_review.get(lesson_id, context(lesson_id, frame_id))

    @routes.post('/frames/{frame_id}')
    def save(lesson_id: str, frame_id: str, payload: ReviewSave):
        return services.vision_review.save(lesson_id, context(lesson_id, frame_id), payload.expected_version,
                                          payload.basis_fingerprint, payload.annotations)

    @routes.get('/frames/{frame_id}/history')
    def history(lesson_id: str, frame_id: str):
        services.store.entry(lesson_id)
        return {'history': services.vision_review.history(lesson_id, frame_id)}

    @routes.post('/frames/{frame_id}/restore')
    def restore(lesson_id: str, frame_id: str, payload: Restore):
        return services.vision_review.save(lesson_id, context(lesson_id, frame_id), payload.expected_version,
                                          payload.basis_fingerprint, [], payload.restore_version)

    @routes.get('/export')
    def export(lesson_id: str):
        archive = services.vision_review.export(services.store, services.store.load(lesson_id))
        def chunks():
            try:
                while data := archive.read(128*1024): yield data
            finally: archive.close()
        return StreamingResponse(chunks(), media_type='application/zip', headers={
            'Content-Disposition': f'attachment; filename="vision-review-{lesson_id}.zip"'})
    return routes
