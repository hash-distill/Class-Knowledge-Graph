"""Grounded model calls; generation concurrency stays bounded."""
from typing import Annotated,Literal
import uuid
from fastapi import APIRouter
from pydantic import BaseModel,Field,StrictStr
from ..provider import ReviewError
from ..retrieval import answer_question
from .http import ReviewRoute


class ChatRequest(BaseModel):
    lesson_id:StrictStr
    query:Annotated[StrictStr,Field(min_length=1,max_length=1200)]
    at_sec:Annotated[float,Field(strict=True,allow_inf_nan=False,ge=0)]=0
    scope:Literal['lesson','current']='lesson'


def router(services):
    routes=APIRouter(route_class=ReviewRoute)

    @routes.post('/api/review/chat')
    def chat(payload:ChatRequest):
        lesson=services.effective(payload.lesson_id)
        if not services.chat_lock.acquire(blocking=False):
            raise ReviewError('model_busy','另一个问题正在生成，请稍后重试。',429)
        try:
            result=answer_question(lesson,payload.query,payload.at_sec,payload.scope,services.provider)
            result['request_id']=str(uuid.uuid4())
            return result
        finally:services.chat_lock.release()
    return routes
