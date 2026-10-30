"""Teacher revisions, follow-up actions and actual response records."""
from typing import Literal
from fastapi import APIRouter,Query
from pydantic import BaseModel,Field,StrictInt,StrictStr
from .http import ReviewRoute


class Versioned(BaseModel):
    expected_version:StrictInt=Field(ge=0)


class Goals(Versioned):
    text:StrictStr


class Correction(Versioned):
    transcript_id:StrictStr
    replacement:StrictStr
    reason:StrictStr=''


class Action(Versioned):
    title:StrictStr
    evidence_ids:list[StrictStr]=Field(default_factory=list)


class Outcome(Versioned):
    status:Literal['planned','done']
    outcome:StrictStr=''


class Assessment(Versioned):
    question:StrictStr
    reference_answer:StrictStr
    evidence_ids:list[StrictStr]


class Attempt(Versioned):
    answer:StrictStr
    feedback:StrictStr=''
    score:float|None=Field(default=None,ge=0,le=1,allow_inf_nan=False,strict=True)


def router(services):
    routes=APIRouter(prefix='/api/review/lessons/{lesson_id}/workspace',route_class=ReviewRoute)
    def original(lesson_id):return services.store.load(lesson_id)

    @routes.get('')
    def get(lesson_id:str):return services.workspace.get(original(lesson_id))

    @routes.get('/history')
    def history(lesson_id:str):
        services.store.entry(lesson_id)
        return {'history':services.workspace.history(lesson_id)}

    @routes.post('/goals')
    def goals(lesson_id:str,payload:Goals):
        return services.workspace.set_goals(original(lesson_id),payload.text,payload.expected_version)

    @routes.post('/corrections')
    def correct(lesson_id:str,payload:Correction):
        return services.workspace.correct(original(lesson_id),payload.transcript_id,payload.replacement,payload.reason,payload.expected_version)

    @routes.delete('/corrections/{transcript_id}')
    def revert(lesson_id:str,transcript_id:str,expected_version:int=Query(ge=0)):
        return services.workspace.revert(original(lesson_id),transcript_id,expected_version)

    @routes.post('/actions')
    def add_action(lesson_id:str,payload:Action):
        return services.workspace.add_action(original(lesson_id),payload.title,payload.evidence_ids,payload.expected_version)

    @routes.patch('/actions/{action_id}')
    def outcome(lesson_id:str,action_id:str,payload:Outcome):
        return services.workspace.update_action(original(lesson_id),action_id,payload.status,payload.outcome,payload.expected_version)

    @routes.post('/assessments')
    def assessment(lesson_id:str,payload:Assessment):
        return services.workspace.add_assessment(original(lesson_id),payload.question,payload.reference_answer,payload.evidence_ids,payload.expected_version)

    @routes.post('/assessments/{assessment_id}/attempts')
    def attempt(lesson_id:str,assessment_id:str,payload:Attempt):
        return services.workspace.add_attempt(original(lesson_id),assessment_id,payload.answer,payload.feedback,payload.score,payload.expected_version)

    @routes.post('/reconcile')
    def reconcile(lesson_id:str,payload:Versioned):return services.workspace.reconcile(original(lesson_id),payload.expected_version)
    return routes
