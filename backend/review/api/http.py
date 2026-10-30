"""Bounded request parsing and compatible public error responses."""
import asyncio
from urllib.parse import urlsplit
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect
from ..provider import ReviewError
from ..courseware_service import MAX_UPLOAD


RESPONSE_HEADERS={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'}

def error_response(exc):
    return JSONResponse({'error':{'code':exc.code,'message':str(exc)}},
                        status_code=exc.status,headers=RESPONSE_HEADERS)

class ReviewRequest(Request):
    """Bound raw bytes before FastAPI parses JSON or accepts a binary upload."""
    async def body(self):
        if not hasattr(self,'_body'):
            try:length=int(self.headers.get('Content-Length','0'))
            except ValueError:raise ReviewError('invalid_length','Invalid request length.',400)
            maximum=MAX_UPLOAD if self.url.path.endswith('/courseware') else 65536
            if not 0<length<=maximum:
                raise ReviewError('body_too_large',f'Request length must be 1–{maximum} bytes.',413)

            async def read():
                chunks=[];total=0
                async for chunk in self.stream():
                    total+=len(chunk)
                    if total>maximum:raise ReviewError('body_too_large','The request exceeds the size limit.',413)
                    if total>length:raise ReviewError('incomplete_upload','The uploaded file length does not match the request length.',400)
                    chunks.append(chunk)
                if total!=length:raise ReviewError('incomplete_upload','The file upload is incomplete.',400)
                return b''.join(chunks)

            try:self._body=await asyncio.wait_for(read(),timeout=60)
            except ClientDisconnect as exc:
                raise ReviewError('incomplete_upload','The file upload is incomplete.',400) from exc
            except TimeoutError as exc:
                raise ReviewError('upload_timeout','Reading the request timed out. Please try again.',408) from exc
        return self._body

def validation_error(exc):
    errors=exc.errors()
    if any(e['type']=='json_invalid' or e['loc']==('body',) for e in errors):
        return ReviewError('invalid_json','The request must be a valid JSON object.',400)
    fields={e['loc'][1] for e in errors if len(e['loc'])>1}
    for field,code,message in [('lesson_id','invalid_lesson','Invalid lesson ID.'),
                               ('query','invalid_query','The question must contain 1–1200 characters.'),
                               ('at_sec','invalid_time','The playback time is outside this lesson.'),
                               ('scope','invalid_scope','Select the whole lesson or the current segment.')]:
        if field in fields:return ReviewError(code,message,400)
    return ReviewError('invalid_json','Invalid request JSON.',400)

class ReviewRoute(APIRoute):
    def get_route_handler(self):
        route_handler=super().get_route_handler()

        async def handle(request):
            request=ReviewRequest(request.scope,request.receive)
            try:
                if request.method in ('POST','PATCH','DELETE'):
                    origin=request.headers.get('Origin')
                    if origin and urlsplit(origin).hostname not in ('127.0.0.1','localhost','::1'):
                        raise ReviewError('origin_rejected','Only requests from the local workspace are allowed.',403)
                if request.method in ('POST','PATCH'):await request.body()
                response=await route_handler(request)
            except RequestValidationError as exc:response=error_response(validation_error(exc))
            except ReviewError as exc:response=error_response(exc)
            except HTTPException as exc:
                response=error_response(ReviewError('invalid_json','Invalid request JSON.',exc.status_code))
            except Exception:
                message={'GET':'Could not read lesson data. Check the server artifacts.',
                         'HEAD':'Could not read lesson data. Check the server artifacts.',
                         'DELETE':'Could not detach the courseware. Refresh to confirm its status.'}.get(request.method,
                            'The review request failed. No substitute answer was generated.')
                response=error_response(ReviewError('internal_error',message,500))
            response.headers.update(RESPONSE_HEADERS)
            return response
        return handle
