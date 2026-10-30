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
            except ValueError:raise ReviewError('invalid_length','请求长度无效。',400)
            maximum=MAX_UPLOAD if self.url.path.endswith('/courseware') else 65536
            if not 0<length<=maximum:
                raise ReviewError('body_too_large',f'请求长度需为 1–{maximum} 字节。',413)

            async def read():
                chunks=[];total=0
                async for chunk in self.stream():
                    total+=len(chunk)
                    if total>maximum:raise ReviewError('body_too_large','请求超过允许大小。',413)
                    if total>length:raise ReviewError('incomplete_upload','请求文件长度不一致。',400)
                    chunks.append(chunk)
                if total!=length:raise ReviewError('incomplete_upload','请求文件未完整上传。',400)
                return b''.join(chunks)

            try:self._body=await asyncio.wait_for(read(),timeout=60)
            except ClientDisconnect as exc:
                raise ReviewError('incomplete_upload','请求文件未完整上传。',400) from exc
            except TimeoutError as exc:
                raise ReviewError('upload_timeout','读取请求超时，请重试。',408) from exc
        return self._body

def validation_error(exc):
    errors=exc.errors()
    if any(e['type']=='json_invalid' or e['loc']==('body',) for e in errors):
        return ReviewError('invalid_json','请求必须是有效的JSON对象。',400)
    fields={e['loc'][1] for e in errors if len(e['loc'])>1}
    for field,code,message in [('lesson_id','invalid_lesson','课堂编号无效。'),
                               ('query','invalid_query','问题需为 1–1200 字。'),
                               ('at_sec','invalid_time','播放时间超出本节课范围。'),
                               ('scope','invalid_scope','请选择整课或当前片段。')]:
        if field in fields:return ReviewError(code,message,400)
    return ReviewError('invalid_json','请求JSON格式无效。',400)

class ReviewRoute(APIRoute):
    def get_route_handler(self):
        route_handler=super().get_route_handler()

        async def handle(request):
            request=ReviewRequest(request.scope,request.receive)
            try:
                if request.method in ('POST','PATCH','DELETE'):
                    origin=request.headers.get('Origin')
                    if origin and urlsplit(origin).hostname not in ('127.0.0.1','localhost','::1'):
                        raise ReviewError('origin_rejected','仅允许本地工作台请求。',403)
                if request.method in ('POST','PATCH'):await request.body()
                response=await route_handler(request)
            except RequestValidationError as exc:response=error_response(validation_error(exc))
            except ReviewError as exc:response=error_response(exc)
            except HTTPException as exc:
                response=error_response(ReviewError('invalid_json','请求JSON格式无效。',exc.status_code))
            except Exception:
                message={'GET':'课堂数据读取失败，请检查服务端产物。',
                         'HEAD':'课堂数据读取失败，请检查服务端产物。',
                         'DELETE':'课件解绑失败，请刷新确认状态。'}.get(request.method,
                            '复盘请求处理失败，未生成替代回答。')
                response=error_response(ReviewError('internal_error',message,500))
            response.headers.update(RESPONSE_HEADERS)
            return response
        return handle
