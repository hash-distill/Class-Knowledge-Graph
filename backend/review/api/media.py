"""Streaming local media without loading entire videos into memory."""
import mimetypes
import re
from fastapi import APIRouter,Request
from fastapi.responses import Response,StreamingResponse
from ..provider import ReviewError
from .http import ReviewRoute


def byte_range(value,size):
    if not value:return 0,size-1,False
    match=re.fullmatch(r'bytes=(\d*)-(\d*)',value)
    if not match or not any(match.groups()):raise ReviewError('invalid_range','Invalid byte range',416)
    left,right=match.groups()
    if not left:
        suffix=int(right)
        if suffix<=0:raise ReviewError('invalid_range','Invalid suffix length',416)
        start,end=max(0,size-suffix),size-1
    else:start,end=int(left),min(int(right),size-1) if right else size-1
    if start>=size or start>end:raise ReviewError('invalid_range','Requested range is outside media',416)
    return start,end,True


def router(services):
    routes=APIRouter(route_class=ReviewRoute)

    @routes.get('/api/review/media/{lesson_id}/{relative:path}')
    @routes.head('/api/review/media/{lesson_id}/{relative:path}',include_in_schema=False)
    def media(lesson_id:str,relative:str,request:Request):
        file=services.store.media(lesson_id,relative);size=file.stat().st_size
        try:start,end,partial=byte_range(request.headers.get('Range'),size)
        except ReviewError:
            return Response(status_code=416,headers={'Content-Range':f'bytes */{size}'})
        headers={'Accept-Ranges':'bytes','Content-Length':str(end-start+1)}
        if partial:headers['Content-Range']=f'bytes {start}-{end}/{size}'
        status=206 if partial else 200
        kind=mimetypes.guess_type(file.name)[0] or 'application/octet-stream'
        if request.method=='HEAD':return Response(status_code=status,headers=headers,media_type=kind)

        def chunks():
            with file.open('rb') as handle:
                handle.seek(start);remaining=end-start+1
                while remaining:
                    chunk=handle.read(min(128*1024,remaining))
                    if not chunk:break
                    yield chunk
                    remaining-=len(chunk)
        return StreamingResponse(chunks(),status_code=status,headers=headers,media_type=kind)
    return routes
