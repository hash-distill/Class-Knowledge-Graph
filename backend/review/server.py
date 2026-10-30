"""Compatibility entry point for the modular local FastAPI application."""
from .api.application import create_app
from .api.chat import ChatRequest
from .api.media import byte_range
from .lesson_store import LessonStore


def serve(host='127.0.0.1',port=8000):
    import uvicorn
    print(f'Jingxi Insight API: http://{host}:{port}/api/review/health',flush=True)
    uvicorn.run(create_app(),host=host,port=port,workers=1,access_log=False,log_level='warning')
