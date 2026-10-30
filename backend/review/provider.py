"""Server-only model configuration; no silent canned-answer fallback."""
from __future__ import annotations
import json
import os
import time
from urllib.parse import urlparse
import requests


class ReviewError(Exception):
    def __init__(self, code, message, status=503, *, diagnostic=None):
        super().__init__(message)
        self.code, self.status = code, status
        # Private generation evidence; the HTTP layer only exposes code/message.
        self.diagnostic = diagnostic or {}


class LLMProvider:
    def __init__(self, *, base=None, model=None, key=None, timeout=None,
                 temperature=0.1, chat_template_kwargs=None, capture_response=False):
        self.base = (base if base is not None else os.getenv('REVIEW_LLM_BASE_URL','http://127.0.0.1:18081/v1')).rstrip('/')
        self.model = model if model is not None else os.getenv('REVIEW_LLM_MODEL','classroom-local')
        self.key = key if key is not None else os.getenv('REVIEW_LLM_API_KEY','')
        self.timeout = float(timeout if timeout is not None else os.getenv('REVIEW_LLM_TIMEOUT','120'))
        self.temperature = float(temperature)
        self.chat_template_kwargs = chat_template_kwargs if chat_template_kwargs is not None else {'enable_thinking':False}
        self.capture_response = capture_response
        parsed = urlparse(self.base)
        if parsed.scheme not in ('http','https') or not parsed.hostname:
            raise ValueError('Invalid REVIEW_LLM_BASE_URL')
        self.source = 'local_llm' if parsed.hostname in ('127.0.0.1','localhost','::1') else 'remote_llm'

    def status(self):
        return {'model':self.model,'source':self.source,'configured':True,
                'endpoint_host':urlparse(self.base).hostname}

    def complete(self, system, prompt, *, json_mode=False, max_tokens=1100, schema=None):
        if self.source == 'local_llm' and self.model in ('classroom-local', 'classroom-corrector', 'classroom-corrector-qwen35'):
            from .model_runtime import model_session
            with model_session():
                return self._complete(system, prompt, json_mode=json_mode, max_tokens=max_tokens, schema=schema)
        return self._complete(system, prompt, json_mode=json_mode, max_tokens=max_tokens, schema=schema)

    def _complete(self, system, prompt, *, json_mode=False, max_tokens=1100, schema=None):
        payload={'model':self.model,'messages':[{'role':'system','content':system},
                  {'role':'user','content':prompt}],'temperature':self.temperature,'max_tokens':max_tokens,'stream':False}
        if self.source=='local_llm': payload['chat_template_kwargs']=self.chat_template_kwargs
        if json_mode: payload['response_format']={'type':'json_object'}
        if schema is not None:
            payload['response_format']={'type':'json_schema','json_schema':{'name':'grounded_answer','strict':True,'schema':schema}}
            json_mode=True
        headers={'Content-Type':'application/json'}
        if self.key: headers['Authorization']='Bearer '+self.key
        started = time.perf_counter()
        try:
            response=requests.post(self.base+'/chat/completions',json=payload,headers=headers,timeout=self.timeout)
        except requests.Timeout as exc:
            raise ReviewError('model_timeout','Model generation timed out. Please try again shortly; no answer was generated.',504) from exc
        except requests.RequestException as exc:
            raise ReviewError('model_unavailable','The local or configured model service is not ready. Start it and try again.') from exc
        if not response.ok:
            # Never expose upstream credential-bearing bodies or HTML to the browser.
            raise ReviewError('model_rejected',f'The model endpoint returned HTTP {response.status_code}; no answer was generated.',502)
        try:
            data=response.json()
            content=data['choices'][0]['message']['content']
            if not isinstance(content,str) or not content.strip(): raise ValueError('empty')
        except (ValueError,KeyError,IndexError,TypeError) as exc:
            raise ReviewError('invalid_model_response','The model returned invalid content; no answer was generated.',502) from exc
        metadata={'model':data.get('model',self.model),'source':self.source,
                  'usage':data.get('usage',{}),'completion_id':data.get('id')}
        if self.capture_response:
            metadata.update(raw_response=content,
                reasoning_content=data['choices'][0]['message'].get('reasoning_content'),
                finish_reason=data['choices'][0].get('finish_reason'),
                elapsed_seconds=round(time.perf_counter()-started,6),
                request_parameters={k:v for k,v in payload.items() if k not in ('messages','model')})
        if data['choices'][0].get('finish_reason')!='stop':
            raise ReviewError('model_incomplete','The model response was incomplete or reached the length limit. Narrow the question and try again.',502,diagnostic=metadata)
        if json_mode:
            try: content=json.loads(content)
            except ValueError as exc:
                raise ReviewError('invalid_model_json','The model did not return the required structured result.',502,diagnostic=metadata) from exc
        return content, metadata
