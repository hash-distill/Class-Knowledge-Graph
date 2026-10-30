"""Dedicated correction role; QA/graph provider settings never select this model."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import re

from .common import ROOT, read_json
from .provider import LLMProvider, ReviewError


NATIVE_SYSTEM = 'You correct Chinese text. Fix errors in the input while preserving its original meaning and language. Output only the corrected text.'


class CorrectionProvider(LLMProvider):
    def __init__(self, *, config_path=None, base=None, model=None, key=None, timeout=None,
                 managed_lifecycle=None):
        path = Path(config_path or os.getenv('REVIEW_CORRECTION_CONFIG', ROOT/'configs/correction_llm.json'))
        self.config = read_json(path) if path.is_file() else {}
        self.managed_lifecycle = bool(self.config.get('managed_lifecycle',False) if managed_lifecycle is None else managed_lifecycle)
        self.runtime_profile = str(path.resolve())
        self._prepared = False
        self._restore_needed = False
        self._session_lease = None
        defaults = self.config.get('request_defaults', {})
        secret_env = self.config.get('api_key_env', 'REVIEW_CORRECTION_API_KEY')
        super().__init__(
            base=base or os.getenv('REVIEW_CORRECTION_BASE_URL') or self.config.get('base_url','http://127.0.0.1:18082/v1'),
            model=model or os.getenv('REVIEW_CORRECTION_MODEL') or self.config.get('model_alias','classroom-corrector'),
            key=key if key is not None else os.getenv(secret_env,''),
            timeout=timeout if timeout is not None else os.getenv('REVIEW_CORRECTION_TIMEOUT',self.config.get('timeout_seconds',180)),
            temperature=defaults.get('temperature',0.1),
            chat_template_kwargs=deepcopy(defaults.get('chat_template_kwargs',{'enable_thinking':False})),
            capture_response=True)

    def status(self):
        model = self.config.get('model', {})
        # Include request/model identity in the cache without private credentials.
        parameters = {'temperature':self.temperature,'chat_template_kwargs':self.chat_template_kwargs}
        return {**super().status(), 'role':'correction',
                'endpoint_digest':hashlib.sha256(self.base.encode()).hexdigest(),
                'model_sha256':model.get('sha256'), 'model_revision':model.get('revision'),
                'request_parameters':parameters,
                'native_system_digest':hashlib.sha256(self.config.get('native_system',NATIVE_SYSTEM).encode()).hexdigest()}

    def prepare(self):
        if self.managed_lifecycle and not self._prepared:
            from .model_runtime import ensure_role, model_session
            # A multi-chunk caller may retry after a failed switch. Reuse the
            # entered outer lease until close() restores QA and releases it.
            if self._session_lease is None:
                lease = model_session()
                lease.__enter__()
                self._session_lease = lease
            # A failed switch may already have unloaded QA; always recover it.
            self._restore_needed = True
            ensure_role('correction',profile_path=self.runtime_profile)
            self._prepared = True

    def close(self):
        try:
            if self._restore_needed:
                from .model_runtime import ensure_role
                ensure_role('qa',profile_path=self.runtime_profile)
        finally:
            self._prepared = False
            self._restore_needed = False
            if self._session_lease is not None:
                lease, self._session_lease = self._session_lease, None
                lease.__exit__(None, None, None)

    finish = close

    def complete(self,*args,**kwargs):
        self.prepare()
        answer,metadata=super().complete(*args,**kwargs)
        metadata.update(role='correction',model_sha256=self.config.get('model',{}).get('sha256'))
        return answer,metadata

    def native_correct(self, text, *, max_tokens=5000):
        """Run the model's plain-text track without the controlled JSON edit prompt.

        Returned text is an unvalidated model hypothesis. The caller must retain
        source text and apply the normal edit gates before publishing any change.
        """
        if not isinstance(text,str) or not text.strip():
            raise ValueError('Native correction input must be nonempty text')
        system = self.config.get('native_system', NATIVE_SYSTEM)
        prompt_template = self.config.get('native_prompt_template','{text}')
        prompt = prompt_template.replace('{text}',text)
        response, metadata = self.complete(system,prompt,max_tokens=max_tokens)
        thoughts = re.findall(r'<think>(.*?)</think>',response,flags=re.DOTALL)
        if isinstance(metadata.get('reasoning_content'),str):
            thoughts.insert(0,metadata['reasoning_content'])
        corrected = re.sub(r'<think>.*?</think>','',response,flags=re.DOTALL).strip()
        if not corrected or '<think>' in corrected:
            raise ReviewError('invalid_correction_response','The correction model did not return separable revised text.',502,diagnostic=metadata)
        metadata.update(track='native_plain_text',reasoning_text='\n'.join(thoughts),
            system_digest=hashlib.sha256(system.encode()).hexdigest(),
            prompt_digest=hashlib.sha256(prompt.encode()).hexdigest(),
            human_verified=False)
        return corrected,metadata
