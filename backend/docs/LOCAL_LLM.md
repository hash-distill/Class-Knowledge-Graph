# Local public language model

The historical local deployment ran official Qwen3-4B Q4_K_M through llama.cpp's OpenAI-compatible API for evidence-based backend Q&A. It involved no training, fine-tuning, or paid service. The model receives text evidence, not raw video, audio, or behavior.

## Pinned assets

| Item | Pinned value |
| --- | --- |
| Weight repository | Official `Qwen/Qwen3-4B-GGUF` |
| Weight revision | `bc640142c66e1fdd12af0bd68f40445458f3869b` |
| File | `Qwen3-4B-Q4_K_M.gguf`, 2,497,280,256 bytes |
| Weight SHA-256 | `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` |
| Architecture / size / quantization | `qwen3` / 4B / Q4_K_M; server reports 4,022,468,096 parameters |
| Weight license | Apache-2.0; retained locally at `artifacts/models/Qwen3-4B-GGUF/LICENSE` |
| Runtime | Official `ggml-org/llama.cpp` Windows x64 release `b11871` |
| Runtime revision | `b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9` |
| Runtime license | MIT; original `LICENSE` retained in each runtime directory |

`configs/local_llm.json` pins official SHA-256 and sizes for CPU, CUDA 12.4, and CUDA DLL archives. Downloads are fully verified before extraction. Original archives remain in `artifacts/runtime/downloads`; model cards/licenses are retained. Local asset records are `artifacts/verification/llm/asset_manifest_cpu.json` and `asset_manifest_cuda.json`.

Official sources: [pinned model revision](https://huggingface.co/Qwen/Qwen3-4B-GGUF/tree/bc640142c66e1fdd12af0bd68f40445458f3869b), [model card/license](https://huggingface.co/Qwen/Qwen3-4B-GGUF), [llama.cpp b11871](https://github.com/ggml-org/llama.cpp/releases/tag/b11871), and [pinned server documentation](https://github.com/ggml-org/llama.cpp/blob/b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9/tools/server/README.md). Parameters were checked against Context7 repository documentation and downloaded `llama-server.exe --help` in the recorded setup.

## Start and stop

Run from `backend/`; no API key is required. The default configuration uses CUDA. The CPU runtime also passed a historical health check.

```powershell
python scripts/fetch_local_llm.py --runtime cuda
python scripts/fetch_local_llm.py --runtime cuda --verify-only
.\scripts\start_local_llm.ps1 -Backend cuda
```

The launcher runs hidden and rechecks the weight digest. Logs, PID, executable path, and process start time are saved under `artifacts/verification/llm/`, including `server_state.json`. Stop checks PID ownership and start time to avoid terminating another process after PID reuse:

```powershell
.\scripts\start_local_llm.ps1 -Stop
```

Stop before changing configuration or runtime. CPU fallback:

```powershell
python scripts/fetch_local_llm.py --runtime cpu
.\scripts\start_local_llm.ps1 -Stop
.\scripts\start_local_llm.ps1 -Backend cpu
```

CPU health succeeded historically; the Chinese generation timings below came from CUDA and are not CPU performance figures.

## API and resources

- Base URL: `http://127.0.0.1:18081/v1`.
- Generation: `POST /v1/chat/completions`; models: `GET /v1/models`; health: `GET /health` at the host root.
- Alias: `classroom-local`.
- Context: 24,576 tokens; one parallel slot; four CPU threads each for generation and prompt processing; full layer offload on CUDA.
- Server metadata confirmed `n_ctx=24576`. Observed whole-GPU usage on an RTX 4060 Laptop 8 GB was about 6,174 MiB; other processes and larger contexts affect available memory.
- Bound combined input/output tokens to context capacity; characters are not tokens. Serialize requests or queue them, with a 180-second timeout.

Illustrative English request:

```json
{
  "model": "classroom-local",
  "messages": [
    {"role": "system", "content": "Answer only from the supplied lesson evidence. State when evidence is insufficient. Copy citation brackets and their complete IDs exactly; do not shorten or rewrite them."},
    {"role": "user", "content": "Insert retrieved evidence and the user's question here."}
  ],
  "temperature": 0.1,
  "max_tokens": 1000,
  "stream": false,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

This deployment disables Qwen3 thinking globally with `--jinja --reasoning off` and sets `chat_template_kwargs.enable_thinking=false` per request. The recorded llama.cpp version supported that request field while deprecating `enable_thinking` in the startup `--chat-template-kwargs`; the launcher therefore uses `--reasoning off`. Removing `<think>` tags from output is not equivalent to disabling thinking.

`temperature=0.1` is the project's evidence-Q&A choice, not a claimed optimum. The model card recommends 0.7 for non-thinking mode. Concept extraction can override `max_tokens=1700` if sufficient context remains.

## Historical checks — 2026-10-25

1. Three offline tests checked pinned sources, same-size tampering/hash failure, and ZIP traversal rejection. They did not evaluate answer accuracy.
2. The CPU runtime loaded and returned health; it was subsequently stopped.
3. CUDA loading, `/health`, and `/v1/models` succeeded with the expected alias, context, and Q4_K metadata.
4. Two Chinese questions were generated live in about 0.72 s and 0.24 s. Test citations were preserved, missing homework evidence was acknowledged, and no thinking tags or `reasoning_content` appeared.
5. A **synthetic load** of 19,000 Chinese characters with `max_tokens=1700` succeeded: 14,186 input tokens, four output tokens, about 5.91 s. This checks that input/configuration, not 1,700-token generation quality or every 24K boundary case.

The historical local replay script is excluded from public source packages; the source test remains available:

```powershell
# Available only with the original local verification artifacts:
python artifacts/verification/llm/smoke_local_llm.py
python -m unittest discover -s tests -p test_llm_assets.py -v
```

Local `live_smoke_cuda.json` and `long_context_cuda.json` retain requests, responses, and timing. The two-question evidence was explicitly synthetic interface data, not an actual lesson transcript. It does not establish real classroom Q&A/citation acceptance. These timings and checks also do not validate subsequently translated English prompts.

The initial model shortened `[probe_E1]` to `E1`, recorded in `live_smoke_cuda_initial.json`. An exact-ID instruction fixed that test, but models can still omit or rewrite citations. The backend must accept only retrieved current-lesson IDs and validate citations before returning answers. Real-corpus accuracy, citation fidelity, and relationship quality require separate acceptance.

Return real service errors. Do not substitute preset text, synthetic answers, or balance-failure templates for model output.
