# Language data in the English application

Jingxi Insight uses English application text, generated report labels, errors,
launcher help, and model instructions. Source evidence keeps its original
language. Translating a quotation, OCR result, transcript, user correction, or
test reference would change the evidence being reviewed.

Chinese literals are intentionally retained in these places:

- `review/alignment.py`: Chinese noun-phrase suffixes and dictionary examples.
- `review/courseware_terms.py`: protected negation/numeral characters,
  mathematical suffix patterns, clause punctuation, and linguistic examples.
- `review/proofreading.py`: negation characters, Chinese number patterns, and
  Chinese pronunciations of Greek letter names. These support phonetic and
  semantic preservation checks.
- `review/retrieval.py`: Chinese query-intent keywords and citation punctuation
  delimiters. English intent aliases use the same whole-lesson and balanced-source
  retrieval policies.
- `review/activity.py`: historical generated section titles and missing-behavior
  notices used only as compatibility identifiers when replacing old report rows.
- `review/qwen_asr.py`: Chinese sentence punctuation for speech segmentation.
- `scripts/compare_correction_models.py`: Chinese numeral and negation patterns
  used to flag possible protected-content changes.
- `scripts/build_unified_chemistry_demo.py`: source-vocabulary punctuation in
  ASR hotword inputs. Vocabulary remains verbatim from observed courseware.
- `scripts/verify_courseware_evidence.py`, `tests/test_*.py`, and
  `tests/fixtures/cec4/`: source transcripts, OCR, homophone errors, expected
  corrections, Traditional/Simplified Chinese examples, source quotations,
  arbitrary Unicode filenames, and user-entered evidence. They test the
  Chinese-language processing and evidence-preservation contract.

The correction configuration and benchmark instruction are English
translations of the previously published Chinese task instruction. Their
policy labels identify that translation. New runs therefore use different
prompt conditions from historical runs with the original Chinese instruction;
the translation does not establish equivalent model behavior or accuracy.

Repository-layout fixtures use `backend`, `frontend`, and `vision`. Private
desktop handshake headers and readiness markers retain their existing protocol
values for compatibility.
