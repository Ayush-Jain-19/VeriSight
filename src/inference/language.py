"""English writing-style estimates and lexical caption/transcript comparison."""
import json
import math
from pathlib import Path
import re
import unicodedata
from functools import lru_cache

from .runtime import positive_index

TEXT = "fakespot-ai/roberta-base-ai-text-detection-v1"
TEXT_MODELS = {TEXT, "GeorgeDrayson/modernbert-ai-detection", "rasbt/ai-text-detector-modernbert"}
STYLE = "Experimental AI-writing style estimate"


@lru_cache(maxsize=1)
def language_factory():
    from langdetect import DetectorFactory, PROFILES_DIRECTORY
    factory = DetectorFactory()
    factory.load_profile(PROFILES_DIRECTORY)
    factory.set_seed(0)
    return factory


def english(text):
    detector = language_factory().create()
    detector.append(text)
    return detector.detect() == "en"


def load_text(name):
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    config = AutoConfig.from_pretrained(name)
    index = positive_index(config.id2label, {"ai", "chatgpt", "generated"}, {"human"})
    temperature = 1.0
    options = {}
    if config.model_type == "modernbert":
        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast
        # The publisher ships a v5 fast-tokenizer JSON; the installed v4 runtime
        # can read the same tokenizer graph through its generic fast wrapper.
        tokenizer = PreTrainedTokenizerFast.from_pretrained(name)
        meta = json.loads(Path(hf_hub_download(name, "detector-config.json")).read_text())
        temperature = float(meta["temperature"])
        if not math.isfinite(temperature) or temperature <= 0 or meta["label_mapping"]["ai"] != index:
            raise ValueError("Invalid publisher calibration metadata")
        config.reference_compile = False
        options["attn_implementation"] = "eager"
        options["torch_dtype"] = "auto"
    else:
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        name, config=config, use_safetensors=True, output_loading_info=True, **options)
    if loading["missing_keys"] or loading["mismatched_keys"]:
        raise ValueError("Text checkpoint did not supply all model weights")
    model.eval()
    if index != positive_index(model.config.id2label, {"ai", "chatgpt", "generated"}, {"human"}):
        raise ValueError("Loaded weights have conflicting class labels")
    return tokenizer, model, index, temperature


def score_chunks(text, tokenizer, model, index, temperature=1.):
    import torch
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    size = min(512, tokenizer.model_max_length) - tokenizer.num_special_tokens_to_add(False)
    overlap = min(96, size // 4)
    spans, weighted, covered = [], 0.0, 0
    for start in range(0, len(ids), size - overlap):
        end = min(len(ids), start + size)
        if getattr(getattr(model, "config", None), "model_type", None) == "modernbert":
            sequence = [tokenizer.cls_token_id, *ids[start:end], tokenizer.sep_token_id]
        else:
            sequence = tokenizer.build_inputs_with_special_tokens(ids[start:end])
        with torch.inference_mode():
            batch = torch.tensor([sequence], dtype=torch.long)
            logits = model(input_ids=batch, attention_mask=torch.ones_like(batch)).logits
            score = float((logits.float()/temperature).softmax(-1)[0, index])
        if not math.isfinite(score):
            raise ValueError("Nonfinite text model output")
        weight = end - covered
        left, right = offsets[start][0], offsets[end-1][1]
        spans.append(dict(start=left, end=right, text=text[left:right], score=score,
                          new_tokens=weight))
        weighted += score * weight
        covered = end
        if covered == len(ids):
            break
    if not covered:
        raise ValueError("Tokenizer returned no usable tokens")
    return weighted / covered, spans


def sentence_evidence(text, bundle, limit=8):
    # Independent sentence estimates with original offsets. A deterministic,
    # evenly spaced subset bounds forwards on long documents.
    sentences = list(re.finditer(r"[^.!?\n]+(?:[.!?]+|(?=\n|$))", text))
    if len(sentences) <= 1:
        return []
    indices = range(len(sentences)) if len(sentences) <= limit else [
        round(i*(len(sentences)-1)/(limit-1)) for i in range(limit)]
    evidence = []
    for i in indices:
        match = sentences[i]
        value = match.group().strip()
        if not value:
            continue
        start = match.start() + len(match.group()) - len(match.group().lstrip())
        score, _ = score_chunks(value, *bundle)
        evidence.append(dict(start=start, end=start+len(value), text=value, score=score))
    return evidence


def analyze_text(text, slot, name):
    result = dict(score=None, available=False, model=name, label=STYLE, reason=None,
                  word_count=len(text.split()), spans=[])
    try:
        if len(text.strip()) < 50 or result["word_count"] < 10:
            result["reason"] = "Text is too short (minimum 50 characters and 10 words)."
        elif not english(text):
            result["reason"] = "Language detector did not identify English."
        else:
            with slot.use(name, lambda: load_text(name)) as bundle:
                score, spans = score_chunks(text, *bundle)
                result["sentences"] = sentence_evidence(text, bundle)
            result.update(score=score, spans=spans, available=True)
    except Exception as exc:
        result["reason"] = f"Text estimation unavailable: {exc}"
    return result


def caption_reference(value, limit_seconds=30):
    """Select SRT/VTT cues starting in the decoded interval. Plain captions pass through."""
    value = (value or "").strip().lstrip("\ufeff")
    if "-->" not in value:
        return value
    timestamp = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})\s*-->")
    words = []
    for block in re.split(r"\n\s*\n", value.replace("\r", "")):
        lines = block.splitlines()
        for i, line in enumerate(lines):
            match = timestamp.match(line.strip())
            if match:
                h, m, sec, ms = match.groups()
                if int(h or 0)*3600 + int(m)*60 + int(sec) + int(ms)/1000 < limit_seconds:
                    words.append(re.sub(r"<[^>]*>", "", " ".join(lines[i+1:])))
                break
    return " ".join(words).strip()


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[^\W_]+", text))


def compare_caption(caption, transcript):
    """Use supplied caption as reference. Edit rates may exceed 1 for insertions."""
    from rapidfuzz.distance import Levenshtein
    from sklearn.feature_extraction.text import TfidfVectorizer
    ref, hyp = normalize(caption), normalize(transcript)
    if not ref or not hyp:
        return {}
    vectors = TfidfVectorizer(tokenizer=str.split, token_pattern=None).fit_transform([ref, hyp])
    cosine = float((vectors[0] @ vectors[1].T)[0, 0])
    cosine = max(0., min(1., cosine))
    return dict(wer=Levenshtein.distance(ref.split(), hyp.split()) / len(ref.split()),
                cer=Levenshtein.distance(ref, hyp) / len(ref),
                cross_modal_consistency=cosine, mismatch=1-cosine,
                similarity_method="TF-IDF lexical cosine")
