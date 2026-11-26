"""Offline contract and preprocessing coverage. Test doubles stay in tests."""
import gc
import subprocess
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.api.schemas import PredictResponse
from src.inference import pipeline as serving
from src.inference.decision import Signal, classify, fuse, STRATEGIES
from src.inference.language import compare_caption, english, score_chunks
from src.inference.pipeline import InferencePipeline
from src.inference.runtime import ModelSlot, positive_index
from src.inference.speech import active_windows, decode


@pytest.mark.parametrize("labels,expected", [({0:"FAKE",1:"REAL"},0),({"0":"real","1":"fake"},1)])
def test_dynamic_labels(labels,expected):
    assert positive_index(labels,{"fake"},{"real"}) == expected


@pytest.mark.parametrize("labels",[{0:"LABEL_0",1:"LABEL_1"},{0:"fake",1:"fake"},{}])
def test_unknown_labels_abstain(labels):
    with pytest.raises(ValueError):
        positive_index(labels,{"fake"},{"real"})


@pytest.mark.parametrize("extension",["wav","m4a","aac","mp3","ogg","flac"])
def test_decode_all_containers(tmp_path,synthetic_audio_wav,extension):
    path = tmp_path/f"converted.{extension}"
    subprocess.run(["ffmpeg","-v","error","-i",str(synthetic_audio_wav),"-ar","44100",
                    "-ac","2",str(path)],check=True)
    wav=decode(path)
    assert wav.ndim == 1
    assert wav.dtype == np.float32
    assert len(wav)/16000 == pytest.approx(2,abs=.10)


@pytest.mark.parametrize("size",[4,5])
def test_window_uses_active_time(size):
    wav=np.r_[np.zeros(32000),np.ones(16000*6)*.1,np.zeros(32000)]
    segments=list(active_windows(wav,size))
    assert len(segments)==2
    assert len(segments[0][0])==size*16000
    assert sum(w for _,w in segments)==6
    assert not list(active_windows(np.zeros(16000),size))


def test_language():
    assert english("After walking across the bridge, I put the small notebook in my pocket.")
    assert not english("Cuando llegamos a la casa de nuestra amiga, nos esperaba una gran sorpresa.")


def test_long_text_spans_cover_tail():
    class Tokens:
        model_max_length=512
        def __call__(self,text,**kw):
            return dict(input_ids=list(range(len(text))),offset_mapping=[(i,i+1) for i in range(len(text))])
        def num_special_tokens_to_add(self,pair): return 2
        def build_inputs_with_special_tokens(self,ids): return [0,*ids,1]

    def model(input_ids,attention_mask):
        assert input_ids.shape[1]<=512
        return SimpleNamespace(logits=torch.tensor([[0.,float(input_ids[0,-2])/500]]))
    passage="x"*1200
    score,spans=score_chunks(passage,Tokens(),model,1)
    assert len(spans)==3 and spans[1]["start"]<spans[0]["end"] and spans[-1]["end"]==len(passage)
    assert sum(s["new_tokens"] for s in spans)==len(passage)
    assert score==pytest.approx(sum(s["score"]*s["new_tokens"] for s in spans)/len(passage))
    assert all(s["text"]==passage[s["start"]:s["end"]] for s in spans)


def test_caption_metrics():
    same=compare_caption("It is here!","it is HERE")
    assert same["wer"]==same["cer"]==0
    assert same["mismatch"]==pytest.approx(0,abs=1e-10)
    diff=compare_caption("cat","cat sat here")
    assert diff["wer"]==2 and 0<diff["mismatch"]<=1
    assert not compare_caption("","abc")


@pytest.mark.parametrize("strategy",STRATEGIES)
def test_fusion_missing_and_conflict(strategy):
    def assess(v,a):
        return fuse([Signal("visual",v,1,(.45,.55)),Signal("acoustic",a,1,(.45,.55))],strategy)
    assert assess(0,None)["prediction"]=="UNCERTAIN"
    assert assess(None,None)["confidence"] is None
    assert assess(.95,.95)["prediction"]=="FAKE"
    assert assess(.1,.05)["prediction"]=="REAL"
    assert assess(.98,.03)["prediction"]==("FAKE" if strategy==STRATEGIES[0] else "UNCERTAIN")
    assert classify(.45,(.45,.55))==classify(.55,(.45,.55))=="UNCERTAIN"


def test_model_slot_releases_before_loading():
    class Model: pass
    slot=ModelSlot()
    with slot.use("one",Model) as model:
        ref=weakref.ref(model)
    del model
    def second():
        gc.collect()
        assert ref() is None
        return Model()
    with slot.use("two",second): pass
    slot.clear()
    assert slot.name is None


def test_short_text_and_missing_models(tmp_path,synthetic_video_mp4,monkeypatch):
    pipe=InferencePipeline()
    assert pipe.predict_text("Hello")["confidence"] is None
    def fail(_): raise OSError("offline weights")
    monkeypatch.setattr(serving,"load_visual",fail)
    r=PredictResponse(**pipe.predict(synthetic_video_mp4))
    assert r.prediction=="UNCERTAIN" and r.video.score is None and r.audio.score is None
    assert "offline weights" in r.video.reason
    assert not pipe.models_ready


def test_asr_caption_api_contract(monkeypatch,synthetic_audio_wav):
    pipe=InferencePipeline()
    pipe.settings.asr_enabled=True
    phrase="I took my bike out for a ride early this morning."
    monkeypatch.setattr(serving,"acoustic",lambda *args:dict(score=0.,available=True,
                        model="fixture",usable_seconds=2,duration_seconds=2))
    monkeypatch.setattr(serving,"transcribe",lambda *args:(phrase,"en"))
    monkeypatch.setattr(serving,"analyze_text",lambda *args:dict(score=.9,model="fixture",
                        available=True,word_count=12,spans=[dict(start=0,end=len(phrase),
                        text=phrase,score=.9,new_tokens=12)]))
    result=PredictResponse(**pipe.predict_audio(synthetic_audio_wav,phrase))
    assert result.cross_modal.transcript==phrase
    assert result.cross_modal.wer==0
    assert result.text.spans[0].score==.9
    assert result.audio.score==0 and result.prediction=="REAL"


def test_sentence_evidence_offsets(monkeypatch):
    from src.inference import language
    text = "One sentence. Another sentence contains an n! Final words"
    monkeypatch.setattr(language, "score_chunks", lambda *args: (.72, []))
    spans = language.sentence_evidence(text, ())
    assert [v["text"] for v in spans] == ["One sentence.","Another sentence contains an n!","Final words"]
    assert all(text[v["start"]:v["end"]] == v["text"] for v in spans)


def test_timed_caption_clips_to_analysis():
    from src.inference.language import caption_reference
    raw = "WEBVTT\n\n00:00.000 --> 00:04.000\nFirst cue\n\n00:31.000 --> 00:33.000\nLater cue"
    assert caption_reference(raw) == "First cue"


def test_benchmark_abstentions_and_exports(tmp_path):
    from scripts.evaluate_benchmark import export, metrics
    rows = [dict(path="first", label="REAL", prediction="FAKE", score=.8,
                 method="visual_only", cohort="video", reason=""),
            dict(path="second", label="FAKE", prediction="UNCERTAIN", score=.5,
                 method="visual_only", cohort="video", reason="Gray zone")]
    report=metrics(rows)
    assert report["accuracy"]==0 and report["abstention_rate"]==.5
    assert report["confusion_matrix"]==[[0,1],[0,0]]
    assert report["roc_auc"]==0 and report["scored"]==2
    export(rows,tmp_path)
    assert (tmp_path/"error_analysis.md").read_text().index("first") < (tmp_path/"error_analysis.md").read_text().index("second")
    assert metrics([dict(rows[1],score=None)])["roc_auc"] is None


def test_selected_acoustic_checkpoint_reaches_response_and_readiness(tmp_path):
    from src.config import load_config
    config = load_config(extra_overrides={"inference": {"audio_model": "Hemgg/Deepfake-audio-detection"}})
    pipeline = InferencePipeline(config)
    result = pipeline.predict_audio(tmp_path / "missing.wav")
    assert result["audio"]["model"] == config.inference.audio_model
    assert result["audio"]["score"] is None
    assert pipeline.get_readiness()["audio_model"]["model_name"] == config.inference.audio_model
