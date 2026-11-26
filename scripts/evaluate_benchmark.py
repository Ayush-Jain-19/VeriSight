"""Inference-only evaluation. One pass per sample, identical branch evidence for both rules."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score)

from src.inference.decision import STRATEGIES, classify, fuse
from src.inference.pipeline import InferencePipeline

LABELS = ["REAL","FAKE"]


def metrics(rows):
    y = [r["label"] for r in rows]
    pred = [r["prediction"] for r in rows]
    covered = [i for i,v in enumerate(pred) if v in LABELS]
    yt, yp = [y[i] for i in covered], [pred[i] for i in covered]
    ranked = [(int(r["label"] == "FAKE"), r["score"]) for r in rows if r["score"] is not None]
    report = classification_report(yt,yp,labels=LABELS,output_dict=True,zero_division=0) if yt else {}
    result = dict(n=len(rows), decided=len(yt), scored=len(ranked), accuracy=accuracy_score(yt,yp) if yt else None,
                  precision=report.get("FAKE",{}).get("precision"), recall=report.get("FAKE",{}).get("recall"),
                  macro_f1=report.get("macro avg",{}).get("f1-score"),
                  weighted_f1=report.get("weighted avg",{}).get("f1-score"), classwise=report,
                  confusion_matrix=confusion_matrix(yt,yp,labels=LABELS).tolist() if yt else [[0,0],[0,0]],
                  abstentions_by_class={v:sum(a==v and b=="UNCERTAIN" for a,b in zip(y,pred)) for v in LABELS},
                  abstention_rate=1-len(yt)/len(rows), roc_auc=None, pr_auc=None)
    if len({v for v,_ in ranked}) == 2:
        truth, scores = zip(*ranked)
        precision, recall, _ = precision_recall_curve(truth,scores)
        result.update(roc_auc=roc_auc_score(truth,scores), pr_auc=auc(recall,precision))
    return result


def evaluate_sample(pipe, path, caption):
    suffix = path.suffix.lower()
    if suffix == ".txt":
        r = pipe.predict_text(path.read_text(encoding="utf-8-sig"))
        yield "text", "text_only", r["prediction"], r["text"]["score"], r["text"].get("reason")
        return
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        r = pipe.predict_image(path)
        yield "image", "image_only", r["prediction"], r["image"]["score"], r["image"].get("reason")
        return

    video = suffix in {".mp4", ".avi", ".mov", ".webm", ".mkv"}
    if not video and suffix not in {".wav", ".m4a", ".mp3", ".aac", ".flac", ".ogg"}:
        raise ValueError(f"Unsupported benchmark file: {path}")
    cohort = "video" if video else "audio"
    r = pipe.predict(path, caption) if video else pipe.predict_audio(path, caption)
    signals = pipe.signals(r, caption, video=video)
    for signal in signals:
        key = {"visual": "video", "acoustic": "audio", "caption mismatch": "cross_modal"}[signal.name]
        method = signal.name.replace(" ", "_") + "_only"
        yield cohort, method, classify(signal.score, signal.band), signal.score, r[key].get("reason")
    text = r.get("text") or {}
    yield (cohort, "text_only", classify(text.get("score"), pipe.settings.text_band),
           text.get("score"), text.get("reason") or r["cross_modal"]["transcript_status"])
    for strategy in STRATEGIES:
        fused = fuse(signals, strategy, pipe.settings.fusion_band)
        yield cohort, strategy, fused["prediction"], fused["fusion"]["ranking_score"], fused["disagreement"]


def write_csv(path, fields, rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export(rows, dest):
    dest.mkdir(parents=True,exist_ok=True)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["cohort"],r["method"])].append(r)
    summary = {f"{c}/{m}":metrics(g) for (c,m),g in groups.items()}
    (dest/"metrics.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    fields=list(rows[0])
    write_csv(dest/"predictions.csv",fields,rows)
    scalar = ["n","decided","scored","accuracy","precision","recall","macro_f1","weighted_f1","roc_auc","pr_auc","abstention_rate"]
    write_csv(dest/"ablation.csv",["cohort_method"]+scalar,
              [dict(cohort_method=k, **{f:v[f] for f in scalar}) for k,v in summary.items()])

    def priority(r):
        score = r["score"]
        if r["prediction"] in LABELS and r["prediction"] != r["label"]:
            return (0,-abs(score-.5) if score is not None else 0)
        return (1 if r["prediction"]=="UNCERTAIN" else 2, abs(score-.5) if score is not None else 1)
    hardest = {}
    for row in sorted(rows,key=priority):
        hardest.setdefault(row["path"],row)
        if len(hardest)==20:
            break
    write_csv(dest/"error_analysis.csv",fields,hardest.values())
    sections=["# Difficult samples", "One row per distinct file. Wrong decisions come first, then abstentions and closest decided cases. Inspect the media and reference label before attributing a cause."]
    for r in hardest.values():
        sections.append(f"## {r['path']}\n\nReference {r['label']} · {r['method']} · {r['prediction']} · score {r['score']}\n\n{r['reason'] or 'No branch diagnostic.'}")
        if r.get("evidence"):
            sections.append("Branch scores: " + r["evidence"])
    (dest/"error_analysis.md").write_text("\n\n".join(sections)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest",type=Path)
    parser.add_argument("--root",type=Path)
    parser.add_argument("--output",type=Path,default=Path("reports"))
    parser.add_argument("--asr",action="store_true")
    args=parser.parse_args()
    pipe=InferencePipeline()
    if args.asr:
        pipe.settings.asr_enabled=True
    if args.manifest.is_dir():
        samples=[dict(path=str(p.resolve()),label=label) for label in LABELS
                 for p in sorted((args.manifest/label).rglob("*")) if p.is_file()]
    else:
        with args.manifest.open(encoding="utf-8-sig",newline="") as f:
            samples=list(csv.DictReader(f))
    rows=[]
    for i,s in enumerate(samples,1):
        label=(s.get("overall_label") or s.get("label") or "").upper()
        location=s.get("video_path") or s.get("path")
        if not location or label not in LABELS:
            parser.error(f"row {i}: expected path/video_path and REAL/FAKE label/overall_label")
        path=(args.root or args.manifest.parent)/location
        if not path.is_file():
            parser.error(f"Missing file on row {i}: {path}")
        observations = list(evaluate_sample(pipe, path, s.get("caption")))
        branch_scores = {method: score for _, method, _, score, _ in observations if method.endswith("_only")}
        diagnostics = json.dumps(branch_scores, ensure_ascii=False)
        for cohort,method,pred,score,reason in observations:
            rows.append(dict(path=str(path),cohort=cohort,method=method,label=label,
                             prediction=pred,score=score,reason=reason,evidence=diagnostics))
    if not rows:
        parser.error("No labeled files")
    export(rows,args.output)
    print(f"{len(samples)} samples, {len(rows)} assessments -> {args.output}")


if __name__=="__main__":
    main()
