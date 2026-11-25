import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_architecture_diagram(output_path: str):
    fig, ax = plt.subplots(figsize=(14, 9.5), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.5)
    ax.axis('off')

    # Color palette
    c_bg = '#f8fafc'
    c_client = '#e0f2fe'  # Light sky
    c_client_border = '#0284c7'
    c_api = '#f1f5f9'     # Slate
    c_api_border = '#475569'
    c_pre = '#fef3c7'     # Amber
    c_pre_border = '#d97706'
    c_dl = '#ede9fe'      # Purple
    c_dl_border = '#7c3aed'
    c_dec = '#dcfce7'     # Green
    c_dec_border = '#16a34a'
    c_out = '#ffe4e6'     # Rose
    c_out_border = '#e11d48'

    # Background canvas
    fig.patch.set_facecolor('#ffffff')

    # Title
    ax.text(7.0, 9.1, "VeriSight: Multimodal Media Forensics System Architecture",
            ha='center', va='center', fontsize=15, fontweight='bold', color='#0f172a')
    ax.text(7.0, 8.8, "Decoupled Cross-Modal Evidence Pipeline with Selective Abstention (CPU-Optimized)",
            ha='center', va='center', fontsize=10, fontstyle='italic', color='#475569')

    # Layer 1: Client UI
    box_ui = patches.FancyBboxPatch((0.5, 7.3), 13.0, 1.2, boxstyle="round,pad=0.1",
                                    facecolor=c_client, edgecolor=c_client_border, linewidth=1.5)
    ax.add_patch(box_ui)
    ax.text(7.0, 8.2, "CLIENT INTERFACE: React 18 + Vite ('The Screening Room' Professional Theme)",
            ha='center', va='center', fontsize=11, fontweight='bold', color='#0369a1')

    tabs = [
        ("1. Video Tab\n(MP4, AVI, WebM)", 2.0),
        ("2. Audio Tab\n(WAV, MP3, AAC)", 5.3),
        ("3. Image Tab\n(Face-Swap vs General AI)", 8.7),
        ("4. Text Tab\n(Passage Forensics)", 12.0)
    ]
    for label, x in tabs:
        t_box = patches.FancyBboxPatch((x - 1.3, 7.4), 2.6, 0.65, boxstyle="round,pad=0.05",
                                       facecolor='#ffffff', edgecolor=c_client_border, linewidth=1.0)
        ax.add_patch(t_box)
        ax.text(x, 7.72, label, ha='center', va='center', fontsize=8, fontweight='semibold', color='#0c4a6e')

    # Layer 2: API Gateway
    box_api = patches.FancyBboxPatch((0.5, 5.9), 13.0, 0.9, boxstyle="round,pad=0.1",
                                     facecolor=c_api, edgecolor=c_api_border, linewidth=1.5)
    ax.add_patch(box_api)
    ax.text(7.0, 6.55, "REST API GATEWAY: FastAPI / Uvicorn Server (Port 8000)",
            ha='center', va='center', fontsize=10.5, fontweight='bold', color='#1e293b')

    endpoints = [
        ("POST /api/predict\n(Multipart Video)", 2.0),
        ("POST /api/predict/audio\n(Audio Stream)", 5.3),
        ("POST /api/predict/image\n(?check_type=...)", 8.7),
        ("POST /api/predict/text\n(JSON Payload)", 12.0)
    ]
    for label, x in endpoints:
        ep_box = patches.FancyBboxPatch((x - 1.3, 6.0), 2.6, 0.45, boxstyle="round,pad=0.05",
                                        facecolor='#ffffff', edgecolor=c_api_border, linewidth=1.0)
        ax.add_patch(ep_box)
        ax.text(x, 6.22, label, ha='center', va='center', fontsize=7.5, color='#334155')

    # Layer 3: Preprocessing & Memory Bounding
    box_pre = patches.FancyBboxPatch((0.5, 4.3), 13.0, 1.1, boxstyle="round,pad=0.1",
                                     facecolor=c_pre, edgecolor=c_pre_border, linewidth=1.5)
    ax.add_patch(box_pre)
    ax.text(7.0, 5.15, "MODALITY PREPROCESSING & CPU MEMORY BOUNDING LAYER",
            ha='center', va='center', fontsize=10.5, fontweight='bold', color='#92400e')

    pre_items = [
        ("Video Preprocessing\n- Uniform 8-frame sampling\n- MTCNN face detection\n- 18% margin crop (224x224)\n- FFmpeg 16kHz mono audio extract", 2.0),
        ("Audio Preprocessing\n- 16 kHz mono resampling\n- 5s chunking (80k samples)\n- Bounded memory windowing\n- Silence & RMS gating (<0.5s)", 5.3),
        ("Image Preprocessing\n[Check 1: Face-Swap]\n- Face bbox detection + 18% pad\n- 224x224 RGB float32 [0..255]\n[Check 2: General AI]\n- Swin AutoImageProcessor", 8.7),
        ("Text Preprocessing\n- 512-token sequence crop\n- Length gate (<50 chars null)\n- Word count gate (<10 words)\n- Non-Latin script gating (>35%)", 12.0)
    ]
    for label, x in pre_items:
        p_box = patches.FancyBboxPatch((x - 1.45, 4.4), 2.9, 0.65, boxstyle="round,pad=0.05",
                                       facecolor='#ffffff', edgecolor=c_pre_border, linewidth=1.0)
        ax.add_patch(p_box)
        ax.text(x, 4.72, label, ha='center', va='center', fontsize=6.5, color='#78350f')

    # Layer 4: Deep Learning Inference Checkpoints
    box_dl = patches.FancyBboxPatch((0.5, 2.5), 13.0, 1.3, boxstyle="round,pad=0.1",
                                    facecolor=c_dl, edgecolor=c_dl_border, linewidth=1.5)
    ax.add_patch(box_dl)
    ax.text(7.0, 3.55, "TASK-TRAINED DEEP LEARNING INFERENCE MODELS (Sequential CPU Execution)",
            ha='center', va='center', fontsize=10.5, fontweight='bold', color='#5b21b6')

    models = [
        ("Visual Face-Swap Backbone\nSamson5827/deepfakeguard-v2\n(ONNX Runtime CPU)\nOutput: P(Real) -> 1 - P(Real)\nMedian frame score over usable faces", 2.0),
        ("Acoustic Speech Classifier\nMelodyMachine/Deepfake-audio\n(Wav2Vec 2.0 Backbone)\nOutput: id2label {0:fake, 1:real}\nMedian score across active windows", 5.3),
        ("Image Synthesis Checkers\n1. DeepfakeGuard-v2 (ONNX)\n   Artifacts on facial boundary\n2. Organika/sdxl-detector\n   Swin Transformer (Whole Image)", 8.7),
        ("Linguistic Style Classifier\nHello-SimpleAI/chatgpt-detector\n(RoBERTa Sequence Classifier)\nOutput: id2label {0:Human, 1:GPT}\nExperimental writing estimate", 12.0)
    ]
    for label, x in models:
        m_box = patches.FancyBboxPatch((x - 1.45, 2.6), 2.9, 0.85, boxstyle="round,pad=0.05",
                                       facecolor='#ffffff', edgecolor=c_dl_border, linewidth=1.0)
        ax.add_patch(m_box)
        ax.text(x, 3.02, label, ha='center', va='center', fontsize=6.5, color='#4c1d95')

    # Layer 5: Evidence, Cross-Modal Disagreement & Findings
    box_dec = patches.FancyBboxPatch((0.5, 0.4), 13.0, 1.6, boxstyle="round,pad=0.1",
                                     facecolor=c_dec, edgecolor=c_dec_border, linewidth=1.5)
    ax.add_patch(box_dec)
    ax.text(7.0, 1.8, "EVIDENCE EVALUATION, DISAGREEMENT REASONING & SELECTIVE ABSTENTION",
            ha='center', va='center', fontsize=10.5, fontweight='bold', color='#14532d')

    dec_boxes = [
        ("Decoupled Video Reporting\n* Visual Score (Faces X/8)\n* Acoustic Score (Speech sec)\n* Cross-Modal Disagreement Callout\n* Synthetic Fusion REMOVED (null)", 3.4),
        ("Strict Selective Abstention\n* Missing face -> 'No usable face'\n* Silence -> 'Silence detected'\n* Short text -> 'Passage too short'\n* Strict Zero-Mock Policy (Score=null)", 7.0),
        ("Findings Sheet Presentation\n* Modality Verdict (REAL/FAKE/UNCERTAIN)\n* Dynamic Confidence Bar\n* Component Metric Badges\n* Subtitles Non-Voting / Neutral", 10.6)
    ]
    for label, x in dec_boxes:
        d_box = patches.FancyBboxPatch((x - 1.65, 0.55), 3.3, 1.05, boxstyle="round,pad=0.05",
                                       facecolor='#ffffff', edgecolor=c_dec_border, linewidth=1.0)
        ax.add_patch(d_box)
        ax.text(x, 1.07, label, ha='center', va='center', fontsize=7.2, color='#14532d')

    # Connective arrows
    arrow_props = dict(arrowstyle="->", color="#64748b", lw=1.5)
    for x in [2.0, 5.3, 8.7, 12.0]:
        ax.annotate("", xy=(x, 6.8), xytext=(x, 7.3), arrowprops=arrow_props)
        ax.annotate("", xy=(x, 5.4), xytext=(x, 5.9), arrowprops=arrow_props)
        ax.annotate("", xy=(x, 3.8), xytext=(x, 4.3), arrowprops=arrow_props)

    # From DL layer to Decision layer
    ax.annotate("", xy=(3.4, 2.0), xytext=(2.0, 2.5), arrowprops=arrow_props)
    ax.annotate("", xy=(3.4, 2.0), xytext=(5.3, 2.5), arrowprops=arrow_props)
    ax.annotate("", xy=(7.0, 2.0), xytext=(7.0, 2.5), arrowprops=arrow_props)
    ax.annotate("", xy=(10.6, 2.0), xytext=(8.7, 2.5), arrowprops=arrow_props)
    ax.annotate("", xy=(10.6, 2.0), xytext=(12.0, 2.5), arrowprops=arrow_props)

    plt.tight_layout()
    output_p = Path(output_path).resolve()
    output_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_p), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagram saved successfully to: {output_p}")

if __name__ == "__main__":
    create_architecture_diagram(r"C:\Users\hp\.gemini\antigravity\scratch\multimodal-deepfake-detection\submission_report\architecture_diagram.png")
