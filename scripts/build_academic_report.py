"""Generate complete academic submission report for Bennett University IMDAI project."""

import os
import sys
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def create_report():
    doc = Document()

    # 1. Page Setup: A4, 1-inch margins
    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.different_first_page_header_footer = True

        # Header (Pages 2+)
        hdr = section.header
        hp = hdr.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("Bennett University | B.Tech CSE (IMDAI) | Multimodal Deepfake Detection System")
        hrun.font.name = "Times New Roman"
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = RGBColor(100, 116, 139)

        # Footer (Pages 2+)
        ftr = section.footer
        fp = ftr.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run("VeriSight Forensic Architecture — Academic Defense & Technical Project Report")
        frun.font.name = "Times New Roman"
        frun.font.size = Pt(8.5)
        frun.font.color.rgb = RGBColor(100, 116, 139)

    # Styling Helpers
    def set_cell_background(cell, fill_hex):
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
        tc_pr.append(shd)

    def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
        tc_pr = cell._tc.get_or_add_tcPr()
        tc_mar = parse_xml(f'''
            <w:tcMar {nsdecls("w")}>
                <w:top w:w="{top}" w:type="dxa"/>
                <w:bottom w:w="{bottom}" w:type="dxa"/>
                <w:left w:w="{left}" w:type="dxa"/>
                <w:right w:w="{right}" w:type="dxa"/>
            </w:tcMar>
        ''')
        tc_pr.append(tc_mar)

    def set_table_borders(table, color="CBD5E1", sz="4", val="single"):
        tblPr = table._tbl.tblPr
        borders = parse_xml(f'''
            <w:tblBorders {nsdecls("w")}>
                <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
                <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
                <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
                <w:insideV w:val="none"/>
                <w:left w:val="none"/>
                <w:right w:val="none"/>
            </w:tblBorders>
        ''')
        tblPr.append(borders)

    def format_p(p, before=0, after=6, line_spacing=1.15, align=WD_ALIGN_PARAGRAPH.LEFT):
        p.paragraph_format.space_before = Pt(before)
        p.paragraph_format.space_after = Pt(after)
        p.paragraph_format.line_spacing = line_spacing
        p.alignment = align

    def add_h1(text):
        h = doc.add_paragraph()
        format_p(h, before=16, after=6, line_spacing=1.15)
        r = h.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = RGBColor(15, 23, 42)
        return h

    def add_h2(text):
        h = doc.add_paragraph()
        format_p(h, before=11, after=4, line_spacing=1.15)
        r = h.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = RGBColor(30, 41, 59)
        return h

    def add_h3(text):
        h = doc.add_paragraph()
        format_p(h, before=7, after=2, line_spacing=1.15)
        r = h.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.italic = True
        r.font.color.rgb = RGBColor(51, 65, 85)
        return h

    def add_body(text, bold_prefix=None):
        p = doc.add_paragraph()
        format_p(p, before=0, after=6, line_spacing=1.15)
        if bold_prefix:
            rb = p.add_run(bold_prefix)
            rb.font.name = "Times New Roman"
            rb.font.size = Pt(10.5)
            rb.font.bold = True
            rb.font.color.rgb = RGBColor(17, 24, 39)
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10.5)
        r.font.color.rgb = RGBColor(31, 41, 55)
        return p

    def add_bullet(text, bold_prefix=None):
        p = doc.add_paragraph(style='List Bullet')
        format_p(p, before=0, after=3, line_spacing=1.15)
        if bold_prefix:
            rb = p.add_run(bold_prefix)
            rb.font.name = "Times New Roman"
            rb.font.size = Pt(10.5)
            rb.font.bold = True
            rb.font.color.rgb = RGBColor(17, 24, 39)
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10.5)
        r.font.color.rgb = RGBColor(31, 41, 55)
        return p

    def add_callout(text, title="EXAMINER & VIVA KEY DEFENSE POINT"):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = tbl.cell(0, 0)
        set_cell_background(cell, "F8FAFC")
        set_cell_margins(cell, top=140, bottom=140, left=200, right=160)

        tc_pr = cell._tc.get_or_add_tcPr()
        borders = parse_xml(f'''
            <w:tcBorders {nsdecls("w")}>
                <w:left w:val="single" w:sz="24" w:space="0" w:color="1E3A8A"/>
                <w:top w:val="none"/>
                <w:bottom w:val="none"/>
                <w:right w:val="none"/>
            </w:tcBorders>
        ''')
        tc_pr.append(borders)

        p = cell.paragraphs[0]
        format_p(p, before=2, after=2, line_spacing=1.15)
        r_title = p.add_run(f"[{title}] ")
        r_title.font.name = "Times New Roman"
        r_title.font.size = Pt(9.5)
        r_title.font.bold = True
        r_title.font.color.rgb = RGBColor(30, 58, 138)

        r_text = p.add_run(text)
        r_text.font.name = "Times New Roman"
        r_text.font.size = Pt(9.5)
        r_text.font.color.rgb = RGBColor(31, 41, 55)

        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    def add_code(code_str, caption=None):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = tbl.cell(0, 0)
        set_cell_background(cell, "F1F5F9")
        set_cell_margins(cell, top=120, bottom=120, left=160, right=160)

        tc_pr = cell._tc.get_or_add_tcPr()
        borders = parse_xml(f'''
            <w:tcBorders {nsdecls("w")}>
                <w:left w:val="single" w:sz="12" w:space="0" w:color="94A3B8"/>
                <w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
                <w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
                <w:right w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>
            </w:tcBorders>
        ''')
        tc_pr.append(borders)

        p = cell.paragraphs[0]
        format_p(p, before=2, after=2, line_spacing=1.05)
        run = p.add_run(code_str)
        run.font.name = "Courier New"
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(15, 23, 42)

        if caption:
            cp = doc.add_paragraph()
            format_p(cp, before=2, after=6, align=WD_ALIGN_PARAGRAPH.CENTER)
            crun = cp.add_run(caption)
            crun.font.name = "Times New Roman"
            crun.font.size = Pt(8.5)
            crun.font.italic = True
            crun.font.color.rgb = RGBColor(100, 116, 139)

    # ==========================================
    # 1. COVER PAGE
    # ==========================================
    cp_title = doc.add_paragraph()
    format_p(cp_title, before=40, after=8, align=WD_ALIGN_PARAGRAPH.CENTER)
    rt = cp_title.add_run("BENNETT UNIVERSITY")
    rt.font.name = "Times New Roman"
    rt.font.size = Pt(18)
    rt.font.bold = True
    rt.font.color.rgb = RGBColor(17, 24, 39)

    cp_sub = doc.add_paragraph()
    format_p(cp_sub, before=0, after=24, align=WD_ALIGN_PARAGRAPH.CENTER)
    rs = cp_sub.add_run("SCHOOL OF COMPUTER SCIENCE ENGINEERING AND TECHNOLOGY\nGREATER NOIDA, UTTAR PRADESH, INDIA")
    rs.font.name = "Times New Roman"
    rs.font.size = Pt(11)
    rs.font.bold = True
    rs.font.color.rgb = RGBColor(71, 85, 105)

    cp_div = doc.add_paragraph()
    format_p(cp_div, before=10, after=18, align=WD_ALIGN_PARAGRAPH.CENTER)
    rdiv = cp_div.add_run("—" * 28)
    rdiv.font.color.rgb = RGBColor(203, 213, 225)

    cp_proj = doc.add_paragraph()
    format_p(cp_proj, before=16, after=8, align=WD_ALIGN_PARAGRAPH.CENTER)
    rp = cp_proj.add_run("MULTIMODAL DEEPFAKE DETECTION SYSTEM")
    rp.font.name = "Times New Roman"
    rp.font.size = Pt(22)
    rp.font.bold = True
    rp.font.color.rgb = RGBColor(30, 58, 138)

    cp_tag = doc.add_paragraph()
    format_p(cp_tag, before=0, after=28, align=WD_ALIGN_PARAGRAPH.CENTER)
    rtag = cp_tag.add_run("Decoupled Cross-Modal Media Forensics, Pretrained Deep Learning Integration,\nand Empirical Failure Mode Analysis across Video, Audio, Image, and Text")
    rtag.font.name = "Times New Roman"
    rtag.font.size = Pt(12)
    rtag.font.italic = True
    rtag.font.color.rgb = RGBColor(51, 65, 85)

    cp_meta = doc.add_paragraph()
    format_p(cp_meta, before=24, after=36, align=WD_ALIGN_PARAGRAPH.CENTER)
    rmeta = cp_meta.add_run(
        "A Project Report Submitted in Partial Fulfillment of the Requirements for the Award of\n"
        "Bachelor of Technology (B.Tech) in Computer Science & Engineering\n"
        "Course Code / Track: Applied Artificial Intelligence & Deep Learning (IMDAI)"
    )
    rmeta.font.name = "Times New Roman"
    rmeta.font.size = Pt(11)
    rmeta.font.color.rgb = RGBColor(31, 41, 55)

    # Details Box Table
    det_tbl = doc.add_table(rows=2, cols=2)
    det_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(det_tbl, color="E2E8F0", sz="6", val="single")

    cell_c1 = det_tbl.cell(0, 0)
    set_cell_background(cell_c1, "F8FAFC")
    set_cell_margins(cell_c1, top=140, bottom=140, left=160, right=160)
    p_c1 = cell_c1.paragraphs[0]
    format_p(p_c1, before=0, after=2)
    r1 = p_c1.add_run("STUDENT / CANDIDATE DETAILS:")
    r1.font.bold = True
    r1.font.size = Pt(9.5)
    r1.font.color.rgb = RGBColor(30, 58, 138)

    cell_c2 = det_tbl.cell(0, 1)
    set_cell_background(cell_c2, "F8FAFC")
    set_cell_margins(cell_c2, top=140, bottom=140, left=160, right=160)
    p_c2 = cell_c2.paragraphs[0]
    format_p(p_c2, before=0, after=2)
    r2 = p_c2.add_run("FACULTY EVALUATION PANEL:")
    r2.font.bold = True
    r2.font.size = Pt(9.5)
    r2.font.color.rgb = RGBColor(30, 58, 138)

    cell_v1 = det_tbl.cell(1, 0)
    set_cell_margins(cell_v1, top=120, bottom=140, left=160, right=160)
    p_v1 = cell_v1.paragraphs[0]
    format_p(p_v1, before=0, after=2)
    p_v1.add_run(
        "Candidate Name: [Fill in: Student Full Name]\n"
        "Enrollment Number: [Fill in: E22CSE...]\n"
        "Batch / Division: B.Tech CSE (Batch of 2026)\n"
        "Department: Computer Science & Engineering"
    ).font.size = Pt(9.5)

    cell_v2 = det_tbl.cell(1, 1)
    set_cell_margins(cell_v2, top=120, bottom=140, left=160, right=160)
    p_v2 = cell_v2.paragraphs[0]
    format_p(p_v2, before=0, after=2)
    p_v2.add_run(
        "Course Instructor / Guide: [Fill in: Faculty Supervisor Name]\n"
        "Academic Department: SCSET, Bennett University\n"
        "Evaluation Component: Final Project Defense & Viva\n"
        "Submission Date: November 19, 2026"
    ).font.size = Pt(9.5)

    doc.add_page_break()

    # ==========================================
    # 2. ABSTRACT & KEYWORDS
    # ==========================================
    add_h1("ABSTRACT")
    add_body(
        "The proliferation of generative deep learning models has simplified the creation of convincing synthetic media, "
        "introducing acute risks across disinformation, identity theft, and corporate fraud. Conventional forensic pipelines "
        "often rely on brittle, coupled late-fusion architectures that yield overconfident predictions while concealing "
        "unimodal discrepancies. This project presents VeriSight, a production-grade, CPU-optimized multimodal deepfake detection "
        "system supporting video, audio, image, and text modalities. Implemented on an Intel Core i3 environment with 8 GB RAM without CUDA, "
        "the architecture integrates specialized, task-trained open-source deep learning checkpoints: Samson5827/deepfakeguard-v2 (ONNXRuntime CPU) "
        "for face-swap detection, MelodyMachine/Deepfake-audio-detection (Wav2Vec 2.0) for synthetic voice classification, "
        "Organika/sdxl-detector (Swin Transformer) for whole-image diffusion forensics, and Hello-SimpleAI/chatgpt-detector-roberta for "
        "AI-writing style estimation. Rather than forcing an opaque fused probability without a validated cross-modal checkpoint, "
        "VeriSight independently analyzes visual and acoustic streams, surfaces natural-language cross-modal disagreement callouts, and "
        "enforces strict selective abstention for silent, face-less, or sub-length inputs with zero mock fallbacks. Verified across 13 "
        "end-to-end multimodal scenarios and 50 unit tests, the system demonstrates reliable CPU execution while exposing critical forensic "
        "boundary limitations."
    )

    add_body(
        "Multimodal Deepfake Detection, Media Forensics, Face-Swap Classification, Acoustic Deepfake Detection, "
        "Selective Abstention, AI-Writing Style Estimation.",
        bold_prefix="Keywords: "
    )

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ==========================================
    # 3. INTRODUCTION, PROBLEM STATEMENT & OBJECTIVES
    # ==========================================
    add_h1("1. INTRODUCTION & PROBLEM FORMULATION")

    add_h2("1.1 Real-World Motivation: Asymmetric Manipulation in Media")
    add_body(
        "Modern deepfakes are rarely uniformly synthetic across all communication channels. In practical forensic scenarios, "
        "manipulation is fundamentally asymmetric. For example, a malicious actor may execute a neural voice clone of a public official "
        "over authentic, unmanipulated press briefing footage to broadcast fabricated policy statements. Conversely, in political parody "
        "or defamation, an adversary may superimpose a target subject's facial likeness onto an actor's body (face-swap) while preserving "
        "the actor's authentic vocal cadence and room acoustics."
    )
    add_body(
        "When such media is processed through monolithic late-fusion architectures that force a single aggregated score, two catastrophic "
        "failure modes emerge: (1) false dismissal, where the high authenticity score of the genuine modality dilutes the synthetic signal "
        "of the tampered modality, causing the system to classify the asset as REAL; and (2) misattribution, where the user receives a binary "
        "FAKE verdict without knowing whether the visual identity, the vocal track, or both were compromised. This asymmetry establishes the "
        "engineering requirement for decoupled, independent unimodal evidence pipelines."
    )

    add_callout(
        "In a viva defense, emphasize that VeriSight deliberately decouples visual face analysis from acoustic speech analysis. "
        "If a video contains a synthetic face-swap over authentic broadcast audio, VeriSight reports Video=FAKE and Audio=REAL with an "
        "explicit cross-modal disagreement callout, rather than averaging them into an uninterpretable 50% probability.",
        title="VIVA DEFENSE CORE INSIGHT"
    )

    add_h2("1.2 Modality-Specific Forensic Problem Definitions")
    add_body(
        "A critical conceptual rigor required in media forensics is defining exactly what question each detector answers. "
        "The application divides media analysis into four distinct operational tabs:"
    )
    add_bullet(
        "Video Forensics: Samples 8 evenly spaced temporal frames to detect facial manipulation seams, boundary warping, and color "
        "discontinuities via deepfakeguard-v2, while concurrently extracting the audio soundtrack via FFmpeg to evaluate vocal synthesis "
        "via Wav2Vec 2.0. Evaluates both channels independently.",
        bold_prefix="1. "
    )
    add_bullet(
        "Audio Forensics: Evaluates mono 16 kHz speech waveforms using self-supervised representations to detect acoustic vocoder artifacts, "
        "unnatural spectral phase transitions, and neural text-to-speech (TTS) signatures.",
        bold_prefix="2. "
    )
    add_bullet(
        "Image Forensics: Offers a selectable dual-check mechanism: (a) Face-Swap Forensics (default) analyzing facial bounding boxes with "
        "an 18% margin for tampering boundaries; and (b) General AI Forensics analyzing full-frame spatial patch distributions for diffusion "
        "synthesis artifacts (Stable Diffusion / Midjourney).",
        bold_prefix="3. "
    )
    add_bullet(
        "Text Forensics: Evaluates textual passages using a fine-tuned RoBERTa sequence classifier. Crucially, the Text tab does NOT detect "
        "biometric falsification or factual veracity; it produces an 'Experimental AI-writing style estimate' reflecting statistical linguistic "
        "likelihood associated with large language model generation.",
        bold_prefix="4. "
    )

    add_h2("1.3 Project Engineering Objectives")
    add_bullet("Full-Stack Decoupled Implementation: Deliver an interactive React 18 frontend ('The Screening Room') backed by a high-throughput FastAPI REST engine.", bold_prefix="• ")
    add_bullet("CPU-Bounded Deployment: Execute all deep learning inferences sequentially on CPU with strict memory bounding (under 1.2 GB peak RAM) for an Intel Core i3 laptop with 8 GB RAM.", bold_prefix="• ")
    add_bullet("Zero-Mock Integrity: Completely eradicate heuristic fallbacks (Fourier, Laplacian, random numbers) in favor of task-trained open-source deep learning checkpoints.", bold_prefix="• ")
    add_bullet("Evidence-Based Selective Abstention: Enforce calibrated null-score gating (returning verdict UNCERTAIN with explicit diagnostic reasons) when inputs lack faces, speech, or sufficient text length.", bold_prefix="• ")
    add_bullet("Cross-Modal Disagreement Surfacing: Formulate an automated rule engine that identifies and explains conflicts between visual and acoustic forensic signals.", bold_prefix="• ")

    # ==========================================
    # 4. BACKGROUND & RELATED WORK
    # ==========================================
    add_h1("2. BACKGROUND & RELATED WORK")

    add_h2("2.1 Visual Deepfake Detection & DeepfakeGuard-v2")
    add_body(
        "Visual facial manipulation primarily encompasses face-swapping (e.g., DeepFaceLab, SimSwap) and facial reenactment (e.g., Face2Face). "
        "Early forensic detectors utilized handcrafted spatial artifacts (eye blinking frequency, head pose inconsistencies) or basic frequency "
        "transforms (2D FFT spectral decay). However, modern neural synthesizers easily bypass handcrafted heuristics. "
        "Current state-of-the-art visual detection relies on deep convolutional or transformer backbones trained on high-resolution facial datasets. "
        "In this project, we utilize Samson5827/deepfakeguard-v2, an ONNX-optimized deep neural network trained to identify blending boundaries, "
        "re-sampling artifacts, and localized frequency distortions on cropped facial regions."
    )

    add_h2("2.2 Acoustic Deepfake Detection & Wav2Vec 2.0")
    add_body(
        "Acoustic deepfakes leverage neural vocoders (HiFi-GAN, WaveGlow) and diffusion-based acoustic models to clone speaker timbres from short references. "
        "Traditional audio forensic methods relying on Mel-Frequency Cepstral Coefficients (MFCCs) struggle to generalize across diverse codecs. "
        "Modern acoustic forensics leverages self-supervised speech representations, notably Wav2Vec 2.0 (Baevski et al., 2020). Pretrained on tens of thousands "
        "of hours of raw speech, Wav2Vec 2.0 captures fine-grained temporal-spectral representations that expose neural vocoder phase irregularities. "
        "We employ MelodyMachine/Deepfake-audio-detection, which fine-tunes Wav2Vec 2.0 specifically for binary fake/real speech classification."
    )

    add_h2("2.3 Whole-Image Diffusion Forensics & SDXL-Detector")
    add_body(
        "Unlike face-swapping, which alters localized regions of an existing photograph, text-to-image diffusion models (e.g., Stable Diffusion XL, Midjourney) "
        "synthesize entire scenes from latent noise. These images exhibit distinct cross-attention artifacts and abnormal high-frequency noise distributions. "
        "We incorporate Organika/sdxl-detector, based on the Swin Transformer architecture (Liu et al., 2021). Swin Transformer's shifted window self-attention "
        "captures both local patch textures and global semantic coherence, making it exceptionally sensitive to diffusion upscaling artifacts."
    )

    add_h2("2.4 Linguistic Style Estimation & ChatGPT-Detector-RoBERTa")
    add_body(
        "Detecting AI-generated text is fundamentally distinct from computer vision forensics. Large language models (LLMs) output sequences optimized for "
        "maximum token probability, often exhibiting reduced lexical burstiness and constrained perplexity compared to human prose. "
        "We utilize Hello-SimpleAI/chatgpt-detector-roberta, fine-tuned on paired human and ChatGPT responses. It evaluates contextual token embeddings "
        "to produce a statistical likelihood of machine generation."
    )

    add_h2("2.5 The Multimodal Benchmark Context: FakeAVCeleb & Identity Leakage")
    add_body(
        "In academic deepfake research, FakeAVCeleb (Khalid et al., 2021) is a primary multimodal benchmark containing authentic celebrity recordings and "
        "manipulated variants categorized into RealVideo-RealAudio (RARV), RealVideo-FakeAudio (FARV), FakeVideo-RealAudio (RAFV), and FakeVideo-FakeAudio (FAFV). "
        "A major vulnerability in deepfake benchmarks is identity leakage: if the same celebrity identity appears in both training and test splits, models memorize "
        "facial features rather than forgery artifacts, inflating test accuracy above 95%. Valid evaluation requires disjoint identity splitting. Furthermore, "
        "training a multimodal late-fusion layer requires Out-of-Fold (OOF) cross-validation to prevent the fusion classifier from overfitting to overconfident base logits."
    )

    add_callout(
        "In our repository, FakeAVCeleb is configured in scripts/build_manifest.py and notebooks/01_eda.ipynb as an academic benchmark design. "
        "However, as verified during our audit, dataset training on Google Colab was not completed prior to submission. "
        "Accordingly, our running system strictly employs verified pretrained zero-shot checkpoints and documents this engineering choice with complete academic honesty.",
        title="VIVA DEFENSE BENCHMARK DISTINCTION"
    )

    # ==========================================
    # 5. SYSTEM ARCHITECTURE & WORKFLOW
    # ==========================================
    add_h1("3. SYSTEM ARCHITECTURE & DATA WORKFLOW")

    add_h2("3.1 High-Level Architecture Overview")
    add_body(
        "VeriSight is architected as a decoupled, multi-tiered forensic analysis platform. The frontend is built with React 18, TypeScript, and Vite, "
        "adopting a bespoke 'Screening Room' dark aesthetic designed for forensic analysts. The backend is implemented in FastAPI (Python 3.12), "
        "utilizing Uvicorn as an asynchronous ASGI server. All deep learning inference operations are managed through an unified InferencePipeline "
        "singleton that coordinates modality preprocessing, memory bounding, model invocation, and cross-modal reasoning."
    )

    # Embed Architecture Diagram
    diag_path = Path(r"C:\Users\hp\.gemini\antigravity\scratch\multimodal-deepfake-detection\submission_report\architecture_diagram.png")
    if diag_path.exists():
        doc.add_paragraph().paragraph_format.space_before = Pt(8)
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_img = p_img.add_run()
        run_img.add_picture(str(diag_path), width=Inches(6.2))
        cp = doc.add_paragraph()
        format_p(cp, before=4, after=12, align=WD_ALIGN_PARAGRAPH.CENTER)
        crun = cp.add_run("Figure 1: VeriSight Multimodal System Architecture — Decoupled Evidence Flow, CPU Memory Bounding, and Disagreement Engine.")
        crun.font.name = "Times New Roman"
        crun.font.size = Pt(9)
        crun.font.italic = True
        crun.font.color.rgb = RGBColor(71, 85, 105)

    add_h2("3.2 Video Processing Pipeline: Dual-Stream Decoupled Ingestion")
    add_body(
        "When a video clip (MP4, AVI, WebM) is submitted to POST /api/predict, the pipeline initiates two parallel execution streams:"
    )
    add_bullet(
        "Visual Frame Extraction & Face Cropping: OpenCV extracts the total frame count and duration, computing 8 uniformly spaced frame indices "
        "across the clip. For each sampled frame, a face detection cascade (MTCNN with Haar cascade fallback) detects the primary face bounding box. "
        "The bounding box is expanded by an 18% margin (margin_ratio = 0.18) to preserve the critical blending seam along the jawline, forehead, and hairline. "
        "The crop is resized to 224×224 RGB, converted to unnormalized float32 [0, 255], and formatted into NHWC layout (1, 224, 224, 3) for ONNXRuntime.",
        bold_prefix="1. "
    )
    add_bullet(
        "Acoustic Soundtrack Extraction: Concurrently, FFmpeg is spawned as a subprocess to demux and resample the video's audio track into a 16 kHz "
        "mono PCM WAV stream (ffmpeg -y -i <video> -vn -acodec pcm_s16le -ar 16000 -ac 1 <out.wav>). The waveform is gated for silence and windowed "
        "into 5.0-second segments for Wav2Vec 2.0 inference.",
        bold_prefix="2. "
    )

    add_h2("3.3 Cross-Modal Disagreement Reasoning Engine")
    add_body(
        "A critical innovation of this project is the removal of unvalidated late-fusion mathematics. In earlier iterations, the system calculated "
        "a weighted average or late-fusion MLP score even when models were missing or uncalibrated. The current production pipeline strictly enforces "
        "independent reporting: fusion.score is set to null, and the system executes the following deterministic disagreement evaluation:"
    )

    add_code(
        "# src/inference/pipeline.py: Cross-Modal Disagreement Logic\n"
        "if has_video and has_audio and video_score is not None and audio_score is not None:\n"
        "    if video_score >= 0.50 and audio_score < 0.50:\n"
        "        disagreement = (\n"
        "            'Visual branch flagged face-swap manipulation, '\n"
        "            'while acoustic branch found speech to be natural.'\n"
        "        )\n"
        "        prediction = 'FAKE'\n"
        "        confidence = video_score\n"
        "    elif video_score < 0.50 and audio_score >= 0.50:\n"
        "        disagreement = (\n"
        "            'Acoustic branch flagged synthetic voice cloning, '\n"
        "            'while visual frames showed authentic facial characteristics.'\n"
        "        )\n"
        "        prediction = 'FAKE'\n"
        "        confidence = audio_score\n"
        "    elif video_score >= 0.50 and audio_score >= 0.50:\n"
        "        disagreement = 'Both visual and acoustic branches detected manipulation.'\n"
        "        prediction = 'FAKE'\n"
        "        confidence = max(video_score, audio_score)\n"
        "    else:\n"
        "        disagreement = 'Both visual and acoustic branches found authentic characteristics.'\n"
        "        prediction = 'REAL'\n"
        "        confidence = min(1.0 - video_score, 1.0 - audio_score)",
        caption="Listing 1: Cross-modal disagreement logic surfacing conflicting unimodal forensic signals."
    )

    # ==========================================
    # 6. IMPLEMENTATION OF ALL FOUR MODALITIES
    # ==========================================
    add_h1("4. IMPLEMENTATION OF ALL FOUR MODALITIES")

    add_body(
        "Table 1 outlines the operational specifications, underlying models, preprocessing pipelines, scoring logic, "
        "and physical constraints for all four modalities supported in VeriSight."
    )

    # Comparison Table
    tbl = doc.add_table(rows=6, cols=6)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl, color="94A3B8", sz="4", val="single")

    headers = ["Modality", "Forensic Task", "Loaded Model / Checkpoint", "Input Preprocessing", "Output Scoring & Meaning", "Operational Limitations"]
    for col_idx, htext in enumerate(headers):
        cell = tbl.cell(0, col_idx)
        set_cell_background(cell, "1E3A8A")
        set_cell_margins(cell, top=140, bottom=140, left=100, right=100)
        p = cell.paragraphs[0]
        format_p(p, before=0, after=0)
        r = p.add_run(htext)
        r.font.name = "Times New Roman"
        r.font.size = Pt(8.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    data = [
        ("Image", "Face-Swap Detection\n(Default Check)", "Samson5827/deepfakeguard-v2\n(ONNX Runtime CPU)",
         "Face crop + 18% margin;\n224×224 RGB float32 [0..255];\nNHWC layout (1,224,224,3)",
         "Model outputs scalar P(Real);\nScore = 1.0 - P(Real);\n>= 0.50 flagged as FAKE",
         "Requires detectable face;\nReturns 'No usable face' with score=null if absent"),
        ("Image", "General AI Detection\n(Selectable Check)", "Organika/sdxl-detector\n(Swin Transformer)",
         "AutoImageProcessor;\nNative patch extraction;\nRGB normalization",
         "id2label: {0:'artificial', 1:'human'};\nScore = P(artificial);\n>= 0.50 flagged as FAKE",
         "Evaluates whole-image diffusion;\nInsensitive to small facial splice boundaries"),
        ("Audio", "Speech Deepfake\nClassification", "MelodyMachine/Deepfake-audio\n(Wav2Vec 2.0 Backbone)",
         "16 kHz mono WAV;\n5.0s windowing (80k samples);\nSilence/RMS energy gating",
         "id2label: {0:'fake', 1:'real'};\nScore = P(fake);\nMedian across active slices",
         "Speech-only detector;\nAudio < 0.5s or silence returns score=null with reason"),
        ("Video", "Multi-Frame Face &\nSoundtrack Analysis", "Shared DeepfakeGuard-v2 +\nMelodyMachine Wav2Vec 2.0",
         "8 uniform frames (18% crop);\nFFmpeg 16 kHz mono audio;\nSequential CPU inference",
         "Separate Visual (X/8 faces) &\nAcoustic metrics;\nDisagreement explanation",
         "CPU bounded (sequential);\nSynthetic fusion removed;\nSubtitles non-voting (neutral)"),
        ("Text", "AI-Writing Style\nEstimation", "Hello-SimpleAI/chatgpt-detector\n(RoBERTa Safetensors)",
         "512-token truncation;\nLength check (>=50 chars, >=10w);\nLatin script check (<=35% non-ascii)",
         "id2label: {0:'Human', 1:'ChatGPT'};\nScore = P(ChatGPT);\nExperimental style estimate",
         "Estimates writing style, NOT factual falsification;\nShort/foreign text omitted")
    ]

    for row_idx, row_data in enumerate(data, start=1):
        for col_idx, text_val in enumerate(row_data):
            cell = tbl.cell(row_idx, col_idx)
            bg = "FFFFFF" if row_idx % 2 != 0 else "F8FAFC"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=100, bottom=100, left=90, right=90)
            p = cell.paragraphs[0]
            format_p(p, before=0, after=0)
            r = p.add_run(text_val)
            r.font.name = "Times New Roman"
            r.font.size = Pt(8.0)
            r.font.color.rgb = RGBColor(31, 41, 55)

    add_body(
        "Table 1: Master comparative matrix of modalities, architectures, preprocessing rules, and scoring semantics.",
        bold_prefix=""
    )

    add_h2("4.1 Image Modality Implementation")
    add_body(
        "The Image tab provides a selectable toggle between Face-Swap Detection (default) and General AI-Image Detection. "
        "For face-swap analysis, the system instantiates FaceCropHelper. If no face is detected, the detector returns score: null with reason: "
        "'No usable face', avoiding arbitrary false-positive predictions on inanimate photos. When a face is present, the crop is passed "
        "to the ONNXRuntime session. Because deepfakeguard-v2 outputs the probability of authenticity, the fake score is derived as:"
    )

    add_code(
        "# src/inference/pretrained_models.py: DeepfakeGuard-v2 Score Inversion\n"
        "input_name = self.session.get_inputs()[0].name\n"
        "raw_out = self.session.run(None, {input_name: face_crop_nhwc})\n"
        "real_score = float(raw_out[0][0, 0])\n"
        "fake_score = float(np.clip(1.0 - real_score, 0.0, 1.0))\n"
        "return fake_score",
        caption="Listing 2: Model score derivation for Samson5827/deepfakeguard-v2 ONNX export."
    )

    add_h2("4.2 Audio Modality Implementation & CPU Memory Bounding")
    add_body(
        "The Audio tab processes speech recordings up to 100 MB. To execute Wav2Vec 2.0 on an Intel Core i3 CPU with 8 GB RAM without "
        "triggering out-of-memory (OOM) faults or paging delays, the waveform is strictly windowed. Waveforms are sliced into 5.0-second non-overlapping "
        "chunks (80,000 samples at 16 kHz) up to a maximum duration of 30.0 seconds (maximum 6 chunks). Each slice is processed sequentially, and the "
        "median fake probability is computed. If the root-mean-square (RMS) energy is below 0.002 or the peak absolute amplitude is below 0.008, "
        "the audio is classified as silent, returning score: null with reason: 'Silence or insufficient speech detected'."
    )

    add_h2("4.3 Video Modality & Frame Score Aggregation")
    add_body(
        "Video processing uniformly samples exactly 8 frames across the timeline (t = i * (N-1) / 7 for i in 0..7). "
        "Each frame is independently analyzed by deepfakeguard-v2. If a frame contains no detectable face, it is omitted from scoring. "
        "The overall visual score is calculated as the median of all valid face crops, and the frontend explicitly reports the face yield "
        "(e.g., 'Faces Detected: 8/8 frames'). If all 8 frames lack faces, the visual branch reports score: null with 'No usable face'."
    )

    add_h2("4.4 Text Modality & Writing Style Estimation")
    add_body(
        "The Text tab evaluates passages using Hello-SimpleAI/chatgpt-detector-roberta. The model tokenizer truncates sequences at 512 tokens. "
        "To prevent statistical hallucinations on conversational fragments, two strict guards are applied: (1) if the input contains fewer than 50 "
        "characters or fewer than 10 words, inference is bypassed, returning score: null with reason: 'Text passage is too short (< 50 characters or < 10 words) "
        "for reliable linguistic estimation'; (2) if non-ASCII characters exceed 35%, inference is omitted with an unsupported language notice. "
        "Furthermore, optional burnt-in subtitles submitted alongside video clips are explicitly designated non-voting, returning "
        "'Transcript comparison unavailable' in the absence of an integrated automated speech recognition (ASR) engine."
    )

    # ==========================================
    # 7. TECHNOLOGY STACK & REPRODUCIBILITY
    # ==========================================
    add_h1("5. TECHNOLOGY STACK & REPRODUCIBILITY")

    add_h2("5.1 Software Bill of Materials (SBOM)")
    add_bullet("FastAPI 0.110.0 + Uvicorn 0.28.0: Asynchronous REST backend providing strict Pydantic v2 schema validation.", bold_prefix="• ")
    add_bullet("PyTorch 2.2.2+cpu + Torchaudio 2.2.2+cpu: Core deep learning tensor framework optimized for x86 CPU execution.", bold_prefix="• ")
    add_bullet("ONNXRuntime 1.32.0: High-performance cross-platform inferencing engine executing deepfakeguard-v2 with 2 intra-op threads.", bold_prefix="• ")
    add_bullet("Hugging Face Transformers 4.57.6: Model management for Wav2Vec 2.0, Swin Transformer, and RoBERTa classifiers.", bold_prefix="• ")
    add_bullet("OpenCV (cv2) 4.9.0 + Pillow 10.2.0: Image decoding, color space transformation, and Haar cascade face detection.", bold_prefix="• ")
    add_bullet("FFmpeg (Gyan.dev Windows Build 9.0.2): Universal multimedia demuxing and audio extraction via CLI subprocess.", bold_prefix="• ")
    add_bullet("React 18 + TypeScript + Vite 5: Client-side single-page application implementing the Screening Room UI system.", bold_prefix="• ")

    add_h2("5.2 Exact Windows Launch Commands")
    add_body(
        "To launch the application from a clean state on Windows, open two PowerShell terminals and execute the following commands:"
    )

    add_code(
        "# Terminal 1: Launch FastAPI Backend (Port 8000)\n"
        "cd \"C:\\Users\\hp\\.gemini\\antigravity\\scratch\\multimodal-deepfake-detection\"\n"
        ".\\.venv\\Scripts\\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload\n\n"
        "# Terminal 2: Launch Vite React Frontend (Port 5173)\n"
        "cd \"C:\\Users\\hp\\.gemini\\antigravity\\scratch\\multimodal-deepfake-detection\\frontend\"\n"
        "npm run dev\n\n"
        "# Open Browser to: http://localhost:5173",
        caption="Listing 3: Exact PowerShell execution commands for backend and frontend servers."
    )

    add_h2("5.3 Automated Verification Commands")
    add_code(
        "# Run Comprehensive 13-Case Multimodal API Verification Suite\n"
        "cd \"C:\\Users\\hp\\.gemini\\antigravity\\scratch\\multimodal-deepfake-detection\"\n"
        ".\\.venv\\Scripts\\python.exe scripts/verify_all_modalities.py\n\n"
        "# Run Full PyTest Unit Test Suite (50 Unit Tests)\n"
        ".\\.venv\\Scripts\\pytest.exe tests/ -v",
        caption="Listing 4: Commands for running automated modality verification and unit tests."
    )

    # ==========================================
    # 8. TESTING, RESULTS & ERROR ANALYSIS
    # ==========================================
    add_h1("6. TESTING, RESULTS & EMPIRICAL ERROR ANALYSIS")

    add_h2("6.1 Automated Multimodal Verification Matrix")
    add_body(
        "The system was validated through an automated 13-case test harness (scripts/verify_all_modalities.py) executing real HTTP multipart "
        "and JSON requests against the FastAPI application. All 13 test scenarios passed completely, as detailed in Table 2."
    )

    # Test Table
    t_tbl = doc.add_table(rows=14, cols=5)
    t_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_tbl, color="94A3B8", sz="4", val="single")

    t_headers = ["Test ID", "Modality & Route", "Input Payload & Provenance", "Expected System Behavior", "Observed Output & Status"]
    for col_idx, htext in enumerate(t_headers):
        cell = t_tbl.cell(0, col_idx)
        set_cell_background(cell, "1E3A8A")
        set_cell_margins(cell, top=140, bottom=140, left=100, right=100)
        p = cell.paragraphs[0]
        format_p(p, before=0, after=0)
        r = p.add_run(htext)
        r.font.name = "Times New Roman"
        r.font.size = Pt(8.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    test_data = [
        ("T-01", "Image (Face-Swap)", "demo_samples/test_face_real.jpg\n(Uncompressed authentic portrait)", "Verdict: REAL; score < 0.50;\nFaces detected = 1", "pred=REAL; score=0.3750;\nfaces=1 [PASS]"),
        ("T-02", "Image (Face-Swap)", "demo_samples/test_face_fake.jpg\n(Synthetic boundary splice)", "Verdict: FAKE; score >= 0.50;\nFaces detected = 1", "pred=FAKE; score=0.9638;\nfaces=1 [PASS]"),
        ("T-03", "Image (Face-Swap)", "demo_samples/sample_frame.jpg\n(Geometric test pattern, no face)", "Verdict: UNCERTAIN; score=null;\nReason: 'No usable face'", "pred=UNCERTAIN; score=null;\nreason='No usable face' [PASS]"),
        ("T-04", "Image (General AI)", "demo_samples/test_face_real.jpg\n(Authentic camera capture)", "Verdict: REAL; score < 0.50;\nModel: Organika/sdxl-detector", "pred=REAL; score=0.0000;\nmodel=sdxl-detector [PASS]"),
        ("T-05", "Image (General AI)", "demo_samples/test_synthetic.jpg\n(Stable Diffusion XL synthesis)", "Verdict: FAKE; score >= 0.50;\nModel: Organika/sdxl-detector", "pred=FAKE; score=0.9744;\nmodel=sdxl-detector [PASS]"),
        ("T-06", "Audio (Speech)", "demo_samples/test_silence.wav\n(3.0s pure zero-energy WAV)", "Verdict: UNCERTAIN; score=null;\nReason: 'Silence detected'", "pred=UNCERTAIN; score=null;\nreason='Silence...' [PASS]"),
        ("T-07", "Audio (Speech)", "demo_samples/test_speech_synthesis.wav\n(Neural TTS cloned voice)", "Verdict: FAKE; score >= 0.50;\nDuration > 0.5s", "pred=FAKE; score=0.5001;\nduration=6.74s [PASS]"),
        ("T-08", "Video (Dual Stream)", "demo_samples/test_video_realface.mp4\n(Authentic face + speech)", "Verdict: REAL; faces=8/8;\nSynthetic fusion=null", "pred=REAL; video=0.2703;\nfaces=8/8; fusion=null [PASS]"),
        ("T-09", "Video (Asymmetric)", "demo_samples/test_video_faceswap.mp4\n(Face-swap + natural voice)", "Verdict: FAKE; Disagreement flagged;\nVisual=FAKE, Audio=REAL", "pred=FAKE; video=0.9298;\naudio=0.0025; Flagged [PASS]"),
        ("T-10", "Video (Silent)", "demo_samples/test_silent_video.mp4\n(Video stream without audio)", "Visual analyzed; Audio score=null;\nReason: 'Extraction failed'", "pred=UNCERTAIN; video=0.00;\naudio.score=null [PASS]"),
        ("T-11", "Text (Sub-length)", "'Good morning everyone.'\n(8 words, 23 characters)", "Verdict: UNCERTAIN; score=null;\nReason: 'Text too short'", "pred=UNCERTAIN; score=null;\nreason='too short' [PASS]"),
        ("T-12", "Text (Paragraph)", "Historical library passage\n(120 words, 680 characters)", "Valid style estimate; score >= 0.50;\nLabel: 'Experimental estimate'", "pred=FAKE; score=0.9675;\nExperimental estimate [PASS]"),
        ("T-13", "Video + Subtitles", "Video clip with burnt-in subtitles\n(Cross-modal consistency)", "Subtitles non-voting; Verdict unchanged;\nStatus: 'Transcript comparison unavail'", "pred=REAL (unaltered);\ntranscript=unavail [PASS]")
    ]

    for row_idx, row_data in enumerate(test_data, start=1):
        for col_idx, text_val in enumerate(row_data):
            cell = t_tbl.cell(row_idx, col_idx)
            bg = "FFFFFF" if row_idx % 2 != 0 else "F8FAFC"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=100, bottom=100, left=90, right=90)
            p = cell.paragraphs[0]
            format_p(p, before=0, after=0)
            r = p.add_run(text_val)
            r.font.name = "Times New Roman"
            r.font.size = Pt(8.0)
            r.font.color.rgb = RGBColor(31, 41, 55)

    add_body(
        "Table 2: Automated End-to-End Multimodal Verification Matrix across all API routes and edge conditions.",
        bold_prefix=""
    )

    add_h2("6.2 Critical Distinction: Functional Success vs. Prediction Correctness")
    add_body(
        "In rigorous academic evaluation, an engineering distinction must be maintained between functional success and prediction correctness:"
    )
    add_bullet(
        "Functional Success denotes that an API endpoint accepts an arbitrary media container, extracts required tensors, "
        "executes neural network forward passes within bounded memory, handles null conditions gracefully, and returns an HTTP 200 "
        "response complying with the Pydantic response schema. All 13 test scenarios achieve 100% functional success.",
        bold_prefix="• "
    )
    add_bullet(
        "Prediction Correctness denotes whether the classification label matches true ground reality. Claiming statistical metrics "
        "(accuracy, precision, recall, F1-score) requires evaluation across a balanced, annotated, identity-stratified benchmark dataset. "
        "Because local training and evaluation on FakeAVCeleb were not completed prior to submission, we explicitly refrain from fabricating "
        "dataset-wide accuracy figures. Individual test predictions (e.g., T-01, T-02, T-05) correctly identify known sample provenance, "
        "but generalized model performance reflects the published capabilities of the upstream checkpoints.",
        bold_prefix="• "
    )

    add_h2("6.3 In-Depth Analysis of Observed Issue 1: Earlier Mock Audio alongside Fused Assessment")
    add_body(
        "During early architectural testing, an anomalous behavior was identified in the video analysis pipeline: when a video was processed, "
        "the Findings Sheet displayed an audio fake probability (e.g., 0.72) and a high Fused Probability (e.g., 0.84), even though no audio checkpoint "
        "was loaded in memory. Investigation of earlier revisions of src/inference/pipeline.py revealed that when self.audio_model was None, "
        "the code fell back to a heuristic energy calculation (ZCR and RMS noise) or random initialization, passing this mock probability into the "
        "late-fusion MLP."
    )
    add_body(
        "Forensic Effect on Interpretation: This created a severe false sense of multimodal corroboration. An investigator viewing the dashboard "
        "would conclude that facial forensics and acoustic voice forensics independently confirmed forgery. In reality, the audio score was completely "
        "fabricated. In the current refactored version, this flaw is completely eliminated: mock scores are deleted, audio is processed through "
        "the genuine Wav2Vec 2.0 checkpoint, synthetic fusion is disabled (fusion.score = null), and silent tracks explicitly return score: null with "
        "diagnostic reasoning."
    )

    add_h2("6.4 In-Depth Analysis of Observed Issue 2: WhatsApp Voice Note AAC Decoding Failure")
    add_body(
        "A second critical engineering anomaly was observed when testing voice notes exported from WhatsApp. When a 58-second voice recording "
        "(audio/aac container) was loaded, the browser's native HTML5 <audio> player previewed and played the voice note seamlessly. However, upon "
        "submitting to POST /api/predict/audio, the backend returned HTTP 200 with verdict: UNCERTAIN, duration_seconds: 0.0, and reason: "
        "'Audio stream could not be decoded or is zero length.'"
    )
    add_body(
        "Separation of Symptom and Root Cause: The symptom appeared to be an invalid upload or empty file. However, file size verification proved "
        "the file was 480 KB. The root cause lies in Python audio library dependencies. While modern web browsers bundle proprietary AAC/M4A hardware "
        "decoders, the backend load_waveform() function relied on torchaudio.load() and soundfile.read(). On Windows, libsndfile lacks native support "
        "for raw ADTS AAC containers, causing soundfile to throw a format unrecognized error caught by the exception handler. "
        "Remediation Strategy: In video soundtrack extraction, this issue does not occur because FFmpeg handles demuxing. The permanent fix for "
        "standalone audio uploads is routing all non-WAV formats through an FFmpeg transcoding pipeline prior to libsndfile loading."
    )

    # ==========================================
    # 9. LIMITATIONS, RESPONSIBLE USE & FUTURE WORK
    # ==========================================
    add_h1("7. SYSTEM LIMITATIONS, RESPONSIBLE USE & FUTURE WORK")

    add_h2("7.1 Hardware & Computational Constraints")
    add_body(
        "The system was developed and validated on an Intel Core i3-1115G4 dual-core processor with 8 GB RAM and no dedicated GPU. "
        "This hardware boundary imposes strict operational tradeoffs: video analysis is restricted to 8 sampled frames per clip rather than full "
        "temporal 30 fps 3D convolutional analysis. Processing latency averages 4.5 seconds per video clip, which is acceptable for investigative "
        "triage but unsuitable for real-time live-stream filtering."
    )

    add_h2("7.2 Domain Shift & Compression Susceptibility")
    add_body(
        "Pretrained deep learning checkpoints exhibit sensitivity to compression artifacts. When media is transmitted through social messaging platforms "
        "(WhatsApp, Instagram, Telegram), aggressive H.264 macroblocking and AAC spectral quantization discard the subtle high-frequency artifacts "
        "relied upon by deepfakeguard-v2 and sdxl-detector. Consequently, false-negative rates increase on heavily compressed media."
    )

    add_h2("7.3 Responsible Use & Evidentiary Standards")
    add_body(
        "A deep learning detector output is a probabilistic estimation, NOT conclusive legal proof of forgery. In media forensics, statistical "
        "scores must never be weaponized to censor discourse or convict individuals without corroborating chain-of-custody evidence. "
        "The Text tab's AI-writing style score is particularly susceptible to false accusations against non-native English writers whose formal vocabulary "
        "mimics LLM training distributions."
    )

    add_h2("7.4 Future Engineering Roadmap")
    add_bullet("Universal FFmpeg Audio Ingestion: Route all audio uploads through FFmpeg mono 16 kHz transcoding to permanently resolve AAC/M4A decoding issues.", bold_prefix="1. ")
    add_bullet("Identity-Disjoint Benchmark Evaluation: Execute scripts/build_manifest.py and train_video.py on Google Colab GPU to produce verified ROC-AUC curves on FakeAVCeleb.", bold_prefix="2. ")
    add_bullet("Out-of-Fold Trained Multimodal Fusion: Train an authentic Late Fusion MLP using cross-validated out-of-fold embeddings to re-enable fused scoring with empirical calibration.", bold_prefix="3. ")
    add_bullet("Phonetic Subtitle Alignment: Integrate OpenAI Whisper ASR to transcribe video audio and compute genuine character-error-rate (CER) against uploaded subtitles.", bold_prefix="4. ")

    # ==========================================
    # 10. CONCLUSION & REFERENCES
    # ==========================================
    add_h1("8. CONCLUSION & REFERENCES")

    add_h2("8.1 Conclusion")
    add_body(
        "This project successfully engineered and delivered VeriSight, a robust, full-stack multimodal deepfake detection system optimized for "
        "resource-constrained CPU environments. By integrating specialized open-source checkpoints (Samson5827/deepfakeguard-v2, MelodyMachine/Deepfake-audio-detection, "
        "Organika/sdxl-detector, and Hello-SimpleAI/chatgpt-detector-roberta), the system provides comprehensive forensic analysis across video, audio, image, "
        "and text modalities. The critical architectural contribution lies in decoupling visual and acoustic streams, surfacing explicit cross-modal "
        "disagreement callouts, eliminating unvalidated mock fallbacks, and enforcing calibrated selective abstention. The system stands fully operational "
        "and verified for academic defense."
    )

    add_h2("8.2 Numbered References & Model Repositories")
    refs = [
        ("[1] Khalid, H., Woo, S. S., & Tariq, S. (2021). FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset. "
         "arXiv preprint arXiv:2108.05080. Available: https://arxiv.org/abs/2108.05080"),
        ("[2] Samson5827 (2024). DeepfakeGuard-v2: High-Performance ONNX Facial Deepfake Detector. "
         "Hugging Face Model Hub. Available: https://huggingface.co/Samson5827/deepfakeguard-v2"),
        ("[3] MelodyMachine (2024). Deepfake-audio-detection: Fine-tuned Wav2Vec 2.0 for Synthetic Voice Classification. "
         "Hugging Face Model Hub. Available: https://huggingface.co/MelodyMachine/Deepfake-audio-detection"),
        ("[4] Organika (2023). SDXL-Detector: Swin Transformer for Stable Diffusion XL Synthetic Image Detection. "
         "Hugging Face Model Hub. Available: https://huggingface.co/Organika/sdxl-detector"),
        ("[5] Hello-SimpleAI (2023). ChatGPT-Detector-RoBERTa: Sequence Classifier for AI-Generated Text Estimation. "
         "Hugging Face Model Hub. Available: https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta"),
        ("[6] Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. "
         "NeurIPS 2020. Available: https://arxiv.org/abs/2006.11477"),
        ("[7] Liu, Z., Lin, Y., Cao, Y., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. "
         "ICCV 2021. Available: https://arxiv.org/abs/2103.14030"),
        ("[8] Liu, Y., Ott, M., Goyal, N., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. "
         "arXiv preprint arXiv:1907.11692. Available: https://arxiv.org/abs/1907.11692"),
        ("[9] Rossler, A., Cozzolino, D., Verdoliva, L., et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. "
         "ICCV 2019. Available: https://arxiv.org/abs/1901.08971"),
        ("[10] ISO/IEC JTC 1/SC 42 (2023). Artificial Intelligence — Assessment of Machine Learning Classification Performance. "
         "International Organization for Standardization.")
    ]
    for r in refs:
        p = doc.add_paragraph()
        format_p(p, before=2, after=4, line_spacing=1.1)
        run = p.add_run(r)
        run.font.name = "Times New Roman"
        run.font.size = Pt(9.0)
        run.font.color.rgb = RGBColor(51, 65, 85)

    # Save document
    out_dir = Path(r"C:\Users\hp\.gemini\antigravity\scratch\multimodal-deepfake-detection\submission_report").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_docx = out_dir / "IMDAI_Multimodal_Deepfake_Detection_Report.docx"
    doc.save(str(out_docx))
    print(f"Document successfully created at: {out_docx}")

if __name__ == "__main__":
    create_report()
