import React, { useState } from 'react';
import { ModalityType, PredictResponse, ImageCheckType } from '../types/api';

interface FindingsSheetProps {
  modality: ModalityType;
  file: File | null;
  textContent: string;
  result: PredictResponse | null;
  isLoading: boolean;
  error: string | null;
  onAnalyze: () => void;
  imageCheckType?: ImageCheckType;
}

const PATH_LABELS: Record<string, string> = {
  fusion: 'Independent Multimodal Evaluation (Visual + Audio)',
  video_only: 'Visual Face-Swap Stream Direct',
  audio_only: 'Acoustic Speech Stream Direct',
  image_only: 'Photographic & Facial Analysis',
  text_only: 'Linguistic AI-Writing Analysis',
  video_text_fusion: 'Audio, visual and caption assessment',
  no_modalities: 'No Usable Signals in Input',
};

export const FindingsSheet: React.FC<FindingsSheetProps> = ({
  modality,
  file,
  textContent,
  result,
  isLoading,
  error,
  onAnalyze,
  imageCheckType = 'face_swap',
}) => {
  const [detailsOpen, setDetailsOpen] = useState(false);

  // Helper for assessment title
  const getAssessmentTitle = (res: PredictResponse) => {
    if (res.prediction === 'UNCERTAIN') {
      if (modality === 'image' && res.image?.score === null && res.image?.reason) {
        return res.image.reason;
      }
      if (modality === 'audio' && res.audio?.score === null && res.audio?.reason) {
        return res.audio.reason;
      }
      if (modality === 'text' && res.text?.score === null && res.text?.reason) {
        return 'Linguistic estimation omitted';
      }
      if (modality === 'video' && res.video?.score === null && res.audio?.score === null) {
        return 'No usable video or audio signals';
      }
      return 'Inconclusive';
    }

    switch (res.prediction) {
      case 'REAL':
        return modality === 'text' ? 'Lower AI-writing style estimate' : 'No manipulation detected';
      case 'FAKE':
        return modality === 'text' ? 'Elevated AI-writing style estimate' : 'Evidence flagged';
      default:
        return 'Inconclusive';
    }
  };

  const getAssessmentClass = (res: PredictResponse) => {
    switch (res.prediction) {
      case 'REAL':
        return 'assessment-heading--real';
      case 'FAKE':
        return 'assessment-heading--fake';
      case 'UNCERTAIN':
      default:
        return 'assessment-heading--uncertain';
    }
  };

  const getAssessmentMeaning = (res: PredictResponse) => {
    if (modality === 'text') {
      return res.text?.reason || 'Experimental AI-writing style estimate. Authorship requires independent evidence.';
    }
    if (res.disagreement) return 'The overall decision includes the available stream findings below.';
    const branch = modality === 'image' ? res.image : res.audio;
    return branch?.reason || (res.prediction === 'FAKE'
      ? 'The selected detector flagged this sample.' : 'The selected detector found a low manipulation score.');
  };

  const getSignalBarFillClass = (score: number, band?: [number, number] | null) => {
    if (band && score >= band[0] && score <= band[1]) return 'signal-bar-fill--uncertain';
    return score >= 0.50 ? 'signal-bar-fill--fake' : 'signal-bar-fill--real';
  };

  const hasInput = modality === 'text' ? textContent.trim().length > 0 : file !== null;

  // State 1: Awaiting Input
  if (!hasInput) {
    return (
      <div className="findings">
        <div className="findings-section-label">ANALYSIS</div>
        <h2 className="findings-empty-heading">
          {modality === 'text' ? 'Awaiting textual input' : 'Awaiting media intake'}
        </h2>
        <p className="findings-empty-desc">
          {modality === 'text'
            ? 'Type or paste content into the inspection desk to begin analysis.'
            : 'Select or drop a file onto the monitor to begin inspection.'}
        </p>

        <hr className="findings-rule" />

        <div className="findings-section-label">EXPECTED FINDINGS</div>
        <div className="findings-index">
          <div className="findings-index-row">
            <span className="findings-index-label">Overall assessment</span>
            <span className="findings-index-value">Awaiting analysis</span>
          </div>
          {modality === 'video' && (
            <>
              <div className="findings-index-row">
                <span className="findings-index-label">Visual face-swap signal (8 frames)</span>
                <span className="findings-index-value">—</span>
              </div>
              <div className="findings-index-row">
                <span className="findings-index-label">Acoustic speech signal (soundtrack)</span>
                <span className="findings-index-value">—</span>
              </div>
            </>
          )}
          {modality === 'audio' && (
            <div className="findings-index-row">
              <span className="findings-index-label">Acoustic speech signal (16 kHz mono)</span>
              <span className="findings-index-value">—</span>
            </div>
          )}
          {modality === 'image' && (
            <div className="findings-index-row">
              <span className="findings-index-label">
                {imageCheckType === 'general_ai' ? 'General AI-image check' : 'Face-swap check (18% margin)'}
              </span>
              <span className="findings-index-value">—</span>
            </div>
          )}
          {modality === 'text' && (
            <div className="findings-index-row">
              <span className="findings-index-label">Experimental AI-writing style estimate</span>
              <span className="findings-index-value">—</span>
            </div>
          )}
        </div>
      </div>
    );
  }

  // State 2: Error
  if (error) {
    return (
      <div className="findings">
        <div className="findings-section-label">ANALYSIS STATUS</div>
        <div className="findings-error">
          <div className="findings-error-label">Analysis could not be completed</div>
          <div>{error}</div>
          <button className="retry-btn" type="button" onClick={onAnalyze}>
            Retry analysis
          </button>
        </div>

        <hr className="findings-rule" />

        <div className="findings-section-label">SAMPLE STATUS</div>
        <div className="findings-index">
          <div className="findings-index-row">
            <span className="findings-index-label">Target modality</span>
            <span className="findings-index-value">{modality.replace('_', ' ').toUpperCase()}</span>
          </div>
          <div className="findings-index-row">
            <span className="findings-index-label">Pipeline state</span>
            <span className="findings-index-value">Failed</span>
          </div>
        </div>
      </div>
    );
  }

  // State 3: Input ready / Analyzing
  if (!result) {
    return (
      <div className="findings">
        <button
          className={`run-analysis-btn${isLoading ? ' run-analysis-btn--loading' : ''}`}
          type="button"
          onClick={onAnalyze}
          disabled={isLoading}
        >
          {isLoading && <span className="run-analysis-spinner" aria-hidden="true" />}
          <span>{isLoading ? 'Running CPU inference…' : `Run ${modality.replace('_', ' ')} analysis`}</span>
        </button>

        <hr className="findings-rule" />

        <div className="findings-section-label">INSPECTION CHECKLIST</div>
        <div className="findings-index">
          <div className="findings-index-row">
            <span className="findings-index-label">Overall assessment</span>
            <span className="findings-index-value">
              {isLoading ? 'Running models…' : 'Ready to inspect'}
            </span>
          </div>
          {modality === 'video' && (
            <>
              <div className="findings-index-row">
                <span className="findings-index-label">Visual frames (deepfakeguard-v2)</span>
                <span className="findings-index-value">
                  {isLoading ? 'Sampling 8 frames & face cropping…' : 'Pending'}
                </span>
              </div>
              <div className="findings-index-row">
                <span className="findings-index-label">Audio soundtrack</span>
                <span className="findings-index-value">
                  {isLoading ? 'Extracting speech to 16 kHz…' : 'Pending'}
                </span>
              </div>
            </>
          )}
          {modality === 'audio' && (
            <div className="findings-index-row">
              <span className="findings-index-label">Acoustic classifier</span>
              <span className="findings-index-value">
                {isLoading ? 'Windowing & evaluating speech…' : 'Pending'}
              </span>
            </div>
          )}
          {modality === 'image' && (
            <div className="findings-index-row">
              <span className="findings-index-label">
                {imageCheckType === 'general_ai' ? 'General AI-image (sdxl-detector)' : 'Face-swap (deepfakeguard-v2)'}
              </span>
              <span className="findings-index-value">
                {isLoading ? 'Evaluating image pixels…' : 'Pending'}
              </span>
            </div>
          )}
          {modality === 'text' && (
            <div className="findings-index-row">
              <span className="findings-index-label">AI writing style (chatgpt-detector-roberta)</span>
              <span className="findings-index-value">
                {isLoading ? 'Tokenizing and evaluating style…' : 'Pending'}
              </span>
            </div>
          )}
        </div>
      </div>
    );
  }

  // State 4: Completed Result
  const { video, audio, timing, warnings, image, text, cross_modal, disagreement } = result;
  const filteredWarnings = (warnings || []).filter((w) => !w.startsWith('[UNVALIDATED]'));

  return (
    <div className="findings">
      {/* 1. Overall Assessment */}
      <div className="findings-section-label">ASSESSMENT</div>

      <h2 className={`assessment-heading ${getAssessmentClass(result)}`}>
        {getAssessmentTitle(result)}
      </h2>

      <p className="assessment-meaning">{getAssessmentMeaning(result)}</p>

      {/* Disagreement Callout (if branches disagree) */}
      {disagreement && (
        <div style={{
          marginTop: '12px',
          padding: '10px 12px',
          background: 'var(--page-alt)',
          borderLeft: '3px solid var(--action)',
          borderRadius: 'var(--radius-sm)',
        }}>
          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--ink-secondary)', letterSpacing: '0.05em', textTransform: 'uppercase' }}>
            Cross-Modal Finding
          </div>
          <div style={{ fontSize: '13px', color: 'var(--ink)', marginTop: '3px' }}>
            {disagreement}
          </div>
        </div>
      )}

      <hr className="findings-rule" />

      {/* 2. Modality Signals */}
      <div className="findings-section-label">MODALITY SIGNALS</div>

      <div className="findings-index">
        {/* VIDEO MODALITY: Visual Stream Signal */}
        {modality === 'video' && video && (
          <div className="signal-row">
            <div className="signal-row-header">
              <span className="signal-row-label">
                Visual face-swap signal ({video.model})
              </span>
              <span className="signal-row-value">
                {video.score !== null ? (
                  `${(video.score * 100).toFixed(1)}% detector score`
                ) : (
                  <span style={{ color: 'var(--ink-muted)' }}>{video.reason || 'No usable face'}</span>
                )}
              </span>
            </div>
            {video.score !== null ? (
              <>
                <div className="signal-bar-track">
                  <div
                    className={`signal-bar-fill ${getSignalBarFillClass(video.score, video.band)}`}
                    style={{ width: `${Math.min(100, Math.max(0, video.score * 100))}%` }}
                  />
                </div>
                <div style={{ fontSize: '11px', color: 'var(--ink-muted)', marginTop: '2px' }}>
                  Median score across {video.faces_detected}/{video.frames_analyzed} frames with detected faces (18% crop margin)
                </div>
              </>
            ) : (
              <div style={{ fontSize: '11px', color: 'var(--ink-muted)', marginTop: '2px' }}>
                {video.reason || 'No human faces detected across sampled 8 frames.'}
              </div>
            )}
          </div>
        )}

        {/* VIDEO MODALITY: Audio Stream Signal */}
        {modality === 'video' && audio && (
          <div className="signal-row">
            <div className="signal-row-header">
              <span className="signal-row-label">
                Soundtrack acoustic speech signal ({audio.model})
              </span>
              <span className="signal-row-value">
                {audio.score !== null ? (
                  `${(audio.score * 100).toFixed(1)}% detector score`
                ) : (
                  <span style={{ color: 'var(--ink-muted)' }}>{audio.reason || 'Silence or no speech'}</span>
                )}
              </span>
            </div>
            {audio.score !== null ? (
              <>
                <div className="signal-bar-track">
                  <div
                    className={`signal-bar-fill ${getSignalBarFillClass(audio.score, audio.band)}`}
                    style={{ width: `${Math.min(100, Math.max(0, audio.score * 100))}%` }}
                  />
                </div>
                <div style={{ fontSize: '11px', color: 'var(--ink-muted)', marginTop: '2px' }}>
                  Acoustic speech evaluated over {audio.duration_seconds.toFixed(1)}s (16 kHz mono)
                </div>
              </>
            ) : (
              <div style={{ fontSize: '11px', color: 'var(--ink-muted)', marginTop: '2px' }}>
                {audio.reason || 'Audio stream missing, silent, or shorter than 0.5s.'}
              </div>
            )}
          </div>
        )}

        {/* AUDIO MODALITY: Acoustic Stream Signal */}
        {modality === 'audio' && audio && (
          <div className="signal-row">
            <div className="signal-row-header">
              <span className="signal-row-label">
                Acoustic speech classifier ({audio.model})
              </span>
              <span className="signal-row-value">
                {audio.score !== null ? (
                  `${(audio.score * 100).toFixed(1)}% detector score`
                ) : (
                  <span style={{ color: 'var(--ink-muted)' }}>{audio.reason || 'Silence or insufficient speech'}</span>
                )}
              </span>
            </div>
            {audio.score !== null ? (
              <div className="signal-bar-track">
                <div
                  className={`signal-bar-fill ${getSignalBarFillClass(audio.score, audio.band)}`}
                  style={{ width: `${Math.min(100, Math.max(0, audio.score * 100))}%` }}
                />
              </div>
            ) : (
              <div style={{ fontSize: '11px', color: 'var(--ink-muted)', marginTop: '2px' }}>
                {audio.reason || 'Audio contains silence, no speech, or is shorter than 0.5s.'}
              </div>
            )}
          </div>
        )}

        {/* IMAGE MODALITY: Image Signal */}
        {modality === 'image' && image && (
          <div className="signal-row">
            <div className="signal-row-header">
              <span className="signal-row-label">
                {image.check_type === 'general_ai' ? 'General AI-image check' : 'Face-swap check'} ({image.model})
              </span>
              <span className="signal-row-value">
                {image.score !== null ? (
                  `${(image.score * 100).toFixed(1)}% detector score`
                ) : (
                  <span style={{ color: 'var(--ink-muted)' }}>{image.reason || 'No usable face'}</span>
                )}
              </span>
            </div>
            {image.score !== null ? (
              <div className="signal-bar-track">
                <div
                  className={`signal-bar-fill ${getSignalBarFillClass(image.score)}`}
                  style={{ width: `${Math.min(100, Math.max(0, image.score * 100))}%` }}
                />
              </div>
            ) : (
              <div style={{ fontSize: '11px', color: 'var(--ink-muted)', marginTop: '2px' }}>
                {image.reason || 'No usable face detected in the uploaded image.'}
              </div>
            )}
          </div>
        )}

        {/* TEXT MODALITY: Text Signal */}
        {modality === 'text' && text && (
          <div className="signal-row">
            <div className="signal-row-header">
              <span className="signal-row-label">
                {text.label || 'Experimental AI-writing style estimate'} ({text.model})
              </span>
              <span className="signal-row-value">
                {text.score !== null ? (
                  `${(text.score * 100).toFixed(1)}% ChatGPT / AI style prob.`
                ) : (
                  <span style={{ color: 'var(--ink-muted)' }}>{text.reason || 'Omitted'}</span>
                )}
              </span>
            </div>
            {text.score !== null ? (
              <div className="signal-bar-track">
                <div
                  className={`signal-bar-fill ${getSignalBarFillClass(text.score, text.band)}`}
                  style={{ width: `${Math.min(100, Math.max(0, text.score * 100))}%` }}
                />
              </div>
            ) : (
              <div style={{ fontSize: '11px', color: 'var(--ink-muted)', marginTop: '2px' }}>
                {text.reason || 'Text passage is too short (< 50 chars or < 10 words) or language is unsupported.'}
              </div>
            )}
          </div>
        )}

        {[['Visual', video?.band], ['Acoustic', audio?.band], ['Writing style', text?.band],
          ['Fusion', result.fusion?.band]].map(([label, band]) => Array.isArray(band) && (
            <p className="evidence-note" key={String(label)}>
              {label} uncertainty band {band[0].toFixed(2)}–{band[1].toFixed(2)} (uncalibrated)
            </p>
          ))}
        {cross_modal && (
          <section className="signal-row">
            <h3>Speech transcript</h3>
            <p>{cross_modal.transcript || cross_modal.transcript_status}</p>
            {cross_modal.reason && <p>{cross_modal.reason}</p>}
            {cross_modal.wer != null && (
              <p>Caption word error rate {cross_modal.wer.toFixed(3)} · Character error rate {cross_modal.cer?.toFixed(3)}
                {' · '}{cross_modal.similarity_method} {cross_modal.cross_modal_consistency?.toFixed(3)}</p>
            )}
          </section>
        )}
        {text && modality !== 'text' && <p>Transcript writing-style estimate {text.score?.toFixed(4) ?? text.reason}</p>}
        {!!text?.sentences?.length && <details className="signal-row" open>
          <summary>Sentence estimates ({text.sentences.length} sampled)</summary>
          <p>Each sentence is scored independently. The passage estimate uses overlapping token windows.</p>
          {text.sentences.map(sentence => <p key={sentence.start}>
            <mark>{sentence.text}</mark> <strong>{sentence.score.toFixed(4)}</strong>
          </p>)}
        </details>}
        {!!text?.spans?.length && <details className="signal-row">
          <summary>Writing-style evidence by passage span ({text.spans.length})</summary>
          {text.spans.map(span => <blockquote key={span.start}>
            <strong>Characters {span.start + 1}–{span.end} · {span.score.toFixed(4)}</strong>
            <p><mark>{span.text}</mark></p>
          </blockquote>)}
        </details>}
        {result.fusion?.score != null && <p>Uncalibrated fusion score {result.fusion.score.toFixed(4)}</p>}
      </div>

      <hr className="findings-rule" />

      {/* 3. Technical Details Disclosure */}
      <div className="tech-details">
        <button
          className="tech-details-toggle"
          type="button"
          onClick={() => setDetailsOpen(!detailsOpen)}
          aria-expanded={detailsOpen}
        >
          <span
            className={`tech-details-toggle-chevron${
              detailsOpen ? ' tech-details-toggle-chevron--open' : ''
            }`}
          />
          <span>Technical details & forensics breakdown</span>
        </button>

        {detailsOpen && (
          <div className="tech-details-content">
            <div className="tech-detail-row">
              <span className="tech-detail-label">Decision path</span>
              <span className="tech-detail-value">
                {PATH_LABELS[result.prediction_path] || result.prediction_path}
              </span>
            </div>

            <div className="tech-detail-row">
              <span className="tech-detail-label">Confidence</span>
              <span className="tech-detail-value">
                {result.confidence == null ? 'Unavailable' : (result.confidence * 100).toFixed(1) + '%'}
              </span>
            </div>

            {text && (
              <div className="tech-detail-row">
                <span className="tech-detail-label">Word count</span>
                <span className="tech-detail-value">{text.word_count} words</span>
              </div>
            )}

            {image && (
              <>
                <div className="tech-detail-row">
                  <span className="tech-detail-label">Detection check</span>
                  <span className="tech-detail-value">{image.model}</span>
                </div>
                {image.resolution && (
                  <div className="tech-detail-row">
                    <span className="tech-detail-label">Resolution</span>
                    <span className="tech-detail-value">{image.resolution}</span>
                  </div>
                )}
              </>
            )}

            {video && (
              <>
                <div className="tech-detail-row">
                  <span className="tech-detail-label">Frames analyzed</span>
                  <span className="tech-detail-value">{video.frames_analyzed} frames (evenly spaced)</span>
                </div>
                <div className="tech-detail-row">
                  <span className="tech-detail-label">Faces tracked</span>
                  <span className="tech-detail-value">{video.faces_detected} usable crops</span>
                </div>
              </>
            )}

            {audio && (
              <div className="tech-detail-row">
                <span className="tech-detail-label">Audio duration</span>
                <span className="tech-detail-value">{audio.duration_seconds.toFixed(2)}s</span>
              </div>
            )}

            <div className="tech-detail-row">
              <span className="tech-detail-label">Pipeline wall-clock</span>
              <span className="tech-detail-value">
                {timing.total_pipeline_sec.toFixed(2)}s
              </span>
            </div>
          </div>
        )}
      </div>

      {/* 4. Substantive Diagnostics */}
      {filteredWarnings.length > 0 && (
        <>
          <hr className="findings-rule" />
          <div className="findings-section-label">DIAGNOSTIC NOTES</div>
          <ul className="warnings-list">
            {filteredWarnings.map((warn, i) => (
              <li key={i}>{warn}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
};

export default FindingsSheet;
