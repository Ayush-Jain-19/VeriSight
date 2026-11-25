import React, { useRef, useState, useCallback, useEffect } from 'react';
import { ModalityType, ImageCheckType } from '../types/api';

const VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.webm', '.mkv'];
const AUDIO_EXTS = ['.wav', '.mp3', '.aac', '.flac', '.m4a', '.ogg'];
const IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.webp'];

const MAX_SIZE_BYTES = 100 * 1024 * 1024;

interface SourceMonitorProps {
  modality: ModalityType;
  file: File | null;
  onFileSelect: (file: File) => void;
  textContent: string;
  onTextChange: (text: string) => void;
  fileError: string | null;
  onFileError: (err: string | null) => void;
  imageCheckType?: ImageCheckType;
  onImageCheckTypeChange?: (val: ImageCheckType) => void;
}

interface MediaMeta {
  duration: number | null;
  width: number | null;
  height: number | null;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec * 100) % 100);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(2, '0')}`;
}

function getFileExt(name: string): string {
  const idx = name.lastIndexOf('.');
  return idx >= 0 ? name.substring(idx).toLowerCase() : '';
}

export const SourceMonitor: React.FC<SourceMonitorProps> = ({
  modality,
  file,
  onFileSelect,
  textContent,
  onTextChange,
  fileError,
  onFileError,
  imageCheckType = 'face_swap',
  onImageCheckTypeChange,
}) => {
  const [dragOver, setDragOver] = useState(false);
  const [mediaMeta, setMediaMeta] = useState<MediaMeta>({ duration: null, width: null, height: null });
  const [mediaUrl, setMediaUrl] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  // Manage object URL
  useEffect(() => {
    if (file && modality !== 'text') {
      const url = URL.createObjectURL(file);
      setMediaUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setMediaUrl(null);
    }
    setMediaMeta({ duration: null, width: null, height: null });
  }, [file, modality]);

  const getAllowedExtensions = () => {
    if (modality === 'audio') return AUDIO_EXTS;
    if (modality === 'image') return IMAGE_EXTS;
    return VIDEO_EXTS;
  };

  const validateAndSelect = useCallback(
    (fileList: FileList | null) => {
      onFileError(null);
      if (!fileList || fileList.length === 0) return;
      const f = fileList[0];

      const allowed = getAllowedExtensions();
      const ext = getFileExt(f.name);
      if (!allowed.includes(ext)) {
        onFileError(`Unsupported format '${ext}'. Accepted: ${allowed.join(', ')}`);
        return;
      }
      if (f.size > MAX_SIZE_BYTES) {
        onFileError('File exceeds the 100 MB limit.');
        return;
      }
      onFileSelect(f);
    },
    [modality, onFileSelect, onFileError]
  );

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (modality !== 'text') setDragOver(true);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const rect = e.currentTarget.getBoundingClientRect();
    const { clientX, clientY } = e;
    if (
      clientX <= rect.left ||
      clientX >= rect.right ||
      clientY <= rect.top ||
      clientY >= rect.bottom
    ) {
      setDragOver(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    if (modality !== 'text' && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSelect(e.dataTransfer.files);
    }
  };

  const handleVideoLoaded = () => {
    const v = videoRef.current;
    if (v) {
      setMediaMeta({
        duration: isFinite(v.duration) ? v.duration : null,
        width: v.videoWidth || null,
        height: v.videoHeight || null,
      });
    }
  };

  const handleAudioLoaded = () => {
    const a = audioRef.current;
    if (a) {
      setMediaMeta({
        duration: isFinite(a.duration) ? a.duration : null,
        width: null,
        height: null,
      });
    }
  };

  const handleImageLoaded = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setMediaMeta({
      duration: null,
      width: img.naturalWidth || null,
      height: img.naturalHeight || null,
    });
  };

  const openFilePicker = () => {
    fileInputRef.current?.click();
  };

  const getMonitorLabel = () => {
    switch (modality) {
      case 'audio':
        return 'ACOUSTIC MONITOR';
      case 'image':
        return 'IMAGE VIEWPORT';
      case 'text':
        return 'TEXT INSPECTION DESK';
      case 'video':
      default:
        return 'SOURCE MONITOR';
    }
  };

  const getDropPrompt = () => {
    switch (modality) {
      case 'audio':
        return 'Drop an audio recording here';
      case 'image':
        return 'Drop a photograph or face crop here';
      case 'video':
      default:
        return 'Drop a recording here';
    }
  };

  const getChoosePrompt = () => {
    switch (modality) {
      case 'audio':
        return 'Choose audio';
      case 'image':
        return 'Choose image';
      case 'video':
      default:
        return 'Choose video';
    }
  };

  const wordCount = textContent.trim() ? textContent.trim().split(/\s+/).length : 0;
  const charCount = textContent.length;

  return (
    <div
      className={`source-monitor${dragOver ? ' drag-over' : ''}`}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Hidden file input */}
      {modality !== 'text' && (
        <input
          ref={fileInputRef}
          type="file"
          accept={getAllowedExtensions().join(',')}
          style={{ display: 'none' }}
          onChange={(e) => {
            validateAndSelect(e.target.files);
            e.target.value = '';
          }}
          aria-label="Choose media file"
        />
      )}

      <div className={`monitor-viewport${dragOver ? ' drag-over' : ''}`}>
        {/* Corner marks */}
        <div className="monitor-corners-bottom" />

        {/* Label */}
        <span className="monitor-label">{getMonitorLabel()}</span>

        {/* Mode: TEXT */}
        {modality === 'text' && (
          <div className="monitor-text-editor">
            <textarea
              className="monitor-textarea"
              value={textContent}
              onChange={(e) => onTextChange(e.target.value)}
              placeholder="Paste or compose article text, transcription, or statement to evaluate for synthetic generation patterns using Hello-SimpleAI/chatgpt-detector-roberta..."
              aria-label="Text to inspect"
            />
          </div>
        )}

        {/* Mode: FILE (Video, Audio, Image, Video+Text) */}
        {modality !== 'text' && !file && !dragOver && (
          <div className="monitor-empty">
            <p className="monitor-empty-text">{getDropPrompt()}</p>
            <button
              className="monitor-choose-btn"
              onClick={openFilePicker}
              type="button"
            >
              {getChoosePrompt()}
            </button>
          </div>
        )}

        {modality !== 'text' && !file && dragOver && (
          <p className="monitor-drag-text">Release to load media</p>
        )}

        {/* Video Display */}
        {modality === 'video' && file && mediaUrl && (
          <video
            ref={videoRef}
            className="monitor-video"
            src={mediaUrl}
            controls
            onLoadedMetadata={handleVideoLoaded}
            preload="metadata"
          />
        )}

        {/* Audio Display */}
        {modality === 'audio' && file && mediaUrl && (
          <div className="monitor-audio-box">
            <div className="monitor-audio-waveform">
              <span className="monitor-audio-bar" style={{ animationDelay: '0.1s' }} />
              <span className="monitor-audio-bar" style={{ animationDelay: '0.3s' }} />
              <span className="monitor-audio-bar" style={{ animationDelay: '0.6s' }} />
              <span className="monitor-audio-bar" style={{ animationDelay: '0.2s' }} />
              <span className="monitor-audio-bar" style={{ animationDelay: '0.5s' }} />
              <span className="monitor-audio-bar" style={{ animationDelay: '0.4s' }} />
              <span className="monitor-audio-bar" style={{ animationDelay: '0.7s' }} />
            </div>
            <audio
              ref={audioRef}
              className="monitor-audio-player"
              src={mediaUrl}
              controls
              onLoadedMetadata={handleAudioLoaded}
            />
          </div>
        )}

        {/* Image Display */}
        {modality === 'image' && file && mediaUrl && (
          <img
            className="monitor-image"
            src={mediaUrl}
            alt="Source preview"
            onLoad={handleImageLoaded}
          />
        )}

        {/* Drag over replace prompt */}
        {file && dragOver && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              background: 'rgba(32, 36, 32, 0.85)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 10,
              borderRadius: 'var(--radius)',
            }}
          >
            <p className="monitor-drag-text">Release to replace media</p>
          </div>
        )}
      </div>

      {/* Image Selectable Check Row (Face-Swap vs General AI-Image) */}
      {modality === 'image' && onImageCheckTypeChange && (
        <div style={{ padding: '10px 14px', borderTop: '1px solid var(--rule-light)', background: 'var(--page)' }}>
          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--ink-secondary)', marginBottom: '6px', letterSpacing: '0.05em', textTransform: 'uppercase' }}>
            Inspection Checkpoint:
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
            <button
              type="button"
              onClick={() => onImageCheckTypeChange('face_swap')}
              style={{
                padding: '8px 10px',
                fontSize: '12px',
                fontWeight: 500,
                borderRadius: 'var(--radius)',
                border: '1px solid ' + (imageCheckType === 'face_swap' ? 'var(--ink)' : 'var(--rule)'),
                background: imageCheckType === 'face_swap' ? 'var(--ink)' : 'var(--page-alt)',
                color: imageCheckType === 'face_swap' ? 'var(--page)' : 'var(--ink)',
                cursor: 'pointer',
                textAlign: 'left',
              }}
            >
              <div style={{ fontWeight: 600 }}>Face-Swap Check</div>
              <div style={{ fontSize: '10px', opacity: 0.8, marginTop: '2px' }}>Samson5827/deepfakeguard-v2</div>
            </button>

            <button
              type="button"
              onClick={() => onImageCheckTypeChange('general_ai')}
              style={{
                padding: '8px 10px',
                fontSize: '12px',
                fontWeight: 500,
                borderRadius: 'var(--radius)',
                border: '1px solid ' + (imageCheckType === 'general_ai' ? 'var(--ink)' : 'var(--rule)'),
                background: imageCheckType === 'general_ai' ? 'var(--ink)' : 'var(--page-alt)',
                color: imageCheckType === 'general_ai' ? 'var(--page)' : 'var(--ink)',
                cursor: 'pointer',
                textAlign: 'left',
              }}
            >
              <div style={{ fontWeight: 600 }}>General AI-Image Check</div>
              <div style={{ fontSize: '10px', opacity: 0.8, marginTop: '2px' }}>Organika/sdxl-detector</div>
            </button>
          </div>
        </div>
      )}

      {/* Video with Text overlay input row */}
      {modality === 'video' && (
        <div className="monitor-text-overlay-input">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <label htmlFor="overlay-text">Subtitles / On-Screen Text (Optional):</label>
            {textContent.length > 0 && (
              <button
                type="button"
                onClick={() => onTextChange('')}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--monitor-muted)',
                  fontSize: '11px',
                  cursor: 'pointer',
                  padding: '0 4px',
                  textDecoration: 'underline',
                }}
              >
                Clear text
              </button>
            )}
          </div>
          <input
            id="overlay-text"
            type="text"
            value={textContent}
            onChange={(e) => onTextChange(e.target.value)}
            placeholder="Optional subtitles (subtitles do not alter face/voice manipulation verdicts)..."
          />
        </div>
      )}

      {/* Footer */}
      <div className="monitor-footer">
        {modality === 'text' ? (
          <span className="monitor-meta" style={{ width: '100%', display: 'flex', alignItems: 'center' }}>
            <span>{wordCount} words</span>
            <span>{charCount} characters</span>
            <span>Est. read: {Math.max(1, Math.ceil(wordCount / 200))} min</span>
            {textContent.length > 0 && (
              <button
                className="monitor-replace-btn"
                onClick={() => onTextChange('')}
                type="button"
                style={{ marginLeft: 'auto' }}
              >
                Clear text
              </button>
            )}
          </span>
        ) : file ? (
          <>
            <span className="monitor-filename" title={file.name}>
              {file.name}
            </span>
            <span className="monitor-meta">
              <span>{formatBytes(file.size)}</span>
              {mediaMeta.duration !== null && (
                <span>{formatDuration(mediaMeta.duration)}</span>
              )}
              {mediaMeta.width !== null && mediaMeta.height !== null && (
                <span>
                  {mediaMeta.width}×{mediaMeta.height}
                </span>
              )}
            </span>
            <button
              className="monitor-replace-btn"
              onClick={openFilePicker}
              type="button"
            >
              Replace file
            </button>
          </>
        ) : (
          <span className="monitor-meta">
            <span>{getAllowedExtensions().map((e) => e.toUpperCase().replace('.', '')).join(' · ')}</span>
            <span>Max 100 MB</span>
          </span>
        )}
      </div>

      {/* Error */}
      {fileError && <div className="monitor-error">{fileError}</div>}
    </div>
  );
};

export default SourceMonitor;
