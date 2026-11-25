import React, { useEffect, useState } from 'react';
import { Film, Music, Image as ImageIcon, FileText } from 'lucide-react';
import { SourceMonitor } from './components/SourceMonitor';
import { FindingsSheet } from './components/FindingsSheet';
import { usePredict } from './hooks/usePredict';
import { HealthResponse, ModalityType, ImageCheckType } from './types/api';

const HEADINGS: Record<ModalityType, { title: string; subtitle: string }> = {
  video: {
    title: 'Inspect video recording',
    subtitle: 'Sample 8 frames for face-swap analysis, inspect audio soundtrack, and evaluate cross-modal signals.',
  },
  audio: {
    title: 'Inspect acoustic recording',
    subtitle: 'Examine speech characteristics and voice cloning with MelodyMachine/Deepfake-audio-detection.',
  },
  image: {
    title: 'Inspect photographic media',
    subtitle: 'Analyze face-swap tampering (deepfakeguard-v2) or general AI image synthesis (sdxl-detector).',
  },
  text: {
    title: 'Inspect textual content',
    subtitle: 'Evaluate passages with Hello-SimpleAI/chatgpt-detector-roberta for synthetic writing style.',
  },
};

export const App: React.FC = () => {
  const [modality, setModality] = useState<ModalityType>('video');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [textContent, setTextContent] = useState<string>('');
  const [fileError, setFileError] = useState<string | null>(null);
  const [imageCheckType, setImageCheckType] = useState<ImageCheckType>('face_swap');

  const { result, isLoading, error, analyzeMedia, reset } = usePredict();

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthChecked, setHealthChecked] = useState(false);
  const [showSystemDetails, setShowSystemDetails] = useState(false);

  useEffect(() => {
    fetch('/api/health')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        setHealth(data);
        setHealthChecked(true);
      })
      .catch(() => {
        setHealth(null);
        setHealthChecked(true);
      });
  }, []);

  const handleModalityChange = (newModality: ModalityType) => {
    if (newModality === modality) return;
    setModality(newModality);
    setSelectedFile(null);
    setTextContent('');
    setFileError(null);
    reset();
  };

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setFileError(null);
    reset();
  };

  const handleTextChange = (text: string) => {
    setTextContent(text);
    if (result) reset();
  };

  const handleAnalyze = () => {
    if (!isLoading) {
      analyzeMedia(modality, selectedFile, textContent, imageCheckType);
    }
  };

  const activeHeading = HEADINGS[modality];

  return (
    <div>
      {/* Top bar */}
      <header className="topbar">
        <div className="topbar-brand">
          <span className="topbar-wordmark">VeriSight</span>
        </div>

        <div className="topbar-right">
          {healthChecked && !health && (
            <div className="backend-unavailable">
              <span className="backend-unavailable-dot" />
              <span>Backend offline</span>
            </div>
          )}

          <button
            className="topbar-btn"
            type="button"
            onClick={() => setShowSystemDetails(!showSystemDetails)}
            aria-expanded={showSystemDetails}
          >
            {showSystemDetails ? 'Hide details' : 'System details'}
          </button>
        </div>
      </header>

      {/* Collapsible System Details */}
      {showSystemDetails && (
        <div className="system-details" role="region" aria-label="System diagnostic details">
          <div className="system-detail-item">
            <span
              className={`system-detail-dot ${
                health ? 'system-detail-dot--ok' : 'system-detail-dot--error'
              }`}
            />
            <span className="system-detail-label">Backend:</span>
            <span className="system-detail-value">
              {health ? `Online (${health.version})` : 'Unreachable'}
            </span>
          </div>

          <div className="system-detail-item">
            <span className="system-detail-label">Hardware Device:</span>
            <span className="system-detail-value">
              {health ? health.device.toUpperCase() : 'CPU'}
            </span>
          </div>

          <div className="system-detail-item">
            <span className="system-detail-label">FFmpeg Core:</span>
            <span className="system-detail-value">
              {health ? (health.ffmpeg_available ? 'Available' : 'Missing') : '—'}
            </span>
          </div>

          <div className="system-detail-item">
            <span className="system-detail-label">Visual Model:</span>
            <span className="system-detail-value">
              Samson5827/deepfakeguard-v2 (ONNX)
            </span>
          </div>

          <div className="system-detail-item">
            <span className="system-detail-label">Acoustic Model:</span>
            <span className="system-detail-value">
              MelodyMachine/Deepfake-audio-detection
            </span>
          </div>

          <div className="system-detail-item">
            <span className="system-detail-label">Text Model:</span>
            <span className="system-detail-value">
              Hello-SimpleAI/chatgpt-detector-roberta
            </span>
          </div>
        </div>
      )}

      {/* Workspace heading & Modality Selector */}
      <div className="workspace-heading">
        <h1 className="workspace-title">{activeHeading.title}</h1>
        <p className="workspace-subtitle">{activeHeading.subtitle}</p>

        {/* Modality Tabs */}
        <nav className="modality-nav" aria-label="Inspection Modality">
          <button
            type="button"
            className={`modality-tab ${modality === 'video' ? 'modality-tab--active' : ''}`}
            onClick={() => handleModalityChange('video')}
          >
            <Film size={15} />
            <span>Video</span>
          </button>

          <button
            type="button"
            className={`modality-tab ${modality === 'audio' ? 'modality-tab--active' : ''}`}
            onClick={() => handleModalityChange('audio')}
          >
            <Music size={15} />
            <span>Audio</span>
          </button>

          <button
            type="button"
            className={`modality-tab ${modality === 'image' ? 'modality-tab--active' : ''}`}
            onClick={() => handleModalityChange('image')}
          >
            <ImageIcon size={15} />
            <span>Image</span>
          </button>

          <button
            type="button"
            className={`modality-tab ${modality === 'text' ? 'modality-tab--active' : ''}`}
            onClick={() => handleModalityChange('text')}
          >
            <FileText size={15} />
            <span>Text</span>
          </button>
        </nav>
      </div>

      {/* Main two-column workspace */}
      <main className="workspace">
        <SourceMonitor
          modality={modality}
          file={selectedFile}
          onFileSelect={handleFileSelect}
          textContent={textContent}
          onTextChange={handleTextChange}
          fileError={fileError}
          onFileError={setFileError}
          imageCheckType={imageCheckType}
          onImageCheckTypeChange={setImageCheckType}
        />

        <FindingsSheet
          modality={modality}
          file={selectedFile}
          textContent={textContent}
          result={result}
          isLoading={isLoading}
          error={error}
          onAnalyze={handleAnalyze}
          imageCheckType={imageCheckType}
        />
      </main>
    </div>
  );
};

export default App;
