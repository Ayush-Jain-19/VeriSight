export interface ModelReadiness {
  loaded: boolean;
  checkpoint_loaded: boolean;
  checkpoint_path?: string | null;
  is_mock: boolean;
  model_name: string;
  error?: string | null;
}

export interface VideoEvidence {
  score: number | null;
  frames_analyzed: number;
  faces_detected: number;
  available: boolean;
  model: string;
  checkpoint_loaded?: boolean;
  is_mock?: boolean;
  reason?: string | null;
}

export interface AudioEvidence {
  score: number | null;
  duration_seconds: number;
  available: boolean;
  model: string;
  checkpoint_loaded?: boolean;
  is_mock?: boolean;
  reason?: string | null;
}

export interface FusionEvidence {
  score: number | null;
  method: string;
  threshold_used: number | null;
  threshold_name: string | null;
  checkpoint_loaded?: boolean;
  is_calibrated?: boolean;
}

export interface ImageEvidence {
  score: number | null;
  faces_detected: number;
  available: boolean;
  model: string;
  resolution?: string | null;
  reason?: string | null;
  check_type?: string;
  sdxl_score?: number | null;
}

export interface TextEvidence {
  score: number | null;
  word_count: number;
  burstiness_score?: number | null;
  perplexity_score?: number | null;
  lexical_diversity?: number | null;
  available: boolean;
  model: string;
  label?: string;
  reason?: string | null;
}

export interface CrossModalTextEvidence {
  cross_modal_consistency?: number | null;
  extracted_text_detected: boolean;
  ocr_word_count: number;
  transcript_status?: string | null;
  reason?: string | null;
}

export interface TimingEvidence {
  frame_sampling_sec?: number;
  face_detection_sec?: number;
  audio_extraction_sec?: number;
  video_inference_sec?: number;
  audio_inference_sec?: number;
  fusion_inference_sec?: number;
  total_pipeline_sec: number;
}

export interface PredictResponse {
  prediction: 'REAL' | 'FAKE' | 'UNCERTAIN';
  confidence: number;
  prediction_path: string;
  is_mock?: boolean;
  models_ready?: boolean;
  video: VideoEvidence;
  audio: AudioEvidence;
  fusion: FusionEvidence;
  timing: TimingEvidence;
  warnings: string[];
  image?: ImageEvidence;
  text?: TextEvidence;
  cross_modal?: CrossModalTextEvidence;
  disagreement?: string | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  device: string;
  ffmpeg_available: boolean;
  models_loaded: boolean;
  models_ready?: boolean;
  thresholds_calibrated?: boolean;
  video_model?: ModelReadiness | null;
  audio_model?: ModelReadiness | null;
  fusion_model?: ModelReadiness | null;
}

export type ModalityType = 'video' | 'audio' | 'image' | 'text';
export type ImageCheckType = 'face_swap' | 'general_ai';
