import { useState } from 'react';
import { ModalityType, PredictResponse } from '../types/api';

export const usePredict = () => {
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const reset = () => {
    setResult(null);
    setError(null);
  };

  const analyzeMedia = async (
    modality: ModalityType,
    file?: File | null,
    textContent?: string,
    imageCheckType?: string
  ) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      let endpoint = '/api/predict';
      let fetchOptions: RequestInit;

      if (modality === 'text') {
        endpoint = '/api/predict/text';
        fetchOptions = {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: textContent || '' }),
        };
      } else {
        if (!file) {
          throw new Error('Please select a file to analyze.');
        }

        const formData = new FormData();
        formData.append('file', file);

        if (modality === 'audio') {
          endpoint = '/api/predict/audio';
        } else if (modality === 'image') {
          endpoint = '/api/predict/image';
          formData.append('check_type', imageCheckType || 'face_swap');
        } else {
          endpoint = '/api/predict';
          if (textContent && textContent.trim()) {
            formData.append('text', textContent.trim());
          }
        }

        fetchOptions = {
          method: 'POST',
          body: formData,
        };
      }

      const response = await fetch(endpoint, fetchOptions);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error (${response.status})`);
      }

      const data: PredictResponse = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred during analysis.');
    } finally {
      setIsLoading(false);
    }
  };

  return { result, isLoading, error, analyzeMedia, reset };
};
