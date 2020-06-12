import { useState, useEffect } from 'react';
import { loadLayersModel} from '@tensorflow/tfjs';

const MODEL_PATH = `${window.location.protocol}//${window.location.host}/tfjs/cmd/model.json`;

export function useModel() {
  const [model, setModel] = useState(null);

  useEffect(() => {
    loadLayersModel(MODEL_PATH, {strict:false})
    .then((model) => {
      setModel(model);
    });
  }, []);

  return model;
}