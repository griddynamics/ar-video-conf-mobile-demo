import { useState, useEffect } from 'react';
import {VIDEO_HEIGHT, VIDEO_WIDTH} from '../constants';

export function useMediaStream() {
  const [mediaStream, setMediaStream] = useState(null);

  useEffect(() => {
    let currentStream = null;
    navigator.mediaDevices
    .getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: VIDEO_WIDTH},
          height: { ideal: VIDEO_HEIGHT}
        }
      })
    .then((stream) => {
      currentStream = stream;
      setMediaStream(stream);
    })
    .catch((error) => {
      console.error('Oops. Something is broken.', error);
    });
    return () => {
      if (currentStream) currentStream.getTracks().forEach(function(track) {
        track.stop();
      });
    }
  }, []);

  return mediaStream;
}