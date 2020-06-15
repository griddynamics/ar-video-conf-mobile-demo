import React, { useState, useEffect }  from 'react';
import * as tf from '@tensorflow/tfjs';
import beach from '../../assets/beach.jpg';
import {container, center, slider, canvas, settings, backgroundSettings, backgroundProgress} from './ArVideoConf.module.scss';
import CircularProgress from '@material-ui/core/CircularProgress';
import {Slider, IconButton, Tooltip} from "@material-ui/core";
import PhotoLibraryIcon from '@material-ui/icons/PhotoLibrary';
import BugReportIcon from '@material-ui/icons/BugReport';
import CloseIcon from '@material-ui/icons/Close';
import DebugPanel from '../DebugPanel/DebugPanel';
import { useModel } from '../../hooks/useModel';
import { useMediaStream } from '../../hooks/useMediaStream';
import {TensorDisplay, disposeIfSet, DebugStats} from '../../util';
import {MODEL_INPUT_WIDTH,MODEL_INPUT_HEIGHT,VIDEO_WIDTH,VIDEO_HEIGHT} from '../../constants';

const tensorDisplay = new TensorDisplay(VIDEO_WIDTH, VIDEO_HEIGHT);
const debugStats = new DebugStats(100);
const SKIP_FRAMES_COUNT = 4;

const ArVideoConf = () => {
    const [loading, setLoading] = useState(true);
    const [backgroundLoading, setBackgroundLoading] = useState(false);
    const [opacity, setOpacity] = useState(0.5);
    const [debug, setDebug]= useState(false);
    const [debugInput, setDebugInput] = useState([]);
    const [backgroundImgSrc, setBackgroundImgSrc] = useState(beach);

    const videoRef = React.useRef();
    const canvasRef = React.useRef();
    const sliderValueRef = React.useRef(opacity);
    const debugRef = React.useRef(debug);

    const backgroundImageRef = React.useRef();
    const backgroundTensorRef = React.useRef();

    const mediaStream = useMediaStream();
    const model = useModel();

    useEffect(() => {
      if(mediaStream && videoRef.current && !videoRef.current.srcObject){
        videoRef.current.srcObject = mediaStream;
      }
    }, [videoRef, mediaStream]);

    useEffect(() => {
      const onload = () => {
        disposeIfSet(backgroundTensorRef.current);
        backgroundTensorRef.current = tf.tidy(() => 
          tf.browser.fromPixels(backgroundImageRef.current).div(tf.scalar(255.0))
        );
        setBackgroundLoading(false);
      } 
      backgroundImageRef.current = new Image(VIDEO_WIDTH, VIDEO_HEIGHT);
      backgroundImageRef.current.crossOrigin = "Anonymous";
      backgroundImageRef.current.addEventListener("load", onload);

      return () => {
        backgroundImageRef.current.removeEventListener("load", onload);
      }

    },[]);

    useEffect(()=> {
      backgroundImageRef.current.src = backgroundImgSrc;
    }, [backgroundImgSrc]);

    let previousSegmentationComplete = true;
    let frameCnt = SKIP_FRAMES_COUNT + 1;

    let videoBackgroundTensor;
    let scaledMask;

    const predictWebCam = async () => {
      if(previousSegmentationComplete){
        const ts = performance.now();
        previousSegmentationComplete=false;

        const input = tf.tidy(()=> tf.browser.fromPixels(videoRef.current).div(tf.scalar(255.0)));            

        if(++frameCnt > SKIP_FRAMES_COUNT){
          updateScaledMask(input);
          updateVirtualBeackgroundTensor();
          frameCnt = 0;

          if(debugRef.current){
            setDebugInput([input, scaledMask, debugStats.summary()]);
            debugRef.current = false;
          }
        }

        const img = tf.tidy(()=> {
          const foregroundTesnor = input.mul(scaledMask);
          return foregroundTesnor.add(videoBackgroundTensor).mul(tf.scalar(255));
        });

        const t0 = performance.now();
        //await tf.browser.toPixels(img, canvasRef.current);     
        await tensorDisplay.show(img, canvasRef.current);
        const t1 = performance.now();
        
        const td = t1 - t0;
        frameCnt === 0 ? debugStats.storePredictionTime(td) : debugStats.storeDisplayTime(td);

        tf.dispose([input, img]);
        previousSegmentationComplete=true;

        const te = performance.now();
        debugStats.storeOverallTime(te-ts);

        requestAnimationFrame(predictWebCam);
      }
    }
    
    const onLoadedData = () => {
      fitCanvasToContainer();
      predictWebCam();
      setLoading(false);
    }

    const fitCanvasToContainer = () => {
      canvasRef.current.style.width='100%';
      canvasRef.current.width  = canvasRef.current.offsetWidth;
      canvasRef.current.height = canvasRef.current.offsetWidth * VIDEO_HEIGHT / VIDEO_WIDTH;
    }

    const changeBackground = () => {
      setBackgroundLoading(true);
      fetch(`https://picsum.photos/${VIDEO_WIDTH}/${VIDEO_HEIGHT}`)
      .then(data => {
        setBackgroundImgSrc(data.url);
      })
      .catch(error => {
        setBackgroundLoading(false);
        console.error("Loading background image faild!", error)
      });
    } 

    const changeOpacity = (_, val) => {
        setOpacity(val);
        sliderValueRef.current = val;
    }

    const updateScaledMask = (input) => {
      disposeIfSet(scaledMask);
      scaledMask = tf.tidy(()=> {
        const x = tf.image.resizeBilinear(input, [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH]).expandDims(0);
        const y = model.predict(x);
        const mask = y.reshape([MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, 1]).mul(tf.scalar(sliderValueRef.current))
        return tf.image.resizeBilinear(mask, [VIDEO_HEIGHT, VIDEO_WIDTH]);
      });
    };

    const updateVirtualBeackgroundTensor = () => {
      disposeIfSet(videoBackgroundTensor);
      videoBackgroundTensor = tf.tidy(()=> { 
        const ones = tf.ones([VIDEO_HEIGHT, VIDEO_WIDTH, 1]);
        const backgroundMask = ones.sub(scaledMask);
        return backgroundMask.mul(backgroundTensorRef.current);
      });
    };

    const showDebug = () => {
      setDebug(true);
      debugRef.current = true;
    }

    const closeDebug = () => {
      setDebug(false);
      setDebugInput([]);
    }

    return (
        <div className={container}>
            <video ref={videoRef} style={{display: "none"}} autoPlay onLoadedData={onLoadedData}></video>
            <div className={center}>   
              {loading && <CircularProgress size={100} />}           
            </div>
            <div className={canvas}>
                <canvas ref={canvasRef} />
                <div hidden={loading}>
                  <div className={settings}>
                    <div className={backgroundSettings}>
                      <Tooltip title="Change Background">
                        <IconButton disabled={backgroundLoading} color="primary" onClick={changeBackground}>
                          <PhotoLibraryIcon/>
                        </IconButton>
                      </Tooltip>
                      {backgroundLoading && <CircularProgress size={50} className={backgroundProgress}/>}
                    </div>
                    
                    <div className={slider}>
                      <Tooltip title="Opacity">
                        <Slider 
                          orientation="vertical" 
                          min={0.0} max={1.0} step={0.01}
                          value={opacity} onChange={changeOpacity} />
                      </Tooltip>
                    </div>

                    <div> 
                      <Tooltip title="Debug">
                      <IconButton color="primary" onClick={showDebug}>
                          <BugReportIcon/>
                        </IconButton>
                      </Tooltip>
                    </div>
                </div>
              </div>
            </div>
            {debug &&
              <div>
                <Tooltip title="Close Debug">
                  <IconButton color="primary" onClick={closeDebug}>
                    <CloseIcon/>
                  </IconButton>
                </Tooltip>
                <DebugPanel input={debugInput}/>
              </div> 
            }
          </div>
    );
}

export default ArVideoConf;