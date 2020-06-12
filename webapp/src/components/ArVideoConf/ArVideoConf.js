import React from 'react';
import * as tf from '@tensorflow/tfjs';
import beach from '../../assets/beach.jpg';
import landscape from '../../assets/landscape.jpg';
import {container, center, slider, canvas, settings, debugClose} from './ArVideoConf.module.scss';
import ClipLoader from "react-spinners/ClipLoader";
import {Slider, IconButton, Tooltip} from "@material-ui/core";
import PhotoLibraryIcon from '@material-ui/icons/PhotoLibrary';
import BugReportIcon from '@material-ui/icons/BugReport';
import CloseIcon from '@material-ui/icons/Close';

const MODEL_PATH = `${window.location.protocol}//${window.location.host}/tfjs/cmd/model.json`;

const MODEL_INPUT_WIDTH = 256;
const MODEL_INPUT_HEIGHT = 256;
const VIDEO_WIDTH = 848;
const VIDEO_HEIGHT = 480;

const video = document.createElement('video');

const backgroundImg = new Image();
const backgroundCanvas = document.createElement('canvas');
const backgroundCanvasCtx = backgroundCanvas.getContext('2d');

const displayRenderCanvas = document.createElement('canvas');
displayRenderCanvas.width = VIDEO_WIDTH;
displayRenderCanvas.height = VIDEO_HEIGHT;

const displayRenderCanvasCtx = displayRenderCanvas.getContext('2d');

const debugRenderCanvas = document.createElement('canvas');
debugRenderCanvas.width = MODEL_INPUT_WIDTH;
debugRenderCanvas.height = MODEL_INPUT_HEIGHT;
const debugRenderCanvasCtx = debugRenderCanvas.getContext('2d');


class ArVideoConf extends React.Component{
    constructor(){
      super();
      this.state = {
        loading: true,
        opaque: 1.0,
        debug: false
      };
    }
    canvasRef = React.createRef();

    canvasDebugInputRef = React.createRef();
    canvasDebugMaskRef = React.createRef();
    canvasDebugOutputRef = React.createRef();
    
    backgroundTensor;
    backgroundChanged = true;
    videoBackgroundTensor;
    scaledMask;
    frameCnt = 5;
    previousSegmentationComplete=true;

    displayTime = [];
    predictionTime = [];
    debug = false;

    componentDidMount() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            this.backgroundSetup();
            const webCamPromise = navigator.mediaDevices
              .getUserMedia({
                audio: false,
                video: {
                  facingMode: "user",
                  width: { ideal: VIDEO_WIDTH},
                  height: { ideal: VIDEO_HEIGHT}
                }
              })
              .then(stream => {
                video.srcObject = stream;
                return new Promise((resolve, _) => {
                    video.onloadedmetadata = () => {
                        video.play();
                        resolve();
                  };
                });
              });

            const modelPromise = tf.loadLayersModel(MODEL_PATH, {strict:false});
            Promise.all([modelPromise, webCamPromise])
              .then(values => {
                this.fitCanvasToContainer();
                console.log(values[0]);
                this.predictWebCam(values[0]);

                this.setState({loading: false});
              })
              .catch(error => {
                console.error(error);
              });
          }
    }
    
    predictWebCam = async (model) => {
      
        if(this.backgroundChanged){
          this.backgroundTensor = tf.browser.fromPixels(backgroundCanvas).div(tf.scalar(255.0));
          this.backgroundChanged = false;
          this.predictWebCam(model);
        }

        if(this.previousSegmentationComplete){
          this.previousSegmentationComplete=false;
          const input = tf.tidy(()=> tf.browser.fromPixels(video).div(tf.scalar(255.0)));            

          if(++this.frameCnt > 4){
            if(this.scaledMask) this.scaledMask.dispose();
            this.scaledMask = tf.tidy(()=> {
              const modelInput = tf.image.resizeBilinear(input, [MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT]);
              const x = modelInput.expandDims(0);
              const y = model.predict(x);
              const mask = y.reshape([MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, 1]).mul(tf.scalar(this.state.opaque));
              
              const GB = tf.fill([MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, 2],255, 'int32');
              const R = tf.ones([MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, 1]).sub(mask).mul(tf.scalar(255));
              const maskImg = tf.concat([R,GB], 2);

              if(this.debug){
                toPixelsSync(modelInput.mul(tf.scalar(255)), this.canvasDebugInputRef.current);
                toPixelsSync(maskImg, this.canvasDebugMaskRef.current);
                toPixelsSync(modelInput.mul(tf.scalar(255)), this.canvasDebugOutputRef.current);
                toPixelsSync(maskImg, this.canvasDebugOutputRef.current, 0.5);
                
                this.debug = false;
              }

              return tf.image.resizeBilinear(mask, [VIDEO_HEIGHT, VIDEO_WIDTH]);
            });

            if(this.videoBackgroundTensor) this.videoBackgroundTensor.dispose();
            this.videoBackgroundTensor = tf.tidy(()=> { 
              const ones = tf.ones([VIDEO_HEIGHT, VIDEO_WIDTH, 1]);
              const bacgroundMask = ones.sub(this.scaledMask);
              return bacgroundMask.mul(this.backgroundTensor);
            });
            
            this.frameCnt = 0;
          }

          const img = tf.tidy(()=> {
            const foregroundTesnor = input.mul(this.scaledMask);
            return foregroundTesnor.add(this.videoBackgroundTensor).mul(tf.scalar(255));
          });

          const t0 = performance.now();
          //await tf.browser.toPixels(img, this.canvasRef.current);
          await toPixels(img, this.canvasRef.current);
          const t1 = performance.now();
          
          const td = t1 - t0;
          this.frameCnt === 0 ? this.predictionTime.push(td)
                              : this.displayTime.push(td);

          if(this.displayTime.length === 100){
            const dt = this.displayTime.reduce((a,c) => a + c)/this.displayTime.length;
            const pt = this.predictionTime.reduce((a,c) => a + c)/this.predictionTime.length - dt;
            
            console.log("display time: ", dt);
            console.log("prediction time: ", pt);
        
            this.predictionTime = [];
            this.displayTime = [];
          }

          tf.dispose([input, img]);
          this.previousSegmentationComplete=true;
          requestAnimationFrame(() => this.predictWebCam(model));
        }
      }
    
    backgroundSetup = () => {
      backgroundCanvas.width = VIDEO_WIDTH;
      backgroundCanvas.height = VIDEO_HEIGHT;
      backgroundImg.src = beach;
      backgroundImg.onload = () => {
        backgroundCanvasCtx.drawImage(backgroundImg, 0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
        this.backgroundChanged = true;
      }
    }

    fitCanvasToContainer(){
      this.canvasRef.current.style.width='100%';
      this.canvasRef.current.width  = this.canvasRef.current.offsetWidth;
      this.canvasRef.current.height = this.canvasRef.current.offsetWidth * VIDEO_HEIGHT / VIDEO_WIDTH;
    }

    changeBackground = () => {
      let img = backgroundImg.src.endsWith(landscape) ? beach : landscape;
      backgroundImg.src = img;
    }

    changeOpaque = (_, val) => {
      this.setState({opaque: val});
    }

    showDebug = () => {
      this.debug = true;
      this.setState({debug:true});
    }

    closeDebug = () => {
      this.setState({debug:false});
    }

    render() {
        return (
          <div className={container}>
            <div className={center}>               
              <ClipLoader
                size={100}
                loading={this.state.loading}
              />
            </div>
            <div className={canvas}>
                <canvas ref={this.canvasRef} />
                <div hidden={this.state.loading}>
                  <div className={settings}>
                    <div>
                      <Tooltip title="Change Background">
                        <IconButton color="primary" onClick={this.changeBackground}>
                          <PhotoLibraryIcon/>
                        </IconButton>
                      </Tooltip>
                    </div>
                    
                    <div className={slider}>
                      <Tooltip title="Opacity">
                        <Slider 
                          orientation="vertical" 
                          min={0.0} max={1.0} step={0.01} 
                          value={this.state.opaque} onChange={this.changeOpaque}  />
                      </Tooltip>
                    </div>

                    <div> 
                      <Tooltip title="Debug">
                      <IconButton color="primary" onClick={this.showDebug}>
                          <BugReportIcon/>
                        </IconButton>
                      </Tooltip>
                    </div>
                </div>
              </div>
            </div>
            
            <div hidden={!this.debug} >
              <div className={debugClose}>
                <Tooltip title="Close Debug">
                  <IconButton color="primary" onClick={this.closeDebug}>
                    <CloseIcon/>
                  </IconButton>
                </Tooltip>
              </div>
              <canvas ref={this.canvasDebugInputRef} width={MODEL_INPUT_WIDTH} height={MODEL_INPUT_HEIGHT} /> 
              <canvas ref={this.canvasDebugMaskRef} width={MODEL_INPUT_WIDTH} height={MODEL_INPUT_HEIGHT} /> 
              <canvas ref={this.canvasDebugOutputRef} width={MODEL_INPUT_WIDTH} height={MODEL_INPUT_HEIGHT} />
            </div>
          </div>
        );
      }
}

async function toPixels(tensor, canvas){     
  const [height, width] = tensor.shape.slice(0, 2);
  const alpha = tf.fill([height, width, 1],255, 'int32');

  const rgba = tf.concat([tensor, alpha], 2);
  
  const bytes = await rgba.data();
  tf.dispose([alpha, rgba]);
  
  const pixelData = new Uint8ClampedArray(bytes);
  const imageData = new ImageData(pixelData, width, height);
  
  displayRenderCanvasCtx.putImageData(imageData, 0, 0);

  const ctx = canvas.getContext('2d');  
  ctx.drawImage(displayRenderCanvas, 0, 0,  canvas.width, canvas.height);
}

function toPixelsSync(tensor, canvas, opacity = 1){     
  const [height, width] = tensor.shape.slice(0, 2);
  const alpha = tf.fill([height, width, 1], opacity * 255, 'int32');
  const rgba = tf.concat([tensor, alpha], 2);
  
  const bytes = rgba.dataSync();
  tf.dispose([alpha, rgba]);

  const pixelData = new Uint8ClampedArray(bytes);
  const imageData = new ImageData(pixelData, width, height);
  
  debugRenderCanvasCtx.putImageData(imageData, 0, 0);

  const ctx = canvas.getContext('2d');  
  ctx.drawImage(debugRenderCanvas, 0, 0,  canvas.width, canvas.height);
}
export default ArVideoConf;