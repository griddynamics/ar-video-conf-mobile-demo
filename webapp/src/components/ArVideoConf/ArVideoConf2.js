import React from 'react';
import * as tf from '@tensorflow/tfjs';
import beach from '../../assets/beach.jpg';
import landscape from '../../assets/landscape.jpg';
import {container, spinner, slider} from './ArVideoConf.module.scss';
import ClipLoader from "react-spinners/ClipLoader";
import {Slider, Button, Grid, Typography} from "@material-ui/core";

import SegregationWorker from './segregation.worker';


const MODEL_INPUT_WIDTH = 256;
const MODEL_INPUT_HEIGHT = 256;

//360p
// const VIDEO_WIDTH = 480;
// const VIDEO_HEIGHT = 360;


//480p
 const VIDEO_WIDTH = 858;
 const VIDEO_HEIGHT = 480;

 //HD
//  const VIDEO_WIDTH = 1280;
//  const VIDEO_HEIGHT = 720;
 


const video = document.createElement('video');
const videoRenderCanvas = document.createElement('canvas');
const videoRenderCanvasCtx = videoRenderCanvas.getContext('2d');

const displayRenderCanvas = document.createElement('canvas');
displayRenderCanvas.width = VIDEO_WIDTH;
displayRenderCanvas.height = VIDEO_HEIGHT;
const displayRenderCanvasCtx = displayRenderCanvas.getContext('2d');

var maskChanged = false;
var maskData;
const worker = new SegregationWorker();
worker.addEventListener('message', (e) => {
  maskData = e.data;
  maskChanged = true;
});

const getInitialMaskAsync = () => {
    return new Promise((resolve, _ ) => {
        let interval = setInterval(()=>{
            if(maskChanged){
                clearInterval(interval);
                resolve();
            }
        }, 20);
    });
}

const backgroundImg = new Image();
const backgroundCanvas = document.createElement('canvas');
const backgroundCanvasCtx = backgroundCanvas.getContext('2d');

class ArVideoConf extends React.Component{
    constructor(){
      super();
      this.state = {
        loading: true,
        opaque: 1.0
      };
    }
    canvasRef = React.createRef();
    backgroundTensor;
    backgroundChanged = true;
    videoBackgroundTensor;
    scaledMask;
    frameCnt = 5;

    async componentDidMount() {
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
            
            await webCamPromise;

            tf.engine().startScope();
            const input = tf.browser.fromPixels(video).div(tf.scalar(255.0));            
            const x = tf.image.resizeBilinear(input, [MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT]);

            if(++this.frameCnt > 4){
                x.data().then(d => worker.postMessage(d));
                this.frameCnt = 0;
            }
            tf.engine().endScope();

            await getInitialMaskAsync();
            this.fitCanvasToContainer();
            this.predictWebCam();
            this.setState({loading: false});
          }
    }
    
    predictWebCam = async () => {
        if(this.backgroundChanged){
          this.backgroundTensor = tf.browser.fromPixels(backgroundCanvas).div(tf.scalar(255.0));
          this.backgroundChanged = false;
        }

        if(maskChanged){      
        //   console.time("update"); 
          if(this.scaledMask) this.scaledMask.dispose();
          this.scaledMask = tf.tidy(()=> {
              const mask = tf.tensor(maskData, [MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, 1]).mul(tf.scalar(this.state.opaque));
              return tf.image.resizeBilinear( mask, [VIDEO_HEIGHT, VIDEO_WIDTH]);
          });
          if(this.videoBackgroundTensor) this.videoBackgroundTensor.dispose();
          this.videoBackgroundTensor = tf.tidy(()=> { 
             const ones = tf.ones([VIDEO_HEIGHT, VIDEO_WIDTH, 1]);
             const bacgroundMask = ones.sub(this.scaledMask);
             return bacgroundMask.mul(this.backgroundTensor);
          });
          maskChanged = false;
        //   console.timeEnd("update");
        }

        //videoRenderCanvasCtx.drawImage(video, 0, 0, this.canvasRef.current.offsetWidth, this.canvasRef.current.offsetHeight);
        tf.engine().startScope();
        // console.time("preparation");
        const input = tf.browser.fromPixels(video).div(tf.scalar(255.0));            
        const x = tf.image.resizeBilinear(input, [MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT]);

        if(++this.frameCnt > 4){
            const data = x.dataSync();
            worker.postMessage(data,[data.buffer]);
            this.frameCnt = 0;
        }
        
        const foregroundTesnor = input.mul(this.scaledMask);
        const img = foregroundTesnor.add(this.videoBackgroundTensor).mul(tf.scalar(255));
        // console.timeEnd("preparation");

        //console.time("pixels");
        await toPixels(img, this.canvasRef.current);
        //await tf.browser.toPixels(input, this.canvasRef.current);
        //console.timeEnd("pixels");

        tf.engine().endScope();
        //const imageData = videoRenderCanvasCtx.getImageData(0, 0, this.canvasRef.current.offsetWidth, this.canvasRef.current.offsetHeight);
        // this.canvasRef.current.getContext('2d').putImageData(imageData, 0, 0, 0, 0 ,this.canvasRef.current.offsetWidth, this.canvasRef.current.offsetHeight );
        
        console.log(new Date().getSeconds());
        requestAnimationFrame(() => this.predictWebCam());
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
      this.canvasRef.current.style.height='100%';
      this.canvasRef.current.width  = this.canvasRef.current.offsetWidth;
      this.canvasRef.current.height = this.canvasRef.current.offsetHeight;
      // videoRenderCanvas.width  = this.canvasRef.current.offsetWidth;
      // videoRenderCanvas.height = this.canvasRef.current.offsetHeight;
    }

    changeBackground = () => {
      let img = backgroundImg.src.endsWith(landscape) ? beach : landscape;
      backgroundImg.src = img;
    }

    changeOpaque = (_, val) => {
      this.setState({opaque: val});
    }

    render() {
        return (
          <div className={container}>
            <div className={spinner}>               
              <ClipLoader
                size={100}
                loading={this.state.loading}
              />
            </div>
            <canvas ref={this.canvasRef}  width={VIDEO_WIDTH} height={VIDEO_HEIGHT}/>
            <div hidden={this.state.loading}>
              <Grid container spacing={2} justify="center" alignContent="center"> 
                <Grid item>
                  <Button variant="contained" color="primary" onClick={this.changeBackground}>Change Background</Button>
                </Grid>
                <Grid item> <Typography id="continuous-slider" gutterBottom> Opaque </Typography></Grid>
                <Grid item className={slider}>
                  <Slider min={0.0} max={1.0} step={0.01} value={this.state.opaque} onChange={this.changeOpaque} aria-labelledby="continuous-slider" />
                </Grid>
              </Grid>
              
            </div>
            
          </div>
        );
      }
}

async function toPixels(tensor, canvas){     
    const [height, width] = tensor.shape.slice(0, 2);
    const alpha = tf.fill([height, width, 1], 255, 'int32');
    const rgba = tf.concat([tensor, alpha], 2);
    
    const bytes = await rgba.data();
    
    const pixelData = new Uint8ClampedArray(bytes);
    const imageData = new ImageData(pixelData, width, height);
    
    displayRenderCanvasCtx.putImageData(imageData, 0, 0);

    const ctx = canvas.getContext('2d');  
    ctx.drawImage(displayRenderCanvas, 0, 0,  canvas.width, canvas.height);
}


export default ArVideoConf;