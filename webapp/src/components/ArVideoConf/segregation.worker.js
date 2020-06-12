import * as tf from '@tensorflow/tfjs';
const MODEL_INPUT_WIDTH = 256;
const MODEL_INPUT_HEIGHT = 256;


const MODEL_PATH = 
    "http://localhost:3000/tfjs/cmd/model.json"; 
    //"https://ar-video-conf-demo-dot-gd-gcp-techlead-experiments.ey.r.appspot.com/tfjs/cmd/model.json";

var model;
var loadedModel = false;

(async () => {
    model = await tf.loadLayersModel(MODEL_PATH, {strict: false});
    loadedModel = true;
})();

onmessage = async (e) => {

    if(!loadedModel){
        await waitAsync();
    }
    // console.time('prediction');
    tf.engine().startScope();
    const x = tf.tensor(e.data, [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, 3]).expandDims(0);
    const y = model.predict(x);
    postMessage(y.dataSync());
    tf.engine().endScope();
    // console.timeEnd('prediction');
};

const waitAsync = () => new Promise((resolve, _) => {
    const interval = setInterval(()=>{
        if(loadedModel){
            resolve();
            clearInterval(interval);
        }
    }, 20);
});

