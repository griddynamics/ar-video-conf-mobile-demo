import {fill, concat, dispose} from '@tensorflow/tfjs';

export class TensorDisplay {
    constructor(width, height) {
        this.displayRenderCanvas = document.createElement('canvas');
        this.displayRenderCanvasCtx = this.displayRenderCanvas.getContext('2d');
        this.displayRenderCanvas.width = width;
        this.displayRenderCanvas.height = height;      
    }

    async show(tensor, canvas, opacity = 1){     
        const [height, width] = tensor.shape.slice(0, 2);
        const alpha = fill([height, width, 1], opacity * 255, 'int32');
      
        const rgba = concat([tensor, alpha], 2);
        
        const bytes = await rgba.data();
        dispose([alpha, rgba]);
        
        const pixelData = new Uint8ClampedArray(bytes);
        const imageData = new ImageData(pixelData, width, height);
        
        this.displayRenderCanvasCtx.putImageData(imageData, 0, 0);
      
        const ctx = canvas.getContext('2d');  
        ctx.drawImage(this.displayRenderCanvas, 0, 0,  canvas.width, canvas.height);
    }
}

export function disposeIfSet(tensor){
    if(tensor)
        tensor.dispose();
}

export class DebugStats{
    displayTime = [];
    predictionTime = [];
    overallTime = [];

    constructor(iterationsCount){
        this.iterationsCount = iterationsCount;
    }

    storeTime = (array, time) =>{
        if(array.length === this.iterationsCount)
            array.shift();
        array.push(time);
    };

    storeDisplayTime = (time) => this.storeTime(this.displayTime, time);
    storePredictionTime = (time) => this.storeTime(this.predictionTime, time);
    storeOverallTime = (time) => this.storeTime(this.overallTime, time);

    summary = () => {
        const sum = (x,y) => x + y;
        const avg = (arr) => arr.reduce(sum)/arr.length;

        const avgDt = avg(this.displayTime);
        const avgPt = avg(this.predictionTime) - avgDt;
        const avgOt = avg(this.overallTime);
        
        return {
            display: `${Math.ceil(avgDt)}ms`,
            prediction: `${Math.ceil(avgPt)}ms`,
            overall: `${Math.ceil(avgOt)}ms`
        };
    }
}