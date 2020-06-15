import React, {useEffect, useRef} from 'react';
import * as tf from '@tensorflow/tfjs';
import {container} from './DebugPanel.module.scss';
import {TensorDisplay} from '../../util';
import {MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT} from '../../constants';

const tensorDisplay = new TensorDisplay(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);

const DebugPanel = (props) => {
    const [inputTensor, maskTensor, stats] = props.input;
    const canvasDebugInputRef = useRef();
    const canvasDebugMaskRef = useRef();
    const canvasDebugOutputRef = useRef();

    useEffect(()=> {
        async function displayTensors(){
            tf.engine().startScope();
            const input = tf.image.resizeBilinear(inputTensor, [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH]).mul(tf.scalar(255));
            const mask = tf.image.resizeBilinear(maskTensor, [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH ]);
            const GB = tf.fill([MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, 2],255, 'int32');
            const R = tf.ones([MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, 1]).sub(mask).mul(tf.scalar(255));
            const maskImg = tf.concat([R,GB], 2);
            await tensorDisplay.show(input, canvasDebugInputRef.current);
            await tensorDisplay.show(maskImg, canvasDebugMaskRef.current);
            await tensorDisplay.show(input, canvasDebugOutputRef.current);
            await tensorDisplay.show(maskImg, canvasDebugOutputRef.current, 0.5);
            tf.engine().endScope();
        }
        if(inputTensor && maskTensor)
            displayTensors();
    },[inputTensor, maskTensor]);

    const displayStats = () => {
        const stringStats = JSON.stringify(stats).replace(new RegExp('"', 'g'), ' ');
        return <ul> {
            stringStats.substring(1, stringStats.length - 1).split(',')
                .map((stat,i) => <li key={i}>{stat}</li>)
        } </ul>;
    }

    return (
    <div className={container}>
        <canvas ref={canvasDebugInputRef} width={MODEL_INPUT_WIDTH} height={MODEL_INPUT_HEIGHT} /> 
        <canvas ref={canvasDebugMaskRef} width={MODEL_INPUT_WIDTH} height={MODEL_INPUT_HEIGHT} /> 
        <canvas ref={canvasDebugOutputRef} width={MODEL_INPUT_WIDTH} height={MODEL_INPUT_HEIGHT} />
        { stats &&  displayStats()}
    </div>
    );
}

export default DebugPanel;