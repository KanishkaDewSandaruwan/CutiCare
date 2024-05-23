import { loadLayersModel, node, image } from '@tensorflow/tfjs-node';
import { readFileSync } from 'fs';
import path from 'path';

// Load the model
let model;

(async () => {
    model = await loadLayersModel('../../../config/model.json');
})();

const classifyImage = async (imagePath) => {
    const imageBuffer = readFileSync(imagePath);
    const imageTensor = node.decodeImage(imageBuffer, 3);
    const resizedImage = image.resizeBilinear(imageTensor, [224, 224]);
    const normalizedImage = resizedImage.div(255.0).expandDims(0);

    const prediction = model.predict(normalizedImage);
    const isRash = (await prediction.data())[0] > 0.5;

    return isRash;
};

export default { classifyImage };