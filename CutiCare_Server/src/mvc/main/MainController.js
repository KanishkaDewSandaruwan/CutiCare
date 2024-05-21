const MainModel = require("./MainModel");

(async () => {
    model = await tf.loadLayersModel('../../../config/model.json');
})();

const checkRash = async (req, res) => {
    try {
        const imageBuffer = req.file.buffer;
        const imageTensor = tf.node.decodeImage(imageBuffer);
        const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]);
        const normalizedImage = resizedImage.div(255.0).expandDims(0);

        const prediction = model.predict(normalizedImage);
        const isRash = (await prediction.data())[0] > 0.5;

        res.status(200).send({ message: 'Checking is Success', isRash });
    } catch (error) {
        console.error(error);
        res.status(500).send({ error: 'Error fetching data from the database' });
    }
};

module.exports = {
    checkRash
}