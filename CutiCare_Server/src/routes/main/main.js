const express = require('express');

const {
    checkRash
} = require('../../mvc/main/MainController');
const { uploadImage } = require('../../../config/fileUpload');

module.exports = (config) => {
    const router = express.Router();

    //login and create
    router.post('/check', uploadImage.single('image'), checkRash);
    return router;
};
