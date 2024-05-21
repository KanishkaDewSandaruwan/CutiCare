const multer = require('multer');
const path = require('path');

const doctorStorage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'src/uploads/profile');
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const imageStorage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'src/uploads/images');
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const uploadProfile = multer({ storage: doctorStorage });
const uploadImage = multer({ storage: imageStorage });

module.exports = { uploadProfile, uploadImage };
