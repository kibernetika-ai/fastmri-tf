from mrcnn import utils
import glob
import os
import cv2
import numpy as np
import logging
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.model import log
import argparse
import tensorflow as tf

class MyDataset(utils.Dataset):

    def __init__(self,files,data_set, height, width):
        super().__init__()
        self.data_set = data_set
        self.height = height
        self.width = width
        self.files = files
        self.load_data()

    def load_data(self):
        self.add_class("generator", 1, "person")
        for i in range(len(self.files)):
            self.add_image("generator",i,self.files[i])

    def load_image(self, image_id):
        name = self.files[image_id]
        img = cv2.imread(os.path.join(self.data_set, 'images', name))
        img = cv2.resize(img, (self.width, self.height))
        return img[:, :, ::-1]

    def image_reference(self, image_id):
        return self.files[image_id]

    def load_mask(self, image_id):
        name = self.files[image_id]
        img = cv2.imread(os.path.join(self.data_set, 'masks', name))
        img = cv2.resize(img, (self.width, self.height))
        img = img[:,:,0]
        img[img < 200] = 0
        img[img >= 200] = 1
        return np.expand_dims(img.astype(np.bool),axis=2), np.ones((1), np.int32)


class MyConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "generator"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128,256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


def create_arg_parser():
    conf_parser = argparse.ArgumentParser(
        add_help=False
    )
    conf_parser.add_argument(
        '--checkpoint_dir',
        default=os.environ.get('TRAINING_DIR', 'training') + '/' + os.environ.get('BUILD_ID', '1'),
        help='Directory to save checkpoints and logs')
    args, remaining_argv = conf_parser.parse_known_args()
    parser = argparse.ArgumentParser(
        parents=[conf_parser],
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    checkpoint_dir = args.checkpoint_dir
    logging.getLogger().setLevel('INFO')
    tf.logging.set_verbosity(tf.logging.INFO)
    parser.add_argument('--batch_size', default=8, type=int, help='Mini batch size')
    parser.add_argument('--num_epochs_1', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--num_epochs_2', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images')
    parser.add_argument('--data_set', type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--init_weight', type=str,default='coco')
    return parser,checkpoint_dir



if __name__ == '__main__':
    parser,checkpoint_dir = create_arg_parser()
    args= parser.parse_args()
    logging.getLogger().setLevel('INFO')
    files = glob.glob(args.data_set + '/masks/*.*')
    train_files = []
    validate_files = []
    train_count = 0.8*len(files)
    for i in range(len(files)):
        name = os.path.basename(files[i])
        if i < train_count:
            train_files.append(name)
        else:
            validate_files.append(name)

    train_set = MyDataset(train_files,args.data_set,args.resolution,args.resolution)
    train_set.prepare()
    val_set = MyDataset(validate_files,args.data_set,args.resolution,args.resolution)
    val_set.prepare()

    config = MyConfig()
    config.IMAGES_PER_GPU = args.batch_size
    config.STEPS_PER_EPOCH = len(train_files)
    config.IMAGE_MIN_DIM = args.resolution
    config.IMAGE_MAX_DIM = args.resolution

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=checkpoint_dir)

    COCO_MODEL_PATH = os.path.join("mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)


    init_with = args.init_weight  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    model.train(train_set, val_set,
                learning_rate=config.LEARNING_RATE,
                epochs=args.num_epochs_1,
                layers='heads')

    model.train(train_set, train_set,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=args.num_epochs_2,
                layers="all")