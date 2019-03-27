import argparse
import random
import numpy as np
import tensorflow as tf
import logging
import os
import json
import time
import cv2



def generate(src_dir,dest_dir,resolution):
    with open(src_dir + '/annotations/instances_train2017.json') as f:
        data = json.load(f)
    step = 0
    data = data['annotations']
    if not tf.gfile.Exists(dest_dir+'/images'):
        tf.gfile.MakeDirs(dest_dir+'/images')
    if not tf.gfile.Exists(dest_dir+'/masks'):
        tf.gfile.MakeDirs(dest_dir+'/masks')
    for a in data:
        if a['category_id'] == 1 and a['iscrowd'] == 0:
            name = '{:012d}.jpg'.format(a['image_id'])
            fname = src_dir + '/train2017/{}'.format(name)
            segmentation = a['segmentation']
            area = a['area']
            if os.path.exists(fname):
                if len(segmentation) < 4 and len(segmentation)>0:
                    img = cv2.imread(fname, cv2.IMREAD_COLOR)[:, :,::-1]
                    img_area = img.shape[0]*img.shape[1]
                    if area>img_area*0.1:
                        m = np.zeros((img.shape[0], img.shape[1]), np.uint8)
                        img = cv2.resize(img, (resolution, resolution))
                        cv2.imwrite(dest_dir+'/images/'+name,img[:, :,::-1])
                        for s in segmentation:
                            p = np.array(s, np.int32)
                            p = np.reshape(p, (1, int(p.shape[0] / 2), 2))
                            m = cv2.fillPoly(m, p, color=(255, 255, 255))
                        m = cv2.resize(m, (resolution, resolution))
                        if os.path.exists(dest_dir+'/masks/'+name):
                            pm = cv2.imread(dest_dir+'/masks/'+name, cv2.IMREAD_COLOR)[:,:]
                            m = np.maximum(m,pm)
                        cv2.imwrite(dest_dir+'/masks/'+name,m)
                        step+=1
                        logging.info("Step: {}/{}".format(step,len(data)))



def main(args):
    generate(args.src_dir,args.dest_dir,args.resolution)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    logging.getLogger().setLevel('INFO')
    tf.logging.set_verbosity(tf.logging.INFO)
    parser.add_argument('--src_dir', type=str,required=True)
    parser.add_argument('--dest_dir', type=str,required=True)
    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images')
    return parser


if __name__ == '__main__':
    logging.getLogger().setLevel('INFO')
    args = create_arg_parser().parse_args()
    random.seed(int(time.time()))
    np.random.seed(int(time.time()))
    main(args)
