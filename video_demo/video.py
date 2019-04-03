import cv2
import argparse
import time
from ml_serving.drivers import driver
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(
        description='Test background'
    )
    parser.add_argument(
        '--size',
        default='640x360',
        help='Image size',
    )
    parser.add_argument(
        '--camera',
        help='Full URL to network camera.',
    )
    parser.add_argument('--model')
    return parser


def get_size(scale):
    t = scale.split('x')
    return int(t[0]), int(t[1])


def imresample(img, h, w):
    im_data = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)  # @UndefinedVariable
    return im_data


def add_overlays(frame, frame_rate):
    if frame_rate != 0:
        cv2.putText(
            frame, str(frame_rate) + " fps", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
            thickness=2, lineType=2
        )


def main():
    frame_interval = 2  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0
    start_time = time.time()

    parser = get_parser()
    args = parser.parse_args()

    drv = driver.load_driver('tensorflow')
    serving = drv()
    serving.load_model('./model')
    if args.camera:
        video_capture = cv2.VideoCapture(args.camera)
    else:
        video_capture = cv2.VideoCapture(0)
    width,height = get_size(args.size)
    back = cv2.imread('./newback.jpg')[:, :, ::-1]
    back = cv2.resize(back,(width,height))
    # back = np.full([height, width, 1], 100)
    back = back.astype(np.float32)
    try:
        while True:
            _, frame = video_capture.read()
            #print("Orginal {}".format(frame.shape))

            frame = imresample(frame,height,width)


            if (frame_count % frame_interval) == 0:
                # BGR -> RGB
                frame = frame[:, :, ::-1]
                frame = frame.astype(np.float32)
                input = cv2.resize(frame, (160,160))
                input = np.asarray(input, np.float32)/255.0
                outputs = serving.predict({'image': np.expand_dims(input, axis=0)})
                mask = outputs['output'][0]
                mask = cv2.resize(mask, (width,height))
                mask = np.expand_dims(mask,2)
                frame = np.concatenate([frame,frame * mask+back*(1-mask)],axis=1)
                #print('rgb_frame {}'.format(rgb_frame.shape))
                #rgb_frame = rgb_frame.astype(np.uint8)
                frame = np.ascontiguousarray(frame[:, :, ::-1],np.uint8)

                # Check our current fps
                end_time = time.time()
                if (end_time - start_time) > fps_display_interval:
                    frame_rate = int(frame_count / (end_time - start_time))
                    start_time = time.time()
                    frame_count = 0



                add_overlays(frame,frame_rate/2)


                cv2.imshow('Video', frame)
            frame_count += 1
            key = cv2.waitKey(1)
            # Wait 'q' or Esc
            if key == ord('q') or key == 27:
                break

    except (KeyboardInterrupt, SystemExit) as e:
        print('Caught %s: %s' % (e.__class__.__name__, e))

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    print('Finished')


if __name__ == "__main__":
    main()
