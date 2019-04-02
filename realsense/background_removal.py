import argparse

import cv2
from ml_serving.drivers import driver
import numpy as np
from scipy import ndimage


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test background'
    )
    parser.add_argument(
        '--model',
        help='Background removal model path',
        required=True,
    )
    parser.add_argument(
        '--color',
        help='Path to color file',
    )
    parser.add_argument(
        '--depth',
        help='Path to depth file',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=1.25,
        help='Threshold distance for pixels drop in the background (relative to foreground)',
    )

    return parser.parse_args()


def main(args):
    drv = driver.load_driver('tensorflow')
    serving = drv()
    serving.load_model(args.model)

    color_frame = np.load(args.color)
    depth_frame = np.load(args.depth)
    width, height = color_frame.shape[1], color_frame.shape[0]

    gray = 55
    back = np.full([height, width, 1], gray)

    frame = color_frame
    frame = frame.astype(np.float32)
    inputs = cv2.resize(frame, (160, 160))
    inputs = np.asarray(inputs, np.float32) / 255.0
    outputs = serving.predict({'image': np.expand_dims(inputs, axis=0)})

    mask = outputs['output'][0]
    mask = cv2.resize(mask, (width, height))

    mask = np.expand_dims(mask, 2)
    threshold = args.threshold
    use_realsense = False
    while True:
        if use_realsense:
            mask_2d = np.copy(mask).reshape(mask.shape[0], mask.shape[1])
            center = np.round(ndimage.measurements.center_of_mass(mask_2d)).astype(np.int)
            x = center[0]
            y = center[1]
            depth = depth_frame[x][y]
            max_depth = depth * threshold
            # Drop pixels which have depth more than foreground * threshold
            mask_2d[depth_frame >= max_depth] = 0

            mask_3d = mask_2d.reshape(mask.shape[0], mask.shape[1], 1)
            show_frame = np.concatenate([frame, frame * mask_3d + back * (1 - mask_3d)], axis=1)
            show_frame = np.ascontiguousarray(show_frame[:, :, ::-1], np.uint8)
        else:
            show_frame = np.concatenate([frame, frame * mask + back * (1 - mask)], axis=1)
            show_frame = np.ascontiguousarray(show_frame[:, :, ::-1], np.uint8)

        cv2.imshow('Video', show_frame)
        key = cv2.waitKey(1)
        # Wait 'q' or Esc
        if key == ord('q') or key == 27:
            break
        if key == 32:
            use_realsense = not use_realsense
        if key in {ord('+'), ord('=')}:
            threshold += 0.025
            print(threshold)
        if key in {ord('-'), ord('_')}:
            threshold -= 0.025
            print(threshold)

    cv2.destroyAllWindows()
    print('Finished')


if __name__ == '__main__':
    main(parse_args())
