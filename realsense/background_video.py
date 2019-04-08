import argparse

import cv2
from ml_serving.drivers import driver
import numpy as np
import pyrealsense2 as rs
from scipy import ndimage


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='.bag file path')
    parser.add_argument(
        '--model',
        help='Background removal model path',
        required=True,
    )
    return parser.parse_args()


def main(args):
    # Create a context object. This object owns the
    # handles to all connected realsense devices
    drv = driver.load_driver('tensorflow')
    serving = drv()
    serving.load_model(args.model)

    gray = 55
    threshold = 1.25
    back = None

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(args.input, repeat_playback=True)

    # Configure the pipeline to stream the depth stream
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color) #, 640, 480, rs.format.rgb8, 30)
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # Create opencv window to render image in
    cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)

    use_realsense = False
    while True:
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        depth_frame = np.asanyarray(depth_frame.get_data())
        color_frame = np.asanyarray(color_frame.get_data())

        color_frame = color_frame[:, :, ::-1]

        if back is None:
            back = np.full([color_frame.shape[0], color_frame.shape[1], 1], gray)

        show_frame = process_frame(
            serving,
            color_frame,
            depth_frame,
            threshold,
            back,
            use_realsense=use_realsense
        )

        images = np.vstack((color_frame, show_frame))

        # Render image in opencv window
        cv2.imshow("Video", images)

        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break
        if key == 32:
            use_realsense = not use_realsense
        if key in {ord('+'), ord('=')}:
            threshold += 0.025
            print(threshold)
        if key in {ord('-'), ord('_')}:
            threshold -= 0.025
            print(threshold)


def process_frame(serving, frame, depth_frame, threshold, background, use_realsense=False):
    frame = frame.astype(np.float32)
    inputs = cv2.resize(frame, (160, 160))
    inputs = np.asarray(inputs, np.float32) / 255.0
    outputs = serving.predict({'image': np.expand_dims(inputs, axis=0)})

    mask = outputs['output'][0]
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask = np.expand_dims(mask, 2)

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
        show_frame = frame * mask_3d + background * (1 - mask_3d)
        show_frame = np.ascontiguousarray(show_frame, np.uint8)
    else:
        show_frame = frame * mask + background * (1 - mask)
        show_frame = np.ascontiguousarray(show_frame, np.uint8)

    return show_frame


if __name__ == '__main__':
    main(parse_args())
