import argparse

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='.bag file path')
    return parser.parse_args()


def main(args):
    # Create a context object. This object owns the
    # handles to all connected realsense devices
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

    while True:
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        depth_frame = np.asanyarray(depth_frame.get_data())
        color_frame = np.asanyarray(color_frame.get_data())

        depth_color_image = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET
        )

        color_image = color_frame
        images = np.vstack((color_image, depth_color_image))

        # Render image in opencv window
        cv2.imshow("Video", images)

        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break
        if key == 32:
            # Resize color image to depth size
            np.save('image_depth.npy', depth_frame)
            np.save('image_color.npy', color_frame)


if __name__ == '__main__':
    main(parse_args())
