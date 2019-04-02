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
    pipeline.start(config)

    # Create opencv window to render image in
    cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)

    while True:
        # Create a pipeline object. This object configures
        # the streaming camera and owns it's handle
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Colorize depth frame to jet colormap
        # depth_color_frame = rs.colorizer().colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_frame = np.asanyarray(depth_frame.get_data())
        color_frame = np.asanyarray(color_frame.get_data())

        depth_color_image = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET
        )

        if depth_color_image.shape[0] >= 720:
            size = (int(depth_color_image.shape[1] // 1.5), int(depth_color_image.shape[0] // 1.5))
            depth_color_image = cv2.resize(depth_color_image, size, interpolation=cv2.INTER_AREA)

        color_image = color_frame
        if color_image.shape != depth_color_image.shape:
            color_image = cv2.resize(
                color_image,
                (depth_color_image.shape[1], depth_color_image.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        images = np.vstack((color_image, depth_color_image))
        # images = color_image

        # Render image in opencv window
        cv2.imshow("Video", images)

        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break
        if key == 32:
            # Resize color image to depth size
            color_frame = cv2.resize(
                color_frame,
                (depth_frame.shape[1], depth_frame.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            np.save('depth.npy', depth_frame)
            np.save('color.npy', color_frame)


if __name__ == '__main__':
    main(parse_args())
