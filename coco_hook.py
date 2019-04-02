import numpy as np
import logging
import cv2

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

PARAMS = {
    'resolution': '320',
    'scaling': '-1:1'
}


def init_hook(**kwargs):
    global PARAMS
    PARAMS.update(kwargs)
    PARAMS['resolution'] = int(PARAMS.get('resolution', 320))
    LOG.info("Init hooks {}".format(PARAMS))


def preprocess(inputs, ctx, **kwargs):
    image = inputs['image'][0]
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
    ctx.w = image.shape[1]
    ctx.h = image.shape[0]
    if ctx.w > ctx.h:
        if ctx.w > 1024:
            ctx.h = int(float(ctx.h) * 1024.0 / float(ctx.w))
            ctx.w = 1024
    else:
        if ctx.h > 1024:
            ctx.w = int(float(ctx.w) * 1024.0 / float(ctx.h))
            ctx.h = 1024
    ctx.input = cv2.resize(image, (ctx.w, ctx.h))
    ctx.input = np.asarray(ctx.input, np.float32)
    image = cv2.resize(image, (PARAMS['resolution'], PARAMS['resolution']))
    input = np.asarray(image, np.float32)
    if PARAMS['scaling'] == '0:1':
        input = input / 255.0
    else:
        input = input / 127.5 - 1
    return {
        'image': np.stack([input], axis=0),
    }


def postprocess(outputs, ctx, **kwargs):
    mask = outputs['output']
    logging.info('Mask shape {}'.format(mask.shape))
    mask = mask[0]
    if mask.shape[0] != ctx.h or mask.shape[1] != ctx.w:
        mask = cv2.resize(mask, (ctx.w, ctx.h))
    mask = np.expand_dims(mask,2)
    output = ctx.input * mask
    output = output.astype(np.uint8)
    output = output[:, :, ::-1]
    logging.info('Out max {}'.format(np.max(output)))
    logging.info('Out mit {}'.format(np.min(output)))
    logging.info('Out shape {}'.format(output.shape))
    _, buf = cv2.imencode('.png', output)
    image = np.array(buf).tostring()
    return {
        'output': image,
    }
