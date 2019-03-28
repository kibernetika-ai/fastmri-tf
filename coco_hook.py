import PIL.Image
import numpy as np
import logging
import io
import base64
import json

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def preprocess(inputs,ctx, **kwargs):
    image = inputs['image'][0]
    original = PIL.Image.open(io.BytesIO(image)).convert('RGB')
    original = original.resize((320,320))
    input = np.asarray(original,np.float32)
    ctx.input = input
    input = input/127.5-1
    return {
        'image': np.stack([input], axis=0),
    }


def postprocess(outputs, ctx, **kwargs):
    mask = outputs['output']
    logging.info('Mask shape {}'.format(mask.shape))
    mask[np.less(mask,128)]=0
    mask[np.greater_equal(mask,128)]=1
    output = ctx.input*mask
    output = (output+1)*127.5
    image_bytes = io.BytesIO()
    img = PIL.Image.fromarray(output[0].astype(np.uint8))
    img.save(image_bytes, format='PNG')
    return {
        'output': image_bytes.getvalue(),
    }
