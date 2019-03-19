import PIL.Image
import numpy as np
import logging
import io
import base64
import json

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

PARAMS = {
    'dictionary': './dictionary.csv',
}
def dictionary(d):
    word_index = {}
    max_index = 0
    with open(d, 'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            p = line.split(',')
            if len(p) != 2:
                continue
            index = int(p[0])
            max_index = max(index, max_index)
            word_index[index] = p[1]
    word_index[max_index + 1] = '<end>'
    word_index[0] = '<start>'
    return word_index

word_index = {}

def init_hook(**kwargs):
    global PARAMS
    PARAMS.update(kwargs)
    LOG.info("Init hooks {}".format(kwargs))
    global word_index
    word_index = dictionary(PARAMS['dictionary'])
    LOG.info("Init hooks")

def preprocess(inputs,ctx, **kwargs):
    image = inputs['images'][0]
    original = PIL.Image.open(io.BytesIO(image))
    image = original.convert('RGB')
    image = image.resize((299,299))
    image = np.asarray(image,np.float32)/127.5-1
    original.putalpha(1)
    ctx.original = original
    return {
        'images': np.stack([image], axis=0),
    }


def postprocess(outputs, ctx, **kwargs):
    LOG.info('outputs: {}'.format(outputs))
    predictions = outputs['labels']
    attentions = outputs['attentions']
    line = []
    for i in predictions[0]:
        t = word_index.get(i,None)
        if t is None or t == '<end>':
            continue
        t = t.replace('_',' ')
        line.append(t)
    line = ' '.join(line)
    img_base = ctx.original
    table = []
    for i in predictions[0]:
        t = word_index.get(i,None)
        if t is None or t == '<end>':
            continue
        t = t.replace('_',' ')
        attention = np.resize(attentions[i][0],(8,8))
        image = PIL.Image.fromarray(attention)
        image.putalpha(0.6)
        image = image.resize((299,299))
        comp = PIL.Image.alpha_composite(img_base, image)
        image_bytes = io.BytesIO()
        comp.save(image_bytes, format='PNG')
        encoded = base64.encodebytes(image_bytes.getvalue()).decode()
        table.append(
            {
                'type': 'text',
                'name': t,
                'prob': float(1),
                'image': encoded
            }
        )
    image_bytes = io.BytesIO()
    img_base.save(image_bytes, format='PNG')
    return {
        'output': image_bytes.getvalue(),
        'caption_output': np.array([line], dtype=np.string_),
        'table_output': json.dumps(table),
    }
