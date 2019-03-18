import PIL.Image
import numpy as np
import logging
import io


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

def preprocess(inputs):
    image = inputs['images'][0]
    image = PIL.Image.open(io.BytesIO(image))
    image = image.convert('RGB')
    image = image.resize((299,299))
    image = np.asarray(image,np.float32)/127.5-1
    return {
        'images': np.stack([image], axis=0),
    }


def postprocess(outputs):
    LOG.info('outputs: {}'.format(outputs))
    predictions = outputs['output']
    line = []
    for i in predictions[0]:
        t = word_index.get(i,None)
        if t is None:
            continue;
        line.append(t)
    LOG.info('outputs: {}'.format(' '.join(line)))
    return {'output': ' '.join(line).encode()}
