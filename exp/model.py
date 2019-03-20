import tensorflow as tf
import exp.attention as attention
import exp.unet as unet
import exp.classify as classify

def null_dataset():
    def _input_fn():
        return None

    return _input_fn

class Model(tf.estimator.Estimator):
    def __init__(
            self,
            params=None,
            model_dir=None,
            config=None,
            warm_start_from=None
    ):
        self._params = params
        def _model_fn(features, labels, mode, params, config):
            if params['net']=='unet':
                return unet.model_fn(
                    features=features,
                    labels=labels,
                    mode=mode,
                    params=params,
                    config=config)
            elif params['net']=='iv3_classify':
                return classify.model_fn(
                    features=features,
                    labels=labels,
                    mode=mode,
                    params=params,
                    config=config)
            else:
                return attention.model_fn(
                    features=features,
                    labels=labels,
                    mode=mode,
                    params=params,
                    config=config)
        super(Model, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from
        )
    def get_input(self,is_training):
        if self._params['net']=='unet':
            return unet.input_fn(self.params,is_training)
        elif self._params['net']=='iv3_classify':
            return classify.input_fn(self.params,is_training)
        else:
            return attention.input_fn(self.params,is_training)