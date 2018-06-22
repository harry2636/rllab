import numpy as np

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
from rllab.misc.ext import iterate_minibatches_generic
from sandbox.rocky.tf.regressors.gaussian_mlp_regressor import GaussianMLPRegressor



class GaussianMLPBaseline(Baseline, Parameterized):

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            input_shape=None,
            num_seq_inputs=1,
            num_slices=1,
            regressor_args=None,
    ):
        Serializable.quick_init(self, locals())
        super(GaussianMLPBaseline, self).__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = GaussianMLPRegressor(
            input_shape= (env_spec.observation_space.flat_dim * num_seq_inputs,),
            output_dim=1,
            name="vf",
            **regressor_args
        )
        self.num_slices = num_slices

    @overrides
    def fit(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        self._regressor.fit(observations, returns.reshape((-1, 1)))

    @overrides
    def predict(self, path):
        obs = path["observations"]
        batch_size = len(obs) // self.num_slices

        prediction = []
        for batch in iterate_minibatches_generic(input_lst=[obs], batchsize=batch_size, shuffle=False):
            part_obs, = batch
            part_pred = self._regressor.predict(part_obs).flatten()
            prediction.append(part_pred)

        full_pred = np.concatenate(prediction, axis=0)

        #real_result = self._regressor.predict(path["observations"]).flatten()
        return full_pred

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)
