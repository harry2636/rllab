from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.atari.atari_env import AtariEnv

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.core.network import ConvNetwork
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer

from rllab.misc.instrument import run_experiment_lite
from rllab.misc.parser import atari_arg_parser
from rllab.misc import logger

import tensorflow as tf
import copy

parser = atari_arg_parser()
parser.add_argument('--n_itr', type=int, default=int(500))
parser.add_argument('--log_dir', help='log directory', default=None)
parser.add_argument('--n_cpu', type=int, default=int(1))
parser.add_argument('--n_parallel', type=int, default=int(16))
parser.add_argument('--resize_size', type=int, default=int(52))
parser.add_argument('--batch_size', type=int, default=int(100000))
parser.add_argument('--step_size', type=float, default=float(0.01))
parser.add_argument('--discount_factor', type=float, default=float(0.995))
parser.add_argument('--value_function', help='Choose value funciton baseline', choices=['zero', 'conj', 'adam'], default='zero')
parser.add_argument('--num_slices', help='Slice big batch into smaller ones to prevent OOM', type=int, default=int(1))
parser.add_argument('--reward_no_scale', help='Turn off reward scaling', action='store_true')

args = parser.parse_args()
logger.log(str(args))

def get_value_network(env):
    value_network = ConvNetwork(
        name='value_network',
        input_shape=env.observation_space.shape,
        output_dim=1,
        # number of channels/filters for each conv layer
        conv_filters=(16, 32),
        # filter size
        conv_filter_sizes=(8, 4),
        conv_strides=(4, 2),
        conv_pads=('VALID', 'VALID'),
        hidden_sizes=(256,),
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=None,
        batch_normalization=False
    )
    return value_network

def main(_):

  env = TfEnv(AtariEnv(
      args.env, force_reset=True, record_video=False, record_log=False, resize_size=args.resize_size))

  policy_network = ConvNetwork(
          name='prob_network',
          input_shape=env.observation_space.shape,
          output_dim=env.action_space.n,
          # number of channels/filters for each conv layer
          conv_filters=(16, 32),
          # filter size
          conv_filter_sizes=(8, 4),
          conv_strides=(4, 2),
          conv_pads=('VALID', 'VALID'),
          hidden_sizes=(256,),
          hidden_nonlinearity=tf.nn.relu,
          output_nonlinearity=tf.nn.softmax,
          batch_normalization=False
      )
  policy = CategoricalMLPPolicy(
      name='policy',
      env_spec=env.spec,
      prob_network=policy_network
  )

  if (args.value_function == 'zero'):
      baseline = ZeroBaseline(env.spec)
  else:
      value_network = get_value_network(env)
      baseline_batch_size = args.batch_size * 10

      if (args.value_function == 'conj'):
          baseline_optimizer = ConjugateGradientOptimizer(
              subsample_factor=1.0,
              num_slices=args.num_slices
          )
      elif (args.value_function == 'adam'):
          baseline_optimizer = FirstOrderOptimizer(max_epochs=3, batch_size=512)
      else:
          logger.log("Inappropirate value function")
          exit(0)


      baseline = GaussianMLPBaseline(
          env.spec,
          num_slices=args.num_slices,
          regressor_args=dict(
              step_size=0.01,
              mean_network=value_network,
              optimizer=baseline_optimizer,
              subsample_factor=1.0,
              batchsize=baseline_batch_size,
              use_trust_region=False
          )
      )


  algo = TRPO(
      env=env,
      policy=policy,
      baseline=baseline,
      batch_size=args.batch_size,
      max_path_length=4500,
      n_itr=args.n_itr,
      discount=args.discount_factor,
      step_size=args.step_size,
      clip_reward=(not args.reward_no_scale),
      optimizer_args={"subsample_factor":1.0,
                      "num_slices":args.num_slices}
#       plot=True
  )


  config = tf.ConfigProto(allow_soft_placement=True,
                          intra_op_parallelism_threads=args.n_cpu,
                          inter_op_parallelism_threads=args.n_cpu)
  config.gpu_options.allow_growth = True  # pylint: disable=E1101
  sess = tf.Session(config=config)
  sess.__enter__()
  algo.train(sess)

'''
main("a")
exit(0)
'''

if __name__ == '__main__':
    run_experiment_lite(
        main,
        # Number of parallel workers for sampling
        n_parallel=args.n_parallel,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=args.seed,
        log_dir=args.log_dir,
        # plot=True,
    )
