from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.atari.atari_env import AtariEnv

from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.core.network import ConvNetwork
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.misc.instrument import run_experiment_lite
from rllab.misc.parser import atari_arg_parser

import tensorflow as tf
'''
FLAGS = {}
FLAGS.env_name = "PongNoFrameskip-v0"
FLAGS.batch_size = 100000
FLAGS.max_steps_per_episode = 4500
FLAGS.num_episodes_train = 10000
FLAGS.step_size = 0.01
FLAGS.discount = 0.99
'''

parser = atari_arg_parser()
parser.add_argument('--n_itr', type=int, default=int(500))
parser.add_argument('--log_dir', help='log directory', default=None)
parser.add_argument('--n_cpu', type=int, default=int(1))
parser.add_argument('--n_parallel', type=int, default=int(16))
args = parser.parse_args()


def main(_):

  env = TfEnv(AtariEnv(
      args.env, force_reset=True, record_video=False, record_log=False, resize_size=84))

  policy = CategoricalMLPPolicy(
      name='policy',
      env_spec=env.spec,
      prob_network=ConvNetwork(
          name='prob_network',
          input_shape=env.observation_space.shape,
          output_dim=env.action_space.n,
          # number of channels/filters for each conv layer
          conv_filters=(32, 64, 64),
          # filter size
          conv_filter_sizes=(8, 4, 3),
          conv_strides=(4, 2, 1),
          conv_pads=('VALID', 'VALID', 'VALID'),
          hidden_sizes=(512,),
          hidden_nonlinearity=tf.nn.relu,
          output_nonlinearity=tf.nn.softmax,
      )
  )

  baseline = ZeroBaseline(env.spec)

  algo = TRPO(
      env=env,
      policy=policy,
      baseline=baseline,
      batch_size=100000,
      max_path_length=4500,
      n_itr=args.n_itr,
      discount=0.99,
      step_size=0.01,
      optimizer_args={"subsample_factor":0.1}
#       plot=True
  )

  config = tf.ConfigProto(allow_soft_placement=True,
                          intra_op_parallelism_threads=args.n_cpu,
                          inter_op_parallelism_threads=args.n_cpu)
  config.gpu_options.allow_growth = True  # pylint: disable=E1101
  sess = tf.Session(config=config)
  sess.__enter__()
  algo.train(sess)

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