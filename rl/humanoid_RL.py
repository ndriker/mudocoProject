import os
import gym
import ray
import glfw
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from os.path import expanduser
import tempfile
import pickle
import logging
from typing import Dict, Callable
from ray.tune.result import NODE_IP
from ray.tune.logger import UnifiedLogger
import datetime

logger = logging.getLogger(__name__)


def datetime_str():
    # format is ok for file/directory names
    date_string = datetime.datetime.now().strftime("%I.%M.%S%p_%b-%d-%Y")
    return date_string

class SpaceSavingLogger(UnifiedLogger):
    """Unified result logger for TensorBoard, rllab/viskit, plain json.

    Arguments:
        config: Configuration passed to all logger creators.
        logdir: Directory for all logger creators to log to.
        loggers (list): List of logger creators. Defaults to CSV, Tensorboard,
            and JSON loggers.
        sync_function (func|str): Optional function for syncer to run.
            See ray/python/ray/tune/syncer.py
        should_log_result_fn: (func) Callable that takes in a train result and outputs
            whether the result should be logged (bool). Used to save space by only logging important
            or low frequency results.
    """

    def __init__(self,
                 config,
                 logdir,
                 trial=None,
                 loggers=None,
                 # sync_function=None,
                 should_log_result_fn: Callable[[Dict], bool] = None,
                 print_log_dir=True,
                 delete_hist_stats=True):

        super(SpaceSavingLogger, self).__init__(config=config,
                                                logdir=logdir,
                                                trial=trial,
                                                loggers=loggers,
                                                # sync_function=sync_function)
                                               )
        self.print_log_dir = print_log_dir
        self.delete_hist_stats = delete_hist_stats
        self.should_log_result_fn = should_log_result_fn

    def on_result(self, result):
        if self.print_log_dir:
            print(f"log dir is {self.logdir}")
        should_log_result = True
        if self.should_log_result_fn is not None:
            should_log_result = self.should_log_result_fn(result)

        if self.delete_hist_stats:
            if "hist_stats" in result:
                del result["hist_stats"]
            try:
                for key in result["info"]["learner"].keys():
                    if "td_error" in result["info"]["learner"][key]:
                        del result["info"]["learner"][key]["td_error"]
            except KeyError:
                pass

        if should_log_result:
            for _logger in self._loggers:
                _logger.on_result(result)
            # self._log_syncer.set_worker_ip(result.get(NODE_IP))
            # self._log_syncer.sync_down_if_needed()


def get_trainer_logger_creator(base_dir: str,
                               experiment_name: str,
                               should_log_result_fn: Callable[[dict], bool],
                               delete_hist_stats: bool = True):

    logdir_prefix = f"{experiment_name}_{datetime_str()}"

    def trainer_logger_creator(config, logdir=None, trial=None):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        if not logdir:
            logdir = tempfile.mkdtemp(
                prefix=logdir_prefix, dir=base_dir)

        return SpaceSavingLogger(config=config,
                                 logdir=logdir,
                                 should_log_result_fn=should_log_result_fn,
                                 delete_hist_stats=delete_hist_stats)

    return trainer_logger_creator


class MujocoCustomEnv(gym.Env):

    def __init__(self, env_config):
        self.BEST_TOTAL_REWARD = 0
        self.CURRENT_TOTAL_REWARD = 0
        self.moves = []
        self.env = gym.make('Humanoid-v2')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        # self.env.render()
        self.moves.append(action)
        self.obs, reward, done, _ = self.env.step(action)
        self.CURRENT_TOTAL_REWARD += reward
        return self.obs, reward, done, _

    def reset(self):
        if (self.CURRENT_TOTAL_REWARD >= self.BEST_TOTAL_REWARD):
            self.BEST_TOTAL_REWARD = self.CURRENT_TOTAL_REWARD    
            with open('bestActionHumanoid.pkl', 'wb') as bestAction:
                pickle.dump(self.moves, bestAction)
                
        self.moves = []
        self.CURRENT_TOTAL_REWARD = 0
        return self.env.reset()

if __name__ == "__main__":
    num_workers = 8
    use_gpu = False

    ray.init(num_cpus=num_workers, num_gpus=int(use_gpu)) #manually set these or dont and ray will just see what the computer has

    results_base_dir = os.path.join(expanduser("~"), "my_cool_custom_ray_results_dir")

    trainer = ppo.PPOTrainer(
        env=MujocoCustomEnv,
        logger_creator=get_trainer_logger_creator(
            base_dir=results_base_dir,
            experiment_name="my_mujoco",
            should_log_result_fn=lambda result: result["training_iteration"] % 1 == 0,
            delete_hist_stats=False
                                    ),
        config={
  "batch_mode": "truncate_episodes",
  "clip_param": 0.4,
  "entropy_coeff": 0.0,
  "entropy_coeff_schedule": None,
  "env": "<class 'humanoid.MujocoCustomEnv'>",
  "env_config": {},
  "framework": "torch",
  "grad_clip": None,
  "kl_coeff": 0.2,
  "kl_target": 0.01,
  "lambda": 0.95,
  "lr": 3.244976050346995e-05,
  "lr_schedule": None,
  "model": {
    "vf_share_layers": False
  },
  "num_gpus": 0,
  "num_gpus_per_worker": 0,
  "num_sgd_iter": 5,
  "num_workers": 3,
  "observation_filter": "NoFilter",
  "rollout_fragment_length": 200,
  "sgd_minibatch_size": 64,
  "shuffle_sequences": True,
  "simple_optimizer": True,
  "train_batch_size": 4096,
  "use_critic": True,
  "use_gae": True,
  "vf_clip_param": 500.0,
  "vf_loss_coeff": 1.0
}
)
    while True:
        print(f"trainer logdir is {trainer.logdir}")
        train_result = trainer.train()
        # Delete verbose debugging info before printing
        if "hist_stats" in train_result:
            del train_result["hist_stats"]

        print(pretty_print(train_result))