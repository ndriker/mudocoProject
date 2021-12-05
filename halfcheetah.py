import os
import gym
import ray
import glfw
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from os.path import expanduser
import tempfile

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
                 sync_function=None,
                 should_log_result_fn: Callable[[Dict], bool] = None,
                 print_log_dir=True,
                 delete_hist_stats=True):

        super(SpaceSavingLogger, self).__init__(config=config,
                                                logdir=logdir,
                                                trial=trial,
                                                loggers=loggers,
                                                sync_function=sync_function)
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
            self._log_syncer.set_worker_ip(result.get(NODE_IP))
            self._log_syncer.sync_down_if_needed()


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
        self.env = gym.make('HalfCheetah-v2')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        # self.env.render()
        self.obs, reward, done, _ = self.env.step(action)
        return self.obs, reward, done, _

    def reset(self):
        return self.env.reset()

# if __name__ = "__main__":
    # num_workers = 4
    # use_gpu = False

    # # ray.init(num_cpus=num_workers, num_gpus=int(use_gpu)) #manually set these or dont and ray will just see what the computer has

    # results_base_dir = os.path.join(expanduser("~"), "my_cool_custom_ray_results_dir")

    # trainer = ppo.PPOTrainer(
    #     env=MujocoCustomEnv,
    #     logger_creator=get_trainer_logger_creator(
    #         base_dir=results_base_dir,
    #         experiment_name="my_mujoco",
    #         should_log_result_fn=lambda result: result["training_iteration"] % 1 == 0,
    #         delete_hist_stats=False
    #                                 ),
    #     config={
    #     'env_config': {},
    #     'framework': 'torch',
    #     'num_gpus': int(use_gpu),
    #     'num_gpus_per_worker': 0,
    #     'num_workers': num_workers,
    #     'simple_optimizer': True
    # })
    # while True:
    #     print(f"trainer logdir is {trainer.logdir}")
    #     train_result = trainer.train()
    #     # Delete verbose debugging info before printing
    #     if "hist_stats" in train_result:
    #         del train_result["hist_stats"]

    #     print(pretty_print(train_result))