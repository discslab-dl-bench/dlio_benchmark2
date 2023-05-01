"""
   Copyright (c) 2022, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import os
import math
import hydra
import logging
import numpy as np
from time import time, perf_counter_ns
from numpy import random

# Reduce TF and CUDA logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Remove PyTorch warning when libtorch_cuda_cu.so isn't found
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from src.utils.utility import utcnow, measure_performance
from omegaconf import DictConfig, OmegaConf
from src.utils.statscounter import StatsCounter
from hydra.core.config_store import ConfigStore
from src.utils.config import LoadConfig, ConfigArguments
from src.common.enumerations import Profiler, DatasetType, StorageType
from src.profiler.profiler_factory import ProfilerFactory
from src.framework.framework_factory import FrameworkFactory
from src.data_generator.generator_factory import GeneratorFactory
from src.storage.storage_factory import StorageFactory


def get_compute_time(workload, num_gpus, batch_size):
    if workload == 'dlrm':
        mean_coefs, mean_bias = [6.31850988e-03, 1.47435947e-06], -0.0035671384995684535
        std_coefs, std_bias = [3.35919846e-04, 6.32267933e-08], -0.0013904127910698847
    elif workload == 'unet3d':
        mean_coefs, mean_bias = [0.00211888, 0.27448834], 0.27738914081193955
        std_coefs, std_bias = [0.00429292, 0.00705136], -0.024112225795433102
    elif workload == 'bert':
        mean_coefs, mean_bias = [0.00622823, 0.11688357], 0.32936322533391393
        std_coefs, std_bias = [ 0.004484, -0.00128301], 0.009669258568234716
    else:
        raise Exception(f'Unknown workload {workload}. Please configure a simulation sleep time in the config.')
    compute_time_mean = np.dot(mean_coefs, [num_gpus, batch_size]) + mean_bias
    compute_time_std = np.dot(std_coefs, [num_gpus, batch_size]) + std_bias
    return compute_time_mean, compute_time_std

    
class DLIOBenchmark(object):
    """
    The Benchmark represents the I/O behavior of deep learning applications.
    """

    def __init__(self, cfg):
        """
        This initializes the DLIO benchmark. Intialization includes:
        <ul>
            <li> argument parser </li>
            <li> profiler instances </li>
            <li> internal components </li>
            <li> local variables </li>
        </ul>
        """
        self.args = ConfigArguments.get_instance()
        LoadConfig(self.args, cfg)
        self.args.validate()
        try:
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            self.args.output_folder = hydra_cfg['runtime']['output_dir']
        except:
            self.args.output_folder = 'output/'
        self.output_folder = self.args.output_folder
        self.logfile = os.path.join(self.output_folder, self.args.log_file)
        self.data_folder = self.args.data_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.storage_root = self.args.storage_root
        self.storage = StorageFactory().get_storage(self.args.storage_type, self.storage_root, self.args.framework)
        if self.args.storage_root:
            self.storage.create_namespace(exist_ok=True)

        self.framework = FrameworkFactory().get_framework(self.args.framework,
                                                          self.args.do_profiling)

        self.my_rank = self.args.my_rank = self.framework.rank()
        self.comm_size = self.args.comm_size = self.framework.size()
        # Delete previous logfile
        if self.my_rank == 0:
            if os.path.isfile(self.logfile):
                os.remove(self.logfile)
        self.framework.barrier()
        # Configure the logging library
        log_level = logging.DEBUG if self.args.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            handlers=[
                logging.FileHandler(self.logfile, mode = "a", encoding='utf-8'),
                logging.StreamHandler()
            ],
            format='[%(levelname)s] %(message)s [%(pathname)s:%(lineno)d]'  # logging's max timestamp resolution is msecs, we will pass in usecs in the message
        )
 

        if self.args.my_rank==0:
            logging.info(f"{utcnow()} Running DLIO with {self.args.comm_size} process(es)")
            try:
                logging.info(f"{utcnow()} Reading YAML config file './configs/workload/{hydra_cfg.runtime.choices.workload}.yaml'" )
            except:
                pass
        
        self.generate_only = self.args.generate_only
        self.do_profiling = self.args.do_profiling

        self.data_generator = None
        self.num_files_train = self.args.num_files_train
        self.num_samples = self.args.num_samples_per_file
        self.total_training_steps = self.args.total_training_steps
        self.total_eval_steps = self.args.total_eval_steps
        
        self.epochs = self.args.epochs
        self.batch_size = self.args.batch_size
        workload = hydra_cfg.runtime.choices.workload
        self.num_gpus = self.args.num_gpus

        # Take the compute time values from config or generate them
        if self.args.computation_time == 0:
            self.computation_time, self.computation_time_stdev = get_compute_time(workload, self.num_gpus, self.batch_size)
        else:
            self.computation_time = self.args.computation_time
            self.computation_time_stdev = self.args.computation_time_std
        
        logging.info(f'Sleep time distrib for {workload} with batch size {self.batch_size} and {self.num_gpus} GPUs: {self.computation_time} {self.computation_time_stdev}')

        if self.do_profiling:
            self.profiler = ProfilerFactory().get_profiler(self.args.profiler)

        if self.args.generate_data:
            self.data_generator = GeneratorFactory.get_generator(self.args.format)

        # Checkpointing support
        self.do_checkpoint = self.args.do_checkpoint
        self.steps_between_checkpoints = self.args.steps_between_checkpoints
        self.epochs_between_checkpoints = self.args.epochs_between_checkpoints
        self.checkpoint_after_epoch = self.args.checkpoint_after_epoch
        
        # Evaluation support
        self.do_eval = self.args.do_eval
        self.num_files_eval = self.args.num_files_eval

        self.batch_size_eval = self.args.batch_size_eval
        self.eval_time = self.args.eval_time
        self.eval_time_stdev = self.args.eval_time_stdev        
        self.eval_after_epoch = self.args.eval_after_epoch
        self.epochs_between_evals = self.args.epochs_between_evals
        self.steps_between_evals = self.args.steps_between_evals
        self.eval_num_samples = self.args.eval_num_samples_per_file

        # Hold various lists/dicts for statistics
        self.time_to_load_train_batch = []
        self.time_to_process_train_batch = []

        self.time_to_load_eval_batch = []
        self.time_to_process_eval_batch = []

        self.epoch_time_ranges = []
        self.eval_time_ranges = []

        self.ckpt_time_ranges = []

        # Indexed by epoch number, contains start-end timestamps and other information
        self.per_epoch_stats = {}
        self.stats = StatsCounter()

    def initialize(self):
        """
        Initializes the benchmark runtime.
        - It generates the required data
        - Start profiling session for Darshan and Tensorboard.
        """
        self.framework.barrier()
        if self.args.debug and self.args.my_rank == 0:
            input("Debug mode: Press enter to start\n")

        if self.args.generate_data:
            if self.args.my_rank==0:
                logging.info(f"{utcnow()} Starting data generation")
            self.data_generator.generate()
            # important to have this barrier to ensure that the data generation is done for all the ranks
            self.framework.barrier()
            if self.args.my_rank==0:
                logging.info(f"{utcnow()} Generation done")

        if not self.generate_only and self.do_profiling:
            self.profiler.start()
            self.framework.start_framework_profiler()
            self.framework.barrier()
            if self.args.my_rank == 0:
                logging.info(f"{utcnow()} Profiling Started with {self.args.profiler}")
        self.framework.init_reader(self.args.format, self.args.data_loader)
        self.framework.barrier()
        self.total_compute_time = 0.0
    
    def _eval(self, epoch):
        """
        Evaluation loop will read a separate dataset and has its own own computation time.
        """
        step = 1
        total = math.floor(self.num_samples * self.num_files_eval / self.batch_size_eval / self.comm_size)
        t0 = time() 
        reader = self.framework.get_reader(DatasetType.VALID)
        total_compute_time = 0.0
        start_time = time()
        for batch in reader.next():
            self.stats.eval_batch_loaded(epoch, step, t0)

            if self.eval_time > 0:
                if self.eval_time_stdev > 0:
                    eval_time = max(0, random.normal(self.eval_time, self.eval_time_stdev))
                else:
                    eval_time = self.eval_time
                total_compute_time += eval_time
                self.framework.compute(epoch, step, eval_time)

            self.stats.eval_batch_processed(epoch, step, t0)

            step += 1
            if step > total or step >= self.total_eval_steps:
                break
                
            self.framework.barrier()
            t0 = time()
        end_time = time()
        self.total_compute_time += total_compute_time
        if self.my_rank == 0 and total_compute_time >0.:            
            logging.info(f"{utcnow()} Epoch {epoch} [evaluation] accelerator_under_utilization: {(end_time - start_time - total_compute_time) / total_compute_time}")
        return step - 1
        
    def _train(self, epoch):
        """
        Training loop for reading the dataset and performing training computations.
        :return: returns total steps.
        """
        block = 1   # A continuous period of training steps, ended by checkpointing
        block_step = overall_step = 1   # Steps are taken within blocks
        max_steps = math.floor(self.num_samples * self.num_files_train / self.batch_size / self.comm_size)
        self.next_eval_step = self.steps_between_evals        
        self.next_checkpoint_step = self.steps_between_checkpoints         

        # Start the very first block
        self.stats.start_block(epoch, block)
        reader = self.framework.get_reader(dataset_type=DatasetType.TRAIN)

        total_compute_time = 0.0
        start_time = time()

        t_iter = t0 = perf_counter_ns()
        for batch in reader.next():

            if self.my_rank == 0:
                logging.info(f"load_batch_mem {perf_counter_ns() - t0}")
                t0 = perf_counter_ns()

            self.framework.barrier()
            # Log a new block, unless it's the first one which we've already logged before the loop
            if block_step == 1 and block != 1:
                self.stats.start_block(epoch, block)
            
            if self.computation_time > 0:
                self.framework.trace_object("Train", overall_step, 1)
                if self.computation_time_stdev > 0:
                    computation_time = random.normal(self.computation_time, self.computation_time_stdev)
                else:
                    computation_time = self.computation_time
                total_compute_time += computation_time
                self.framework.compute(epoch, block_step, computation_time)
            self.framework.barrier()

            if self.my_rank == 0:
                logging.info(f"all_compute {perf_counter_ns() - t0}")
                logging.info(f"step_end {perf_counter_ns() - t_iter}")

            # Perform evaluation during epochs if required
            # Assume that evaluation happens on all GPU
            if self.do_eval and overall_step == self.next_eval_step:
                # Before starting the evaluation, terminating the current block
                self.stats.end_block(epoch, block, block_step)

                # Initialize the eval data loader & perform evaluation
                self.stats.start_eval(epoch)
                self.framework.get_reader(DatasetType.VALID).read(epoch)
                self.framework.barrier()
                self._eval(epoch)
                self.stats.end_eval(epoch)
                self.framework.barrier()
                self.framework.get_reader(DatasetType.VALID).finalize()
                self.next_eval_step += self.steps_between_evals

                # Start recording the next block
                self.stats.start_block(epoch, block)


            if self.do_checkpoint and self.steps_between_checkpoints >= 0 and overall_step == self.next_checkpoint_step:
                self.stats.end_block(epoch, block, block_step)
                self.stats.start_ckpt(epoch, block, overall_step)
                self.framework.checkpoint(epoch, overall_step)
                self.stats.end_ckpt(epoch, block)
                self.framework.barrier()
                block += 1
                # Reset the number of steps after every checkpoint to mark the start of a new block
                block_step = 1
                self.next_checkpoint_step += self.steps_between_checkpoints
                self.stats.start_block(epoch, block)
            else:
                block_step += 1

            if overall_step >= max_steps or overall_step == self.total_training_steps:
                self.framework.barrier()
                if self.args.my_rank==0:
                    logging.info(f"{utcnow()} Maximum number of steps reached")
                if (block_step!=1 and self.do_checkpoint) or (not self.do_checkpoint):
                    self.stats.end_block(epoch, block, block_step-1)
                break
                
            overall_step += 1
            t_iter = t0 = perf_counter_ns()

        end_time = time()
        self.total_compute_time += total_compute_time
        if self.my_rank == 0 and total_compute_time >0.0:            
            logging.info(f"{utcnow()} Epoch {epoch} [training] accelerator_under_utilization: {(end_time - start_time - total_compute_time) / total_compute_time}")
        return overall_step

    def run(self):
        """
        Run the total epochs for training. 
        On each epoch, it prepares dataset for reading, it trains, and finalizes the dataset.
        If evaluation is enabled, it reads the eval dataset, performs evaluation and finalizes.
        """
        self.start_timestamp=time()
        if not self.generate_only:
            # Print out the expected number of steps for each epoch and evaluation
            if self.my_rank == 0:
                total = math.floor(self.num_samples * self.num_files_train / self.batch_size / self.comm_size)
                logging.info(f"{utcnow()} Max steps per epoch: {total} = {self.num_samples} * {self.num_files_train} / {self.batch_size} / {self.comm_size} (samples per file * num files / batch size / comm size)")

                if self.do_eval:
                    total = math.floor(self.num_samples * self.num_files_eval / self.batch_size_eval / self.comm_size)
                    logging.info(f"{utcnow()} Steps per eval: {total} = {self.num_samples} * {self.num_files_eval} / {self.batch_size_eval} / {self.comm_size} (samples per file * num files / batch size eval / comm size)")
            
            # Keep track of the next epoch at which we will evaluate
            next_eval_epoch = self.eval_after_epoch
            next_checkpoint_epoch = self.checkpoint_after_epoch

            for epoch in range(1, self.epochs + 1):
                self.stats.start_epoch(epoch)

                # Initialize the dataset
                self.framework.get_reader(dataset_type=DatasetType.TRAIN).read(epoch)
                self.framework.barrier()

                steps = self._train(epoch)
                self.stats.end_epoch(epoch, steps)
                logging.debug(f"{utcnow()} Rank {self.my_rank} returned after {steps} steps.")

                self.framework.barrier()
                self.framework.get_reader(DatasetType.TRAIN).finalize()

                # Perform evaluation if enabled
                if self.do_eval and epoch >= next_eval_epoch:
                    next_eval_epoch += self.epochs_between_evals

                    self.stats.start_eval(epoch)
                
                    # Initialize the eval dataset
                    self.framework.get_reader(DatasetType.VALID).read(epoch)
                    self.framework.barrier()
                    
                    self._eval(epoch)
                    self.stats.end_eval(epoch)

                    self.framework.barrier()
                    self.framework.get_reader(DatasetType.VALID).finalize()

                if self.do_checkpoint and epoch == next_checkpoint_epoch:
                    next_checkpoint_epoch += self.epochs_between_checkpoints
                    # UNET3D Checkpoints only at the very end
                    self.stats.start_ckpt(epoch, 0, steps)
                    self.framework.checkpoint(epoch, steps)
                    self.stats.end_ckpt(epoch, 0)
                    self.framework.barrier()

        self.stop_timestamp=time()

        
    def finalize(self):
        """
        It finalizes the dataset once training is completed.
        """
        self.framework.barrier()
        if not self.generate_only:
            if self.do_profiling:
                self.profiler.stop()
                self.framework.stop_framework_profiler()
                self.framework.barrier()
                if self.my_rank == 0:
                    logging.info(f"{utcnow()} Profiling stopped")
            if not self.args.keep_files:
                logging.info(f"{utcnow()} Keep files set to False. Deleting dataset")
                self.framework.barrier()
                if self.my_rank == 0:
                    if self.storage.get_node(self.args.data_folder):
                        self.storage.delete_node(self.args.data_folder)
                        logging.info(f"{utcnow()} Deleted data files")
            
            # Save collected stats to disk
            self.stats.save_data()
        self.framework.barrier()
        total_elapsed_time = self.stop_timestamp - self.start_timestamp

        if self.my_rank == 0 and self.total_compute_time >0.:            
            logging.info(f"{utcnow()} Overall accelerator_under_utilization: {(total_elapsed_time - self.total_compute_time) / self.total_compute_time}")
 
        if self.my_rank==0:
            logging.info(f"{utcnow()} Saved outputs in {self.output_folder}")        

@measure_performance
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg : DictConfig) -> None:
    """
    The main method to start the benchmark runtime.
    """
    os.environ["DARSHAN_DISABLE"] = "1"
    benchmark = DLIOBenchmark(cfg['workload'])
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()

if __name__ == '__main__':
    OmegaConf.register_new_resolver("eval", eval)
    main()
    exit(0)
