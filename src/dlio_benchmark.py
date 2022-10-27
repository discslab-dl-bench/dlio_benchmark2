"""
   Copyright 2021 UChicago Argonne, LLC

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
from time import time

from src.common.enumerations import Profiler
from src.data_generator.generator_factory import GeneratorFactory
from src.framework.framework_factory import FrameworkFactory
from src.profiler.profiler_factory import ProfilerFactory
from src.utils.argument_parser import ArgumentParser
from src.utils.utility import utcnow

import math
import os
import shutil
import logging
import pandas as pd

# Remove (some) TF and CUDA logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class DLIOBenchmark(object):
    """
    The Benchmark represents the I/O behavior of deep learning applications.
    """

    def __init__(self):
        """
        This initializes the DLIO benchmark. Intialization includes:
        <ul>
            <li> argument parser </li>
            <li> profiler instances </li>
            <li> internal components </li>
            <li> local variables </li>
        </ul>
        """
        self.arg_parser = ArgumentParser.get_instance()
        self.output_folder = self.arg_parser.args.output_folder
        self.logfile = os.path.join(self.output_folder, self.arg_parser.args.log_file)
        # self.iostat_file = os.path.join(self.output_folder, 'iostat.json')

        # Configure the logging library
        log_level = logging.DEBUG if self.arg_parser.args.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            handlers=[
                logging.FileHandler(self.logfile, mode = "a", encoding='utf-8'),
                logging.StreamHandler()
            ],
            format='%(message)s [%(pathname)s:%(lineno)d]'  # logging's max timestamp resolution is msecs, we will pass in usecs in the message
        )

        self.framework = FrameworkFactory().get_framework(self.arg_parser.args.framework,
                                                          self.arg_parser.args.profiling)

        self.my_rank = self.arg_parser.args.my_rank = self.framework.rank()
        self.comm_size = self.arg_parser.args.comm_size = self.framework.size()
        self.framework.init_reader(self.arg_parser.args.format)

        self.generate_only = self.arg_parser.args.generate_only
        self.do_profiling = self.arg_parser.args.profiling

        self.data_generator = None
        self.num_files_train = self.arg_parser.args.num_files_train
        self.num_samples = self.arg_parser.args.num_samples
        
        self.epochs = self.arg_parser.args.epochs
        self.batch_size = self.arg_parser.args.batch_size
        self.computation_time = self.arg_parser.args.computation_time

        if self.do_profiling:
            self.darshan = ProfilerFactory().get_profiler(Profiler.DARSHAN)

        if self.arg_parser.args.generate_data:
            self.data_generator = GeneratorFactory.get_generator(self.arg_parser.args.format)

        # Checkpointing support
        self.do_checkpoint = self.arg_parser.args.checkpoint
        self.checkpoint_steps = self.arg_parser.args.checkpoint_steps
        self.checkpoint_epochs = self.arg_parser.args.checkpoint_epochs
        self.checkpoint_after_epoch = self.arg_parser.args.checkpoint_after_epoch
        
        # Evaluation support
        self.do_eval = self.arg_parser.args.do_eval
        self.num_files_eval = self.arg_parser.args.num_files_eval

        self.batch_size_eval = self.arg_parser.args.batch_size_eval
        self.eval_time = self.arg_parser.args.eval_time
        self.eval_after_epoch = self.arg_parser.args.eval_after_epoch
        self.eval_every_epoch = self.arg_parser.args.eval_every_epoch
        
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

    def initialize(self):
        """
        Initializes the benchmark runtime.
        - It generates the required data
        - Start profiling session for Darshan and Tensorboard.
        """
        if self.arg_parser.args.debug and self.arg_parser.args.my_rank == 0:
            input("Press enter to start\n")
        self.framework.barrier()
        if self.arg_parser.args.generate_data:
            logging.info(f"{utcnow()} Starting data generation")
            self.data_generator.generate()
            logging.info(f"{utcnow()} Generation done")

        if self.do_profiling:
            self.darshan.start()
            self.framework.start_framework_profiler()
            self.framework.barrier()
            if self.arg_parser.args.my_rank == 0:
                logging.info(f"{utcnow()} Profiling Started")\

        self.framework.barrier()

    def _eval(self, epoch_number):
        """
        Evaluation loop will read a separate dataset and has its own own computation time.
        """
        step = 1
        total = math.ceil(self.num_samples * self.num_files_eval / self.batch_size_eval / self.comm_size)
        t1 = time() 
        for batch in self.framework.get_reader().next():
            logging.info(f"{utcnow()} Rank {self.my_rank} loaded {self.batch_size_eval} samples in {time() - t1} seconds")

            self.time_to_load_eval_batch.append([utcnow(), epoch_number, time() - t1])
            if self.eval_time > 0:
                self.framework.compute(epoch_number, step, self.eval_time)

            self.time_to_process_eval_batch.append([utcnow(), epoch_number, time() - t1])
            step += 1
            if step > total:
                return step - 1
            self.framework.barrier()
            logging.info(f"{utcnow()} Rank {self.my_rank} processed {self.batch_size_eval} samples in {time() - t1} seconds")
            t1 = time()
        return step - 1

    def _train(self, epoch):
        """
        Training loop for reading the dataset and performing training computations.
        :return: returns total steps.
        """
        step = total_steps = 1
        block = 1   # A continuous period of training steps, ended by checkpointing
        max_steps = math.ceil(self.num_samples * self.num_files_train / self.batch_size / self.comm_size)
        
        t1 = time() 
        for batch in self.framework.get_reader().next():
            logging.debug(f"{utcnow()} Rank {self.my_rank} loaded {self.batch_size} samples in {time() - t1} seconds")
            # Can I save this to a map directly?
            self.time_to_load_train_batch.append([utcnow(), epoch, time() - t1])

            # Blocks are periods of training steps separated by checkpointing
            if step == 1 and self.my_rank == 0:
                logging.info(f"{utcnow()} Starting block {block}")
            
            if self.computation_time > 0:
                self.framework.compute(epoch, step, self.computation_time)

            self.framework.barrier()

            self.time_to_process_train_batch.append([utcnow(), epoch, time() - t1])
            logging.info(f"{utcnow()} Rank {self.my_rank} processed {self.batch_size} samples in {time() - t1} seconds")

            step += 1
            total_steps += 1

            if total_steps > max_steps:
                self.framework.barrier()
                if self.my_rank == 0:
                    logging.info(f"{utcnow()} Ending block {block}")
                return total_steps - 1

            t1 = time()
            start_ts = utcnow()
            if self.do_checkpoint and epoch > self.checkpoint_after_epoch and epoch % self.checkpoint_epochs and total_steps % self.checkpoint_steps == 0:
                logging.info(f"{utcnow()} Ending block {block}")
                block += 1
                if self.my_rank == 0:
                    self.framework.checkpoint(total_steps)
                self.framework.barrier()
                # Reset the number of steps after every checkpoint to mark the start of a new block
                step = 1

            self.ckpt_time_ranges.append([epoch, start_ts, utcnow()])
            t1 = time()

        return total_steps - 1

    
    def run(self):
        """
        Run the total epochs for training. 
        On each epoch, it prepares dataset for reading, it trains, and finalizes the dataset.
        If evaluation is enabled, it reads the eval dataset, performs evaluation and finalizes.
        """
        if not self.generate_only:
            # Print out the expected number of steps for each epoch and evaluation
            if self.my_rank == 0:
                total = math.ceil(self.num_samples * self.num_files_train / self.batch_size / self.comm_size)
                logging.info(f"{utcnow()} Steps per epoch: {total} = {self.num_samples} * {self.num_files_train} / {self.batch_size} / {self.comm_size} (samples per file * num files / batch size / comm size)")
                if self.do_eval:
                    total = math.ceil(self.num_samples * self.num_files_eval / self.batch_size_eval / self.comm_size)
                    logging.info(f"{utcnow()} Steps per eval: {total} = {self.num_samples} * {self.num_files_eval} / {self.batch_size_eval} / {self.comm_size} (samples per file * num files / batch size eval / comm size)")
            
            # Keep track of the next epoch at which we will evaluate
            next_eval_at = self.eval_after_epoch
            for epoch in range(1, self.epochs + 1):
                # start_ts = 

                if self.my_rank == 0:
                    self.per_epoch_stats[epoch] = {
                        'start': utcnow()
                    }
                    logging.info(f"{utcnow()} Starting epoch {epoch}")

                start_time = time()
                # Initialize the dataset
                self.framework.get_reader().read(epoch, do_eval=False)
                self.framework.barrier()

                if self.my_rank == 0:
                    logging.info(f"{utcnow()} Training dataset initialized for all ranks in {time() - start_time} seconds")
                
                start_time = time()
                steps = self._train(epoch)
                self.framework.barrier()

                if self.my_rank == 0:
                    self.per_epoch_stats[epoch]['end'] = utcnow()
                    self.per_epoch_stats[epoch]['duration'] = self.per_epoch_stats[epoch]['end'] - self.per_epoch_stats[epoch]['start']
                    # statscounter.epoch_end(epoch)
                    logging.info(f"{utcnow()} Ending epoch {epoch} - {steps} steps completed in {time() - start_time} seconds")
                    # self.epoch_time_ranges.append([epoch, start_ts, utcnow()])

                self.framework.get_reader().finalize()

                # Perform evaluation if enabled
                if self.do_eval and epoch == next_eval_at:
                    start_ts = utcnow()
                    next_eval_at += self.eval_every_epoch
                
                    if self.my_rank == 0:
                        self.per_epoch_stats[epoch]['eval'] = {
                            'start': utcnow()
                        }
                        logging.info(f"{utcnow()} Starting eval")

                    start_time = time()
                    # Initialize the eval dataset
                    self.framework.get_reader().read(epoch, do_eval=True)
                    self.framework.barrier()

                    if self.my_rank == 0:
                        logging.info(f"{utcnow()} Eval dataset loaded for all ranks in {time() - start_time} seconds")
                    
                    start_time = time()
                    steps = self._eval(epoch)
                    self.framework.barrier()

                    if self.my_rank == 0:
                        self.per_epoch_stats[epoch]['eval']['end'] = utcnow()
                        logging.info(f"{utcnow()} Ending eval - {steps} steps completed in {time() - start_time} seconds")
                        # self.eval_time_ranges.append([epoch, start_ts, utcnow()])

                    self.framework.get_reader().finalize()

    def save(self, df_like, name):
        """
        Helper function to save a dataframe-like to a csv.
        """
        df = pd.DataFrame(df_like)
        df.to_csv(os.path.join(self.output_folder, name), header=False, index=False)

    def finalize(self):
        """
        It finalizes the dataset once training is completed.
        """
        self.framework.barrier()
        if not self.generate_only:
            # self.iostat.stop()
            if self.do_profiling:
                self.darshan.stop()
                self.framework.stop_framework_profiler()
                self.framework.barrier()
                if self.my_rank == 0:
                    logging.info(f"{utcnow()} profiling stopped")
            if not self.arg_parser.args.keep_files:
                logging.info(f"{utcnow()} Keep files set to False. Deleting dataset")
                self.framework.barrier()
                if self.my_rank == 0:
                    if os.path.exists(self.arg_parser.args.data_folder):
                        shutil.rmtree(self.arg_parser.args.data_folder)
                        logging.info(f"{utcnow()} Deleted data files")
            
            # Dump statistic counters to files for postprocessing
            # Overall stats
            if self.my_rank == 0:
                self.save(self.epoch_time_ranges, 'epoch_time_ranges.csv')
                if self.do_eval:
                    self.save(self.eval_time_ranges, 'eval_time_ranges.csv')
                if self.do_checkpoint:
                    self.save(self.ckpt_time_ranges, f'ckpt_time_ranges.csv')

            # Save individual rank stats
            self.save(self.time_to_load_train_batch, f'{self.my_rank}_time_to_load_train_batch.csv')
            self.save(self.time_to_process_train_batch, f'{self.my_rank}_time_to_process_train_batch.csv')
            if self.do_eval:
                self.save(self.time_to_load_eval_batch, f'{self.my_rank}_time_to_load_eval_batch.csv')
                self.save(self.time_to_process_eval_batch, f'{self.my_rank}_time_to_process_eval_batch.csv')

        self.framework.barrier()


def main():
    """
    The main method to start the benchmark runtime.
    """
    os.environ["DARSHAN_DISABLE"] = "1"
    benchmark = DLIOBenchmark()
    benchmark.initialize()
    benchmark.run()
    benchmark.finalize()


if __name__ == '__main__':
    main()
    exit(0)
