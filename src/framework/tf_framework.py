import os
import logging

from src.utils.utility import utcnow
from src.common.error_code import ErrorCodes
from src.framework.framework import Framework
from src.reader.reader_factory import ReaderFactory
from src.profiler.profiler_factory import ProfilerFactory
from src.common.enumerations import FrameworkType, Profiler, FormatType

import tensorflow as tf

print(tf.sysconfig.get_link_flags())
import horovod.tensorflow as hvd

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
hvd.init()

class TFFramework(Framework):
    __instance = None

    def __init__(self, profiling):
        super().__init__()
        self.profiling = profiling
        if profiling:
            self.tensorboard = ProfilerFactory.get_profiler(Profiler.TENSORBOARD)
        self.reader_handler = None

    def init_reader(self, format_type):
        if format_type == FormatType.DATA_LOADER:
            raise Exception(str(ErrorCodes.EC1001))
        self.reader_handler = ReaderFactory.get_format(format_type)

    def get_type(self):
        return FrameworkType.TENSORFLOW

    @staticmethod
    def get_instance(profiling):
        """ Static access method. """
        if TFFramework.__instance is None:
            TFFramework.__instance = TFFramework(profiling)
        return TFFramework.__instance

    def barrier(self):
        """
        Barrier implementation using horovod's all-reduce
        """
        const = tf.constant(1)
        reduced = hvd.allreduce(const)

    def rank(self):
        return hvd.rank()

    def size(self):
        return hvd.size()

    def start_framework_profiler(self):
        if self.profiling:
            self.tensorboard.start()

    def stop_framework_profiler(self):
        if self.profiling:
            self.tensorboard.stop()

    def trace_object(self, string, step, r):
        return tf.profiler.experimental.Trace(string, step_num=step, _r=r)

    def checkpoint(self, step_number):
        logging.info(f"{utcnow()} Starting checkpoint: Step {step_number}")
        """
        Performs Checkpointing for a specific step number. It writes different file of different sizes.
        """
        output_folder = self.arg_parser.args.output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        model_file = os.path.join(output_folder, f"model_{step_number}_{self.args.my_rank}.bin")
        meta_file = os.path.join(output_folder, f"meta_{step_number}_{self.args.my_rank}.bin")
        index_file = os.path.join(output_folder, f"index_{step_number}_{self.args.my_rank}.bin")

        f = open(model_file, "w")
        string_val = "x" * self.args.model_size 
        f.write(string_val)
        f.close()
        # Should these scale with the model size?
        f = open(index_file, "w")
        string_val = "x" * (17371)
        f.write(string_val)
        f.close()
        f = open(meta_file, "w")
        string_val = "x" * (24740228)
        f.write(string_val)
        f.close()
        logging.info(f"{utcnow()} Ending checkpoint: Step {step_number}")

    def compute(self, epoch_number, step, computation_time):
        tf.function(self.model)(epoch_number, step, computation_time)

    def get_reader(self):
        return self.reader_handler
