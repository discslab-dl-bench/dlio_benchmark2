"""
The binary file generator designed for simulating DLRM in DLIO
"""

from src.common.enumerations import Compression
from src.data_generator.data_generator import DataGenerator

import logging
import numpy as np
from numpy import random
import math
import os

from src.utils.utility import progress
from shutil import copyfile


CATEGORY_RANGES = [10000000,    38985,    17278,     7418,    20242,        3,
           7120,     1543,       63,  9999978,  2642264,   397262,
             10,     2208,    11931,      155,        4,      976,
             14, 10000000,  9832963, 10000000,   573162,    12969,
            108,       36]


class BINGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    def generate(self):
        """
        Generate binary data for training and testing.
        """
        super().generate()

        for i in range(self.my_rank, int(self.total_files_to_generate), self.comm_size):
            progress(i+1, self.total_files_to_generate, "Generating Binary Data")
            out_path_spec = self.storage.get_uri(self._file_list[i])
            # File size will be different depending on training or validation file
            if i < self.num_files_train:
                # Generating Training files
                segment_size = 91681240*5
                num_instance = self.num_samples #4195198976 for dlrm training
                parts = math.ceil(num_instance / segment_size)
                for k in range(0, parts):
                    num_written = segment_size if k < parts-1 else num_instance - k*segment_size
                    X_int = np.random.randint(2557264, size = (num_written, 13))
                    X_cat = np.random.randint(0, CATEGORY_RANGES, size = (num_instance, 26))
                    y = np.random.randint(2, size=num_written)
                    np_data = np.concatenate([y.reshape(-1, 1), X_int, X_cat], axis=1)
                    np_data = np_data.astype(np.int32)
                    if self.compression != Compression.ZIP:
                        with open(out_path_spec, 'ab') as output_file:
                            output_file.write(np_data.tobytes())
                            output_file.flush()
                            os.fsync(output_file.fileno())
            else:
                # Generating Evaluation files
                segment_size = 91681240*5
                num_instance = self.eval_num_samples_per_file #4195198976 for dlrm training
                parts = math.ceil(num_instance / segment_size)
                for k in range(0, parts):
                    num_written = segment_size if k < parts-1 else num_instance - k*segment_size
                    X_int = np.random.randint(2557264, size = (num_written, 13))
                    X_cat = np.random.randint(0, CATEGORY_RANGES, size = (num_instance, 26))
                    y = np.random.randint(2, size=num_written)
                    np_data = np.concatenate([y.reshape(-1, 1), X_int, X_cat], axis=1)
                    np_data = np_data.astype(np.int32)
                    if self.compression != Compression.ZIP:
                        with open(out_path_spec, 'ab') as output_file:
                            output_file.write(np_data.tobytes())
                            output_file.flush()
                            os.fsync(output_file.fileno())
