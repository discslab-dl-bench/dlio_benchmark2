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

        samples_per_iteration = 10_000_000

        for i in range(self.my_rank, int(self.total_files_to_generate), self.comm_size):
            out_path_spec = self.storage.get_uri(self._file_list[i])

            if i < self.num_files_train:
                samples_written = 0

                with open(out_path_spec, 'ab') as output_file:

                    while samples_written < self.num_samples:
                        progress(samples_written, self.num_samples, "Generating DLRM training data samples")
                        X_int = np.random.randint(2557264, size = (samples_per_iteration, 13))
                        X_cat = np.random.randint(0, CATEGORY_RANGES, size = (samples_per_iteration, 26))
                        y = np.random.randint(2, size=samples_per_iteration)
                        np_data = np.concatenate([y.reshape(-1, 1), X_int, X_cat], axis=1)
                        np_data = np_data.astype(np.int32)
                        output_file.write(np_data.tobytes())
                        output_file.flush()
                        samples_written += samples_per_iteration

            else:
                # Generating Evaluation files
                samples_written = 0

                with open(out_path_spec, 'ab') as output_file:
                    while samples_written < self.eval_num_samples_per_file:
                        progress(samples_written, self.num_samples, "Generating DLRM eval data samples")
                        X_int = np.random.randint(2557264, size = (samples_per_iteration, 13))
                        X_cat = np.random.randint(0, CATEGORY_RANGES, size = (samples_per_iteration, 26))
                        y = np.random.randint(2, size=samples_per_iteration)
                        np_data = np.concatenate([y.reshape(-1, 1), X_int, X_cat], axis=1)
                        np_data = np_data.astype(np.int32)
                        output_file.write(np_data.tobytes())
                        output_file.flush()
                        samples_written += samples_per_iteration
