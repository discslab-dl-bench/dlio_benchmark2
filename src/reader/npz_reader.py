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
from src.common.enumerations import Shuffle, FileAccess
from src.reader.reader_handler import FormatReader
import numpy as np
import math
from numpy import random
import tensorflow as tf


from src.utils.utility import progress

class NPZReader(FormatReader):
    """
    Reader for NPZ files
    """
    def __init__(self):
        super().__init__()

    def read(self, epoch_number):
        """
        for each epoch it opens the npz files and reads the data into memory
        :param epoch_number:
        """
        super().read(epoch_number)
        packed_array = []
        for file in self._local_file_list:
            with np.load(file, allow_pickle=True) as data:
                rows = data['x']
                packed_array.append({
                    'dataset': rows,
                    'current_sample': 0,
                    'total_samples': rows.shape[2]
                })
        self._dataset =  packed_array

    def next(self):
        """
        The iterator of the dataset just performs memory sub-setting for each portion of the data.
        :return: piece of data for training.
        """
        super().next()
        total = 0
        count = 1
        for element in self._dataset:
            current_index = element['current_sample']
            total_samples = element['total_samples']
            if FileAccess.MULTI == self.file_access:
                num_sets = list(range(0, int(math.ceil(total_samples / self.batch_size))))
            else:
                total_samples_per_rank = int(total_samples / self.comm_size)
                part_start, part_end = (int(total_samples_per_rank * self.my_rank / self.batch_size),
                                        int(total_samples_per_rank * (self.my_rank + 1) / self.batch_size))
                num_sets = list(range(part_start, part_end))
            total += len(num_sets)
            if self.memory_shuffle != Shuffle.OFF:
                if self.memory_shuffle == Shuffle.SEED:
                    random.seed(self.seed)
                random.shuffle(num_sets)
            for num_set in num_sets:
                with tf.profiler.experimental.Trace('HDF5 Input', step_num=num_set / self.batch_size, _r=1):
                    progress(count, total, "Reading NPZ Data")
                    count += 1
                    images = element['dataset'][:][:][num_set * self.batch_size:(num_set + 1) * self.batch_size - 1]
                yield images

    def finalize(self):
        pass
