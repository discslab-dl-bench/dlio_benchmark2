from src.common.enumerations import Shuffle
from src.format.reader_handler import FormatReader
import numpy as np
import math
from numpy import random

class NPZReader(FormatReader):
    def __init__(self):
        super().__init__()

    def read(self, epoch_number):
        super().read(epoch_number)
        packed_array = []
        for file in self._local_file_list:
            with np.load(file) as data:
                rows = data['x']
                packed_array.append({
                    'dataset': rows,
                    'current_sample': 0,
                    'total_samples': rows.shape[2]
                })
        self._dataset =  packed_array

    def next(self):
        super().next()
        for element in self._dataset:
            current_index = element['current_sample']
            total_samples = element['total_samples']
            num_sets = list(range(0, int(math.ceil(total_samples / self.batch_size))))
            if self.memory_shuffle != Shuffle.OFF:
                if self.memory_shuffle == Shuffle.SEED:
                    random.seed(self.seed)
                random.shuffle(num_sets)
            for num_set in num_sets:
                yield element['dataset'][:][:][num_set * self.batch_size:(num_set + 1) * self.batch_size - 1]

    def finalize(self):
        pass