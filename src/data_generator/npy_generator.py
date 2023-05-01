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

from src.common.enumerations import Compression
from src.data_generator.data_generator import DataGenerator

import logging
import numpy as np
from numpy import random

from src.utils.utility import progress, utcnow
from shutil import copyfile

"""
Generator for creating data in NPZ format.
"""
class NPZGenerator(DataGenerator):
    def __init__(self):
        super().__init__()

    def generate(self):
        """
        Generator for creating UNET3D's NPY format training data.
        """
        super().generate()
        for i in range(self.my_rank, int(self.total_files_to_generate), self.comm_size):

            size1 = random.randint(128, 471)
            size2 = random.randint(186, 444)
            img = np.random.uniform(low=-2.340702, high=2.639792, size=(1, size1, size2, size2))
            mask = np.random.randint(0, 2, size=(1, size1, size2, size2))
            img = img.astype(np.float32)
            mask = mask.astype(np.uint8)
            
            out_path_spec = self.storage.get_uri(self._file_list[i])

            fnx = f"{out_path_spec}_x.npy"
            fny = f"{out_path_spec}_y.npy"
            np.save(fnx, img)
            np.save(fny, mask)

            progress(i+1, self.total_files_to_generate, "Generating NPY Data")
