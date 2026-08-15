"""Shared unpickler for loading CUDA-trained Bayesian detectors on CPU."""

import io
import pickle

import torch


class CPU_Unpickler(pickle.Unpickler):
    """Load CUDA-pickled detectors on CPU and remap pre-refactor module paths.

    Detector pickles bundle a torch SynthID logits_processor and were saved when
    the modules lived at the repo root; remap them to the current `src.*` layout.
    """

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        if module == 'bayesian_detector':
            module = 'src.bayesian_detector'
        elif module in ['model_utils', 'execution_utils', 'report_generator']:
            module = f'src.{module}'
        return super().find_class(module, name)
