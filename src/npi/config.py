import torch
from npi.models.classifiers import Classifier, GenerationClassifier

from npi.models.npi import NPINetwork


class NPIConfig:
    def __init__(
        self,
        device: torch.device,
        gpt_model="gpt2",
        npi_type="adversarial",
        save_folder="models/npi",
        content_classifier_path=None,
        generation_classifier_path=None,
        batch_size=10,
        perturbation_indices=[5, 11],
        max_seq_len=10,
        num_seq_iters=10,
    ):
        self.gpt_model = gpt_model
        self.npi_type = npi_type
        self.save_folder = save_folder
        self.content_classifier_path = content_classifier_path
        self.batch_size = batch_size
        self.perturbation_indices = perturbation_indices
        self.max_seq_len = max_seq_len
        self.num_seq_iters = num_seq_iters
        self.device = device
