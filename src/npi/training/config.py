import torch
from npi.models.classifiers import Classifier, GenerationClassifier

from npi.models.npi import NPINetwork


class NPIConfig:
    def __init__(
        self,
        device: torch.device,
        gpt_model="gpt2",
        npi_model_folder=None,
        content_classifier_path=None,
        generation_classifier_path=None,
        batch_size=10,
        perturbation_indices=[5, 11],
        max_seq_len=10,
        num_seq_iters=10,
    ):
        self.gpt_model = gpt_model
        self.npi_model_folder = npi_model_folder
        self.content_classifier_path = content_classifier_path
        self.batch_size = batch_size
        self.perturbation_indices = perturbation_indices
        self.max_seq_len = max_seq_len
        self.num_seq_iters = num_seq_iters
        self.device = device

    def load_all_models(self, input_activs_shape, input_targ_shape):

        # Creating NPI Model
        npi_model = NPINetwork(input_activs_shape, input_targ_shape).float()
        if self.npi_model_path is not None:
            npi_model.load_state_dict(torch.load(self.npi_model_path, map_location="cpu"))
            npi_model.eval()
        npi_model.cuda()

        # Creating ContentClassifier Model
        content_class_model = Classifier()
        if self.content_classifier_path is not None:
            print("LOADING PRE-TRAINED CONTENT CLASSIFIER NETWORK")
            content_class_model.load_state_dict(torch.load(self.content_classifier_path, map_location=torch.device('cpu')))
            content_class_model.eval()
        else:
            raise NotImplementedError("Classifier should be pretrained. Pass in the path to the classifer.")
        content_class_model.cuda()

        # Creating GenerationClassifier Model
        generate_class_model = GenerationClassifier(input_activs_shape, input_targ_shape).float()
        if self.generation_classifier_path is not None:
            generate_class_model.load_state_dict(torch.load(self.generation_classifier_path, map_location="cpu"))
            generate_class_model.eval()

        generate_class_model.cuda()

        return npi_model, content_class_model, generate_class_model