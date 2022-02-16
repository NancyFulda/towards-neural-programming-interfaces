from copy import deepcopy
import gc
import time
import numpy as np
import pickle as pkl

import torch
import torch.optim as optim


from tqdm import tqdm
from npi.dataset.npi_dataset import NPIDataLoader

from npi.config import NPIConfig
from npi.models.training_models import NPITrainingModels
from npi.training.train_npi import NPILoss, load_models

# first helper fcns


def my_accuracy(x, y):
    """
    Accepts vector of ground truth labels and vector of generated labels
    Order not important, as long as dims are equal
        x, y are both 1-dim torch.Tensor objects or np.ndarray
    """
    x, y = x.squeeze().data.cpu().numpy(), y.squeeze().data.cpu().numpy()
    x = np.array([round(xi) for xi in x])
    y = np.array([round(yi) for yi in y])
    if len(x) != 0:
        return len(x[x == y]) / len(x)
    else:
        return 0.0


# TODO: Make generator optional
class NPITrainer:
    def __init__(
        self,
        config: NPIConfig,
        save_freq=10,
        test_freq=5,
        batch_size=5,
        headstart=5,
        loss_boosting_coeff=10000.0,
        discrim_coeff=3.0,
        style_coeff=10.0,
        similarity_coeff=1.0,
        npi_lr=1e-6,
        disc_lr=1e-6,
    ):
        # Set hyperparameters
        self.config = config
        self.save_freq = save_freq
        self.test_freq = test_freq
        self.batch_size = batch_size
        self.headstart = headstart
        self.loss_boosting_coeff = loss_boosting_coeff
        self.discrim_coeff = discrim_coeff
        self.style_coeff = style_coeff
        self.similarity_coeff = similarity_coeff
        self.npi_lr = npi_lr
        self.disc_lr = disc_lr

        # Initialize loss functions
        self.npi_objective = NPILoss(discrim_coeff, style_coeff, similarity_coeff)
        self.generate_class_objective = torch.nn.BCELoss()
        self.bce_loss = torch.nn.BCELoss()
        self.mse_loss = torch.nn.MSELoss()
        self.list_average = lambda x: sum(x) / float(len(x))

        # Initialize data structures to store training results
        self.train_metadata = {
            "npi_losses": [],
            "generate_class_losses": [],
            "generate_class_accuracies": [],
        }
        self.train_batch_metadata = deepcopy(self.train_metadata)

        self.test_metadata = {
            "npi_test_losses": [],
            "content_class_test_losses": [],
            "generate_class_tests_losses": [],
            "generate_false_class_test_losses": [],
            "content_class_test_accuracies": [],
            "generate_class_test_accuracies": [],
            "generate_class_false_test_accuracies": [],
        }
        self.test_batch_metadata = deepcopy(self.test_metadata)

        # Seed torch
        torch.manual_seed(1)

    def train_generator_step(self, orig_activ, pred_gpt2_outs):
        self.generate_class_model.train()

        for p in self.npi_model.parameters():
            p.requires_grad = False
        for p in self.generate_class_model.parameters():
            p.requires_grad = True

        self.generate_class_model.zero_grad()  # generate_class_optimizer.zero_grad()

        # labels
        y_real_GPT2 = (
            torch.zeros(self.functional_batch_size).float().cuda()
        )  # 0 = real GPT2
        y_fake_GPT2 = (
            torch.ones(self.functional_batch_size).float().cuda()
        )  # 1 = fake GPT2
        # y_real_GPT2, y_fake_GPT2 = Variable(y_real_GPT2), Variable(y_fake_GPT2)

        # Now predict and get loss
        real_gen_pred = self.generate_class_model(orig_activ)
        fake_gen_pred = self.generate_class_model(pred_gpt2_outs.detach())
        # loss
        real_loss = self.generate_class_objective(
            real_gen_pred.squeeze(), y_real_GPT2.squeeze()
        )
        fake_loss = self.generate_class_objective(
            fake_gen_pred.squeeze(), y_fake_GPT2.squeeze()
        )
        g_class_loss = self.loss_boosting_coeff * (real_loss + fake_loss)
        # record and .backward()
        g_class_loss_item = g_class_loss.item()
        self.generate_class_batch_losses.append(g_class_loss_item)
        g_class_loss.backward()
        self.generate_class_optimizer.step()
        return g_class_loss_item

    def train_npi_step(self, orig_activ, data_idx):
        functional_batch_size = orig_activ.shape[0]

        # prepare the batch for model processing
        orig_activ = orig_activ.cuda(non_blocking=True).float()

        # ~~~~ TRAINING SEGMENT open ~~~~

        curr_rows = self.train_loader.get_row_data(data_idx)
        for i in range(len(curr_rows)):
            curr_rows[i] = [None] * 4 + curr_rows[i][4:]

        # Get perturbed activations that we'll need throughout training iteration
        pred_activs = self.npi_model(orig_activ)
        pred_gpt2_outs, _ = self.get_pred_gpt2_outs(curr_rows, pred_activs)

        # UPDATE NPI WEIGHTS

        self.npi_model.train()

        for p in self.npi_model.parameters():
            p.requires_grad = True
        for p in self.generate_class_model.parameters():
            p.requires_grad = False

        self.npi_model.zero_grad()  # npi_optimizer.zero_grad()

        self.npi_objective.generation_classifier_model = self.generate_class_model

        # labels
        y_word = (
            torch.ones(functional_batch_size).float().cuda()
        )  # ones here corresponds to having NO sexist slurs
        y_real_GPT2 = torch.zeros(functional_batch_size).float().cuda()

        # pred activations already calculated
        resulting_gpt2_outs = pred_gpt2_outs

        # get classifications and loss
        content_classification = self.content_class_model(resulting_gpt2_outs)
        gen_classification = self.generate_class_model(resulting_gpt2_outs)
        # loss
        discrim_loss = self.bce_loss(
            gen_classification.squeeze(), y_real_GPT2.squeeze()
        )
        style_loss = self.bce_loss(content_classification.squeeze(), y_word.squeeze())
        similarity_loss = self.mse_loss(resulting_gpt2_outs, orig_activ)
        npi_loss = self.loss_boosting_coeff * (
            self.discrim_coeff * discrim_loss
            + self.style_coeff * style_loss
            + self.similarity_coeff * similarity_loss
        )
        # now record and report state to terminal and then .backward()
        self.npi_batch_losses.append(npi_loss.item())

        npi_loss.backward()
        self.npi_optimizer.step()

        return npi_loss

    def get_pred_gpt2_outs(self, curr_rows, pred_activs):
        return self.gpt2_with_npi.obtain_perturbed_GPT2WithNPI_outputs(
            pred_activs,
            self.config.perturbation_indices,
            curr_rows,
            tokenizer=self.gpt2_tokenizer,
            max_seq_len=self.config.max_seq_len,
            num_seq_iters=self.config.num_seq_iters,
            device=self.config.device,
        )

    def test_npi(self, batch_size, test_loader, gpt2_with_npi):
        # print("Testing: START")
        # perform npi_model testing

        for test_batch, (test_x, test_t, test_y, test_inds) in enumerate(test_loader):

            # For testing we don't even deal with weirdly sized batches because that messes with averages
            if test_x.shape[0] != batch_size:
                continue

            # Now we know functional_batch_size == batch_size
            y_real_GPT2 = torch.zeros(batch_size).float().cuda()  # 0 = real GPT2
            y_fake_GPT2 = torch.ones(batch_size).float().cuda()  # 1 = fake GPT2
            y_word = torch.ones(batch_size).float().cuda()

            test_x, test_t, test_y = (
                test_x.cuda(non_blocking=True).float(),
                test_t.cuda(non_blocking=True).float(),
                test_y.cuda(non_blocking=True).float(),
            )

            curr_rows = test_loader.get_row_data(test_inds)
            for i in range(len(curr_rows)):
                curr_rows[i] = [None] * 4 + curr_rows[i][4:]

            test_deltas = self.npi_model(test_x)
            test_gpt2_outs, test_text = self.get_pred_gpt2_outs(
                curr_rows, test_deltas, gpt2_with_npi
            )

            self.generate_class_model.eval()
            test_real_gen_pred = self.generate_class_model(test_x)
            test_fake_gen_pred = self.generate_class_model(test_gpt2_outs)
            test_real_gen_loss = self.generate_class_objective(
                test_real_gen_pred.squeeze(), y_real_GPT2.squeeze()
            )
            test_fake_gen_loss = self.generate_class_objective(
                test_fake_gen_pred.squeeze(), y_fake_GPT2.squeeze()
            )
            test_g_class_loss = self.loss_boosting_coeff * (
                test_real_gen_loss + test_fake_gen_loss
            )

            # append losses and get accuracy
            self.test_batch_metadata["generate_class_tests_losses"].append(
                test_g_class_loss.item()
            )  # note this is the sum of real and fake loss
            self.test_batch_metadata["generate_false_class_test_losses"].append(
                test_fake_gen_loss.item()
            )
            test_real_gen_acc = my_accuracy(
                test_real_gen_pred.squeeze(), y_real_GPT2.squeeze()
            )
            test_fake_gen_acc = my_accuracy(
                test_fake_gen_pred.squeeze(), y_fake_GPT2.squeeze()
            )
            test_avg_gen_acc = (test_real_gen_acc + test_fake_gen_acc) / 2.0
            self.test_batch_metadata["generate_class_test_accuracies"].append(
                test_avg_gen_acc
            )
            self.test_batch_metadata["generate_class_false_test_accuracies"].append(
                test_fake_gen_acc
            )

            self.npi_model.eval()
            test_content_classification = self.content_class_model(test_gpt2_outs)
            test_gen_classification = test_fake_gen_pred
            test_discrim_loss = self.bce_loss(
                test_gen_classification.squeeze(), y_real_GPT2.squeeze()
            )
            test_style_loss = self.bce_loss(
                test_content_classification.squeeze(), y_word.squeeze()
            )
            test_similarity_loss = self.mse_loss(test_gpt2_outs, test_x)
            test_npi_loss = self.loss_boosting_coeff * (
                self.discrim_coeff * test_discrim_loss
                + self.style_coeff * test_style_loss
                + self.similarity_coeff * test_similarity_loss
            )
            # append losses and get accuracy
            self.test_batch_metadata["npi_test_losses"].append(test_npi_loss.item())
            # Don't forget the accuracy number from the classifier
            acc_from_content_class = my_accuracy(
                test_content_classification.squeeze(), y_word.squeeze()
            )
            self.test_batch_metadata["content_class_test_accuracies"].append(
                acc_from_content_class
            )

    def visualize_training(self):
        print("Saving Data Visualizations: START")

        print("obtaining NPI visualizations")
        (
            npi_avg_epoch_train_losses,
            npi_avg_epoch_test_losses,
            npi_test_epochs,
        ) = make_npi_plots(epoch, save_file_path, npi_epoch_losses, npi_test_losses)

        with open(
            save_file_path + "{}_averages_for_visualization_plots.pkl".format("NPI"),
            "wb",
        ) as outfile:
            pkl.dump(
                {
                    "avg_epoch_train_losses": npi_avg_epoch_train_losses,
                    "avg_epoch_test_losses": npi_avg_epoch_test_losses,
                    "test_epochs": npi_test_epochs,
                },
                outfile,
            )

        print("obtaining ContentClassifier visualizations")
        (
            content_class_avg_epoch_train_losses,
            content_class_avg_epoch_test_losses,
            content_class_avg_epoch_false_test_losses,
            content_class_avg_epoch_train_accuracies,
            content_class_avg_epoch_test_accuracies,
            content_class_avg_epoch_false_test_accuracies,
            content_test_epochs,
        ) = make_classifier_plots(
            "ContentClassifier",
            epoch,
            save_file_path,
            None,
            content_false_class_tests,
            content_class_tests,
            None,
            content_class_false_test_accuracies,
            content_class_test_accuracies,
        )

        with open(
            save_file_path
            + "{}_averages_for_visualization_plots.pkl".format("ContentClassifier"),
            "wb",
        ) as outfile:
            pkl.dump(
                {
                    "avg_epoch_train_losses": content_class_avg_epoch_train_losses,
                    "avg_epoch_test_losses": content_class_avg_epoch_test_losses,
                    "avg_epoch_false_test_losses": content_class_avg_epoch_false_test_losses,
                    "avg_epoch_train_accuracies": content_class_avg_epoch_train_accuracies,
                    "avg_epoch_test_accuracies": content_class_avg_epoch_test_accuracies,
                    "avg_epoch_false_test_accuracies": content_class_avg_epoch_false_test_accuracies,
                    "test_epochs": content_test_epochs,
                },
                outfile,
            )

        print("obtaining GenerationClassifier visualizations")
        (
            gen_class_avg_epoch_train_losses,
            gen_class_avg_epoch_test_losses,
            gen_class_avg_epoch_false_test_losses,
            gen_class_avg_epoch_train_accuracies,
            gen_class_avg_epoch_test_accuracies,
            gen_class_avg_epoch_false_test_accuracies,
            gen_test_epochs,
        ) = make_classifier_plots(
            "GenerationClassifier",
            epoch,
            save_file_path,
            generate_class_epoch_losses,
            generate_false_class_tests,
            generate_class_tests,
            generate_class_train_accuracies,
            generate_class_false_test_accuracies,
            generate_class_test_accuracies,
        )

        with open(
            save_file_path
            + "{}_averages_for_visualization_plots.pkl".format("GenerationClassifier"),
            "wb",
        ) as outfile:
            pkl.dump(
                {
                    "avg_epoch_train_losses": gen_class_avg_epoch_train_losses,
                    "avg_epoch_test_losses": gen_class_avg_epoch_test_losses,
                    "avg_epoch_false_test_losses": gen_class_avg_epoch_false_test_losses,
                    "avg_epoch_train_accuracies": gen_class_avg_epoch_train_accuracies,
                    "avg_epoch_test_accuracies": gen_class_avg_epoch_test_accuracies,
                    "avg_epoch_false_test_accuracies": gen_class_avg_epoch_false_test_accuracies,
                    "test_epochs": gen_test_epochs,
                },
                outfile,
            )

        print("Saving Data Visualizations: STOP")

    def train_adversarial_npi(
        self, npi_training_models: NPITrainingModels, num_epochs, train_data, test_data
    ):
        # Initialize model
        (
            self.npi_model,
            self.generate_class_model,
            self.content_class_model,
        ) = npi_training_models.load_training_models()

        self.gpt2_with_npi, self.gpt2_tokenizer = npi_training_models.load_gpt2()

        # Initialize Data to train
        self.train_loader = NPIDataLoader(
            train_data, batch_size=self.batch_size, pin_memory=True
        )
        self.test_loader = NPIDataLoader(
            test_data, batch_size=self.batch_size, pin_memory=True
        )

        # Initialize optimizer
        self.npi_optimizer = optim.Adam(self.npi_model.parameters(), lr=self.npi_lr)
        self.generate_class_optimizer = optim.Adam(
            self.generate_class_model.parameters(), lr=self.disc_lr
        )
        self.npi_objective.content_classifier_model = self.content_class_model
        self.npi_objective.generation_classifier_model = self.generate_class_model

        print("Training")

        for epoch in range(num_epochs):
            gc.collect()
            print("############ Epoch == ", epoch, " ############")

            npi_batch_losses = []
            generate_class_batch_losses = []
            generate_class_train_batch_accuracies = []

            # Looping through training batches
            loop = tqdm(total=len(self.train_loader), position=0, leave=False)
            for batch, (orig_activ, real_label, target_label, data_idx) in enumerate(
                self.train_loader
            ):
                self.functional_batch_size = orig_activ.shape[0]
                print(
                    f"Debug: See if batch size needs to be set. configured batch size:{self.batch_size}. functional batch size:{self.functional_batch_size}"
                )

                if epoch >= self.headstart:
                    g_class_loss_item = self.train_generator_step(orig_activ)

                npi_loss = self.train_npi_step(orig_activ, data_idx)

                # will be None if we are still in the headstart
                if g_class_loss_item is not None:
                    loop.set_description(
                        f"epoch:{epoch}, gen_class_loss:{g_class_loss_item:.2f}, npi_loss:{npi_loss:.2f}"
                    )
                else:
                    loop.set_description(
                        f"epoch:{epoch}, gen_class_loss:N/A, npi_loss:{npi_loss.item():.2f}"
                    )
                    # This marks the end of looping through the training batches! :D

            # collect more averages for meta data
            for key, value in self.train_metadata.items():
                if self.train_batch_metadata[key]:
                    value.append(
                        (epoch, self.list_average(self.train_batch_metadata[key]))
                    )

            # TESTING

            # and epoch >= 1: # AFTER TRAINING PERFORM ANY REQUIRED TESTS
            if epoch % self.test_freq == 0 and epoch >= self.headstart:
                self.test_npi()
                for key, value in self.test_metadata.items():
                    # Testing: Storing loss avgs
                    if self.test_batch_metadata[key]:
                        value.append(
                            (epoch, self.list_average(self.test_batch_metadata[key]))
                        )
                out_path = f"{self.config.save_folder}{self.config.npi_type}_npi_train_summaries_epoch{epoch}.pkl"
                with open(out_path, "wb") as outfile:
                    pkl.dump(self.train_metadata, outfile)
                out_path = f"{self.config.save_folder}npi_test_summaries_epoch{epoch}.pkl"
                with open(out_path, "wb") as outfile:
                    pkl.dump(self.test_metadata, outfile)

            # report current state to terminal
            torch.cuda.empty_cache()
            loop.update(1)

        print("end of regular epoch")

        if epoch % self.save_freq == 0 and epoch >= self.headstart:
            self.save_models()
            self.visualize_training()
        torch.cuda.empty_cache()
        loop.close()
