import gc
import time
import numpy as np
import torch

from tqdm import tqdm
from npi.dataset.npi_dataset import NPIDataLoader

from npi.training.config import NPIConfig
from npi.training.train_npi import load_models

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


class NPITrainer:
    def __init__(
        self,
        config: NPIConfig,
        save_freq=10,
        test_freq=5,
        batch_size=5,
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
        self.loss_boosting_coeff = loss_boosting_coeff
        self.discrim_coeff = discrim_coeff
        self.style_coeff = style_coeff
        self.similarity_coeff = similarity_coeff
        self.npi_lr = npi_lr
        self.disc_lr = disc_lr

        # Initialize model
        self.npi_model = load_models()
        self.content_class_model = None
        self.generate_class_model = None
        self.gpt2_with_npi = None
        self.gpt2_tokenizer = None

        # Initialize loss functions
        self.npi_objective = None
        self.generate_class_objective = None
        self.bce_loss = None
        self.mse_loss = None

        # Initialize optimizer
        self.npi_optimizer = None
        self.generate_class_optimizer = None

        # Initialize Data to train
        # TODO: remove data initialization from this script. Set when starting training.
        self.train_loader = None
        self.functional_batch_size = None

        # Initialize data structure to store training results
        self.npi_batch_losses = []
        self.generate_class_batch_losses = []

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
        npi_test_batch_losses = []
        content_class_test_losses = []
        content_false_class_test_losses = []
        generation_class_test_losses = []
        generation_false_class_test_losses = []

        content_class_test_batch_accuracies = []
        generate_class_test_batch_accuracies = []
        content_class_false_test_batch_accuracies = []
        generate_class_false_test_batch_accuracies = []

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
            generation_class_test_losses.append(
                test_g_class_loss.item()
            )  # note this is the sum of real and fake loss
            generation_false_class_test_losses.append(test_fake_gen_loss.item())
            test_real_gen_acc = my_accuracy(
                test_real_gen_pred.squeeze(), y_real_GPT2.squeeze()
            )
            test_fake_gen_acc = my_accuracy(
                test_fake_gen_pred.squeeze(), y_fake_GPT2.squeeze()
            )
            test_avg_gen_acc = (test_real_gen_acc + test_fake_gen_acc) / 2.0
            generate_class_test_batch_accuracies.append(test_avg_gen_acc)
            generate_class_false_test_batch_accuracies.append(test_fake_gen_acc)

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
            npi_test_batch_losses.append(test_npi_loss.item())
            # Don't forget the accuracy number from the classifier
            acc_from_content_class = my_accuracy(
                test_content_classification.squeeze(), y_word.squeeze()
            )
            content_class_false_test_batch_accuracies.append(acc_from_content_class)

            if file_num == len(train_file_names) - 1 and test_batch == 0:
                class_sample_meta_data["testing data"]["epoch {}".format(epoch)] = {
                    "real array classifications": test_real_gen_pred.squeeze()
                    .data.cpu()
                    .numpy(),
                    "NPI-produced array classifications": test_fake_gen_pred.squeeze()
                    .data.cpu()
                    .numpy(),
                    "testing loss": test_g_class_loss.cpu().item(),
                    "testing accuracy": test_avg_gen_acc,
                }
                npi_sample_meta_data["testing data"]["epoch {}".format(epoch)] = {
                    "style loss": test_style_loss.cpu().item(),
                    "similarity loss": test_similarity_loss.cpu().item(),
                    "discrim loss": test_discrim_loss.cpu().item(),
                    "content classifier classifications": test_content_classification.squeeze()
                    .data.cpu()
                    .numpy(),
                    "text samples": test_text,
                }

        # Testing: STOP

    def save_models(self):
        print("Saving NPI Model")
        out_path = save_file_path + "{}_npi_network_epoch{}.bin".format(npi_type, epoch)
        torch.save(npi_model.state_dict(), out_path)

        print("Saving NPI Loss Summary")
        out_path = save_file_path + "{}_npi_loss_summaries_epoch{}.pkl".format(
            npi_type, epoch
        )
        with open(out_path, "wb") as outfile:
            pkl.dump(
                {
                    "epoch_losses": npi_epoch_losses,
                    "test_losses": npi_test_losses,
                    "accuracies_from_content_class": content_class_false_test_accuracies,
                    "sample_meta_data": npi_sample_meta_data,
                },
                outfile,
            )

        # print("Saving ContentClassifier Loss Summary")
        # out_path = save_file_path + "{}_loss_summaries_epoch{}.pkl".format("ContentClassifier", epoch)
        # with open(out_path, 'wb') as outfile:
        #    pkl.dump({"false_test_losses": content_false_class_tests,
        #                "avg_test_losses": content_class_tests,
        #                "false_test_accuracies": content_class_false_test_accuracies,
        #                "avg_test_accuracies": content_class_test_accuracies,
        #             }, outfile)

        print("Saving GenerationClassifier Model")
        out_path = save_file_path + "{}_network_epoch{}.bin".format(
            "GenerationClassifier", epoch
        )
        torch.save(generate_class_model.state_dict(), out_path)

        print("Saving GenerationClassifier Loss Summary")
        out_path = None
        out_path = save_file_path + "{}_loss_summaries_epoch{}.pkl".format(
            "GenerationClassifier", epoch
        )
        with open(out_path, "wb") as outfile:
            pkl.dump(
                {
                    "epoch_losses": generate_class_epoch_losses,
                    "false_tests": generate_false_class_tests,
                    "avg_tests": generate_class_tests,
                    "training_accuracies": generate_class_train_accuracies,
                    "false_test_accuracies": generate_class_false_test_accuracies,
                    "avg_test_accuracies": generate_class_test_accuracies,
                    "sample_meta_data": class_sample_meta_data,
                },
                outfile,
            )

        print("Done saving for current epoch")

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

    def train_adversarial_npi(self, num_epochs, train_loader, test_loader):
        print("Training")

        for epoch in range(num_epochs):
            gc.collect()
            print("############ Epoch == ", epoch, " ############")

            npi_batch_losses = []
            generate_class_batch_losses = []
            generate_class_train_batch_accuracies = []

            # Looping through training batches
            loop = tqdm(total=len(train_loader), position=0, leave=False)
            for batch, (orig_activ, real_label, target_label, data_idx) in enumerate(
                train_loader
            ):

                if epoch >= self.headstart:
                    g_class_loss_item = self.train_generator_step(orig_activ)

                npi_loss = self.train_npi_step(orig_activ, data_idx)

                if (
                    g_class_loss_item is not None
                ):  # will be None if we are still in the headstart
                    loop.set_description(
                        f"epoch:{epoch}, gen_class_loss:{g_class_loss_item:.2f}, npi_loss:{npi_loss:.2f}"
                    )
                else:
                    loop.set_description(
                        f"epoch:{epoch}, gen_class_loss:N/A, npi_loss:{npi_loss.item():.2f}"
                    )

                # save meta data
                if (
                    (epoch % save_freq == 0)
                    and file_num == len(train_file_names) - 1
                    and batch == 0
                    and epoch >= HEAD_START_NUM
                ):
                    class_sample_meta_data["training data"][
                        "epoch {}".format(epoch)
                    ] = {
                        "real array classifications": real_gen_pred.squeeze()
                        .data.cpu()
                        .numpy(),
                        "NPI-produced array classifications": fake_gen_pred.squeeze()
                        .data.cpu()
                        .numpy(),
                        "training loss": g_class_loss.cpu().item(),
                    }
                    npi_sample_meta_data["training data"]["epoch {}".format(epoch)] = {
                        "style loss": style_loss.cpu().item(),
                        "similarity loss": similarity_loss.cpu().item(),
                        "discrim loss": discrim_loss.cpu().item(),
                        "content classifier classifications": content_classification.squeeze()
                        .data.cpu()
                        .numpy(),
                        "test_samples": training_text,
                    }

                    # This marks the end of looping through the training batches! :D

            # collect more averages for meta data
            if npi_batch_losses:
                npi_epoch_losses.append(
                    (sum(npi_batch_losses) / float(len(npi_batch_losses)))
                )

            if generate_class_batch_losses:
                generate_class_epoch_losses.append(
                    (
                        sum(generate_class_batch_losses)
                        / float(len(generate_class_batch_losses))
                    )
                )

            if (
                epoch % test_freq == 0
                and generate_class_train_batch_accuracies
                and epoch >= HEAD_START_NUM
            ):
                generate_class_train_accuracies.append(
                    (
                        epoch,
                        (
                            sum(generate_class_train_batch_accuracies)
                            / float(len(generate_class_train_batch_accuracies))
                        ),
                    )
                )

            # TESTING

            # and epoch >= 1: # AFTER TRAINING PERFORM ANY REQUIRED TESTS
            if epoch % test_freq == 0 and epoch >= HEAD_START_NUM:
                self.test_npi()
                # Testing: Storing loss avgs
                if npi_test_batch_losses:
                    npi_test_losses.append(
                        (
                            epoch,
                            (
                                sum(npi_test_batch_losses)
                                / float(len(npi_test_batch_losses))
                            ),
                        )
                    )
                if content_class_test_losses:
                    content_class_tests.append(
                        (
                            epoch,
                            (
                                sum(content_class_test_losses)
                                / float(len(content_class_test_losses))
                            ),
                        )
                    )
                if content_false_class_test_losses:
                    content_false_class_tests.append(
                        (
                            epoch,
                            (
                                sum(content_false_class_test_losses)
                                / float(len(content_false_class_test_losses))
                            ),
                        )
                    )
                if generation_class_test_losses:
                    generate_class_tests.append(
                        (
                            epoch,
                            (
                                sum(generation_class_test_losses)
                                / float(len(generation_class_test_losses))
                            ),
                        )
                    )
                if generation_false_class_test_losses:
                    generate_false_class_tests.append(
                        (
                            epoch,
                            (
                                sum(generation_false_class_test_losses)
                                / float(len(generation_false_class_test_losses))
                            ),
                        )
                    )

                # Testing: Storing accuracy avgs
                if content_class_test_batch_accuracies:
                    content_class_test_accuracies.append(
                        (
                            epoch,
                            (
                                sum(content_class_test_batch_accuracies)
                                / float(len(content_class_test_batch_accuracies))
                            ),
                        )
                    )
                if generate_class_test_batch_accuracies:
                    generate_class_test_accuracies.append(
                        (
                            epoch,
                            (
                                sum(generate_class_test_batch_accuracies)
                                / float(len(generate_class_test_batch_accuracies))
                            ),
                        )
                    )
                if content_class_false_test_batch_accuracies:
                    content_class_false_test_accuracies.append(
                        (
                            epoch,
                            (
                                sum(content_class_false_test_batch_accuracies)
                                / float(len(content_class_false_test_batch_accuracies))
                            ),
                        )
                    )
                if generate_class_false_test_batch_accuracies:
                    generate_class_false_test_accuracies.append(
                        (
                            epoch,
                            (
                                sum(generate_class_false_test_batch_accuracies)
                                / float(len(generate_class_false_test_batch_accuracies))
                            ),
                        )
                    )

            # report current state to terminal
            torch.cuda.empty_cache()
            if g_class_loss_item is not None:
                loop.set_description(
                    "epoch:{}, gen_class_loss:{:.2f}, npi_loss:{:.2f}, time_elapsed:{:.1f}".format(
                        epoch,
                        g_class_loss_item,
                        npi_loss.item(),
                        (time.time() - start_time),
                    )
                )
            else:
                loop.set_description(
                    "epoch:{}, gen_class_loss:N/A, npi_loss:{:.2f}, time_elapsed:{:.1f}".format(
                        epoch, npi_loss.item(), (time.time() - start_time)
                    )
                )
            loop.update(1)

        print("end of regular epoch")

        if epoch % self.save_freq == 0 and epoch >= self.headstart:
            # save the current version of the npi_model
            self.save_models()

            # ~~~~~~NOW for the visualizations~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.visualize_training()
        torch.cuda.empty_cache()
        loop.close()
