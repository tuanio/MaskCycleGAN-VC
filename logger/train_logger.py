"""
Logs training metrics and metadata to visualize on tensorboard.
Inherits BaseLogger class.
"""

from time import time
from logger.base_logger import BaseLogger
import torchvision.utils as vutils
import wandb

""" Logs metrics and metadata to visualize on tensorboard """

import os
from datetime import datetime
from tensorboardX import SummaryWriter


class BaseLogger(object):
    """
    Class which implements a Logger.
    Attributes:
        args : Namespace Program arguments
        batch_size (int): Batch size
        dataset_len (int): Number of samples in the dataset
        save_dir (str): Save directory path
        summary_writer (tensorboardX.SummaryWriter): Writes entires to event files in args.log_dir
        log_path (str): Path to .log file <- not the same as the event file
        epoch (int): Current epoch
        iter (int): Current iteration within epoch
        global_step (int): Current iteration overall
    Methods:
        _log_text(text_dict): Log all strings in a dict as scalars to TensorBoard
        _log_scalars(scalar_dict, print_to_stdout): Log all values in a dict as scalars to TensorBoard
        write(message, print_to_stdout): Write a message to the .log file. If print_to_stdout is True, also print to stdout
        start_iter(): Log info for start of an iteration
        end_iter(): Log info for end of an iteration
        start_epoch(): Log info for start of an epoch
        end_epoch(metrics): Log info for end of an epoch. Save model parameters and update learning rate
    """

    def __init__(self, args, dataset_len):
        """
        Args:
            args (Namespace): Program arguments
            dataset_len (int): Number of samples in dataset
        """

        def round_down(x, m):
            """Round x down to a multiple of m."""
            return int(m * round(float(x) / m))

        self.args = args
        self.batch_size = args.batch_size
        self.dataset_len = dataset_len
        self.save_dir = args.save_dir

        # log_dir, is the directory for tensorboard logs: hpylori/logs/
        log_dir = os.path.join(
            args.save_dir,
            "logs",
            args.name + "_" + datetime.now().strftime("%y%m%d_%H%M%S"),
        )
        self.summary_writer = SummaryWriter(log_dir=log_dir)
        self.log_path = os.path.join(
            self.save_dir, args.name, "{}.log".format(args.name)
        )
        self.epoch = args.start_epoch
        self.iter = 0
        self.global_step = round_down((self.epoch - 1) * dataset_len, args.batch_size)

    def log_text(self, text_dict):
        """
        Log all strings in a dict as scalars to TensorBoard.
        Args:
            text_dict (dict): str to str dictionary
        """
        for k, v in text_dict.items():
            self.summary_writer.add_text(k, str(v), self.global_step)

    def _log_scalars(self, scalar_dict, print_to_stdout=True):
        """
        Log all values in a dict as scalars to TensorBoard.
        Args:
            scalar_dict (dict): str to scalar dictionary
            print_to_stdout (bool): If True, print scalars to stdout
        """
        for k, v in scalar_dict.items():
            if print_to_stdout:
                self.write("[{}: {}]".format(k, v))
            # Group in TensorBoard by split. eg. (D_A, D_B) are grouped, (G_A, G_B) are grouped
            k = k.replace("_", "/")
            self.summary_writer.add_scalar(k, v, self.global_step)

    def write(self, message, print_to_stdout=True):
        """
        Write a message to the .log file. If print_to_stdout is True, also print to stdout.
        Args:
            message (str): Message to write to .log (and stdout if applicable)
            print_to_stdout (bool): If True, print message to stdout
        """
        with open(self.log_path, "a") as log_file:
            log_file.write(message + "\n")
        if print_to_stdout:
            print(message)

    def start_iter(self):
        """Log info for start of an iteration."""
        raise NotImplementedError

    def end_iter(self):
        """Log info for end of an iteration."""
        raise NotImplementedError

    def start_epoch(self):
        """Log info for start of an epoch."""
        raise NotImplementedError

    def end_epoch(self, metrics):
        """
        Log info for end of an epoch. Save model parameters and update learning rate.
        Args:
        metrics (dict): str to scalar dictionary containing metrics such as losses to log
        """
        raise NotImplementedError


def visualize(images, save_path=None):
    """
    Visualize a grid of outputs and save to path if paths is provided.
    Otherwise returns the grid.
    Args:
        images (list(torch.Tensor)): list of tensors
        save_path (bool): path to save image to
    Returns:
        torch.Tensor: all images as grid
    """
    gridded_images = vutils.make_grid(images, padding=2, normalize=True)
    if save_path:
        vutils.save_image(gridded_images, save_path)
        print(f"Saved images to {save_path}")

    return gridded_images


class AverageMeter(object):
    """
    Class which computes and stores the average and current value.
    Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Attributes
    ----------
    avg : float
        Average value
    val : float
        Current value
    sum : float
        Sum of values
    count : float
        Number of values
    Methods
    -------
    reset():
        Reinitializes the class object
    update(val, n):
        Updates attributes with new value
    """

    def __init__(self):
        self.avg = 0
        self.val = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reinitializes the class object."""
        self.__init__()

    def update(self, val, n=1):
        """
        Updates attributes with new value.
        Parameters
        ----------
        val : float
            New value
        n : int
            Number of values to update
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TrainLogger(BaseLogger):
    """
    Class for logging training info to the console and saving model parameters to disk.
    In inherits from BaseLogger.
    Attruibutes:
        iter_start_time (time): Records the start time of each iteration
        epoch_start_time (time): Records the start time of each epoch
        steps_per_print (int): Number of iterations between metrics are logged and printed
        num_epochs (int): Total number of epochs to train
    Methods:
        log_hparams(args): Log all the hyper parameters in tensorboard
        log_iter(loss_dict={}): Log results from a training iteration
        log_metrics(metrics): Log scalar metrics from training
        start_iter(): Log info for start of an iteration
        end_iter(): Log info for end of an iteration
        start_epoch(): Log info for start of an epoch
        end_epoch(metrics): Log info for end of an epoch. Save model parameters and update learning rate.
        is_finished_training(): Return True if finished training, otherwise return False.
    """

    def __init__(self, args, dataset_len):
        super(TrainLogger, self).__init__(args, dataset_len)
        """
        Args:
            args (Namespace): Program arguments
            dataset_len (int): Number of samples in dataset
        """
        self.iter_start_time = None
        self.epoch_start_time = None
        self.steps_per_print = args.steps_per_print
        self.num_epochs = args.num_epochs

    def log_hparams(self, args):
        """
        Log all the hyper parameters in tensorboard.
        Args:
            args (Namespace): Program arguments
        """
        hparams = {}
        args_dict = vars(args)
        for key in args_dict:
            hparams.update({"hparams/" + key: args_dict[key]})

        self.log_text(hparams)

    def log_iter(self, loss_dict={}):
        """
        Log results from a training iteration.
        Args:
            loss_dict (dict): str to scalar dictionary of losses
        """
        if not hasattr(self, "loss_meters"):
            self.loss_meters = {
                loss_name: AverageMeter() for loss_name in loss_dict.keys()
            }
        for loss_name, meter in self.loss_meters.items():
            meter.update(loss_dict[loss_name], self.batch_size)

        # Periodically write to the log and TensorBoard
        if self.iter % self.steps_per_print == 0:
            # Write a header for the log entry
            avg_time = (time() - self.iter_start_time) / self.batch_size
            message = "(epoch: %d, iter: %d, time: %.3f) " % (
                self.epoch,
                self.iter,
                avg_time,
            )
            for loss_name, meter in self.loss_meters.items():
                message += "%s: %.3f " % (loss_name, meter.avg)

            # Write all errors as scalars to the graph
            self._log_scalars(
                {loss_name: meter.avg for loss_name, meter in self.loss_meters.items()},
                print_to_stdout=False,
            )

            for _, meter in self.loss_meters.items():
                meter.reset()

            # write to .log file
            self.write(message)

    def log_metrics(self, metrics):
        """
        Logs scalar metrics from training.
        Args:
            metrics (dict): str to scalar dictionary containing metrics such as losses to log
        """
        self._log_scalars(metrics)

    def log_img(self, img, name):
        self.summary_writer.add_image(name, img, self.global_step)

    def log_audio(self, audio, name, sampling_rate):
        self.summary_writer.add_audio(name, audio, self.global_step, sampling_rate)

    def start_iter(self):
        """Log info for start of an iteration."""
        self.iter_start_time = time()

    def end_iter(self):
        """Log info for end of an iteration."""
        self.iter += self.batch_size
        self.global_step += self.batch_size

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0
        self.write("[start of epoch {}]".format(self.epoch))

    def end_epoch(self, metrics=None):
        """
        Log info for end of an epoch.
        Args:
            metrics (dict): str to scalar dictionary of metric values.
        """
        self.write(
            "[end of epoch {}/{}, epoch time: {:.2g}]".format(
                self.epoch, self.num_epochs, time() - self.epoch_start_time
            )
        )
        if metrics:
            self._log_scalars(metrics)
        self.epoch += 1

    def is_finished_training(self):
        """Return True if finished training, otherwise return False."""
        return 0 < self.num_epochs < self.epoch

    def visualize_outputs(self, img_dict):
        """
        Visualize predictions and targets in TensorBoard in grid form.
        Args:
            img_dict (dict): str to Tensor dictionary of images
        Returns:
            int: Number of examples visualized to TensorBoard.
        """
        imgs = []
        names = "-".join(list(img_dict.keys()))
        for name, img in img_dict.items():
            imgs.append(img)

        self.summary_writer.add_image(names, visualize(imgs), self.global_step)

        return len(img_dict)


class WandbLogger(BaseLogger):
    def __init__(self, args, dataset_len):
        super(TrainLogger, self).__init__(args, dataset_len)
        """
        Args:
            args (Namespace): Program arguments
            dataset_len (int): Number of samples in dataset
        """
        self.iter_start_time = None
        self.epoch_start_time = None
        self.steps_per_print = args.steps_per_print
        self.num_epochs = args.num_epochs

    def log_hparams(self, args):
        """
        Log all the hyper parameters in tensorboard.
        Args:
            args (Namespace): Program arguments
        """
        hparams = {}
        args_dict = vars(args)
        for key in args_dict:
            hparams.update({"hparams/" + key: args_dict[key]})

        self.log_text(hparams)

    def log_iter(self, loss_dict={}):
        """
        Log results from a training iteration.
        Args:
            loss_dict (dict): str to scalar dictionary of losses
        """
        if not hasattr(self, "loss_meters"):
            self.loss_meters = {
                loss_name: AverageMeter() for loss_name in loss_dict.keys()
            }
        for loss_name, meter in self.loss_meters.items():
            meter.update(loss_dict[loss_name], self.batch_size)

        # Periodically write to the log and TensorBoard
        if self.iter % self.steps_per_print == 0:
            # Write a header for the log entry
            avg_time = (time() - self.iter_start_time) / self.batch_size
            message = "(epoch: %d, iter: %d, time: %.3f) " % (
                self.epoch,
                self.iter,
                avg_time,
            )
            for loss_name, meter in self.loss_meters.items():
                message += "%s: %.3f " % (loss_name, meter.avg)

            # Write all errors as scalars to the graph
            self._log_scalars(
                {loss_name: meter.avg for loss_name, meter in self.loss_meters.items()},
                print_to_stdout=False,
            )

            for _, meter in self.loss_meters.items():
                meter.reset()

            # write to .log file
            self.write(message)

    def log_metrics(self, metrics):
        """
        Logs scalar metrics from training.
        Args:
            metrics (dict): str to scalar dictionary containing metrics such as losses to log
        """
        self._log_scalars(metrics)

    def log_img(self, img, name):
        self.summary_writer.add_image(name, img, self.global_step)

    def log_audio(self, audio, name, sampling_rate):
        self.summary_writer.add_audio(name, audio, self.global_step, sampling_rate)

    def start_iter(self):
        """Log info for start of an iteration."""
        self.iter_start_time = time()

    def end_iter(self):
        """Log info for end of an iteration."""
        self.iter += self.batch_size
        self.global_step += self.batch_size

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0
        self.write("[start of epoch {}]".format(self.epoch))

    def end_epoch(self, metrics=None):
        """
        Log info for end of an epoch.
        Args:
            metrics (dict): str to scalar dictionary of metric values.
        """
        self.write(
            "[end of epoch {}/{}, epoch time: {:.2g}]".format(
                self.epoch, self.num_epochs, time() - self.epoch_start_time
            )
        )
        if metrics:
            self._log_scalars(metrics)
        self.epoch += 1

    def is_finished_training(self):
        """Return True if finished training, otherwise return False."""
        return 0 < self.num_epochs < self.epoch

    def visualize_outputs(self, img_dict):
        """
        Visualize predictions and targets in TensorBoard in grid form.
        Args:
            img_dict (dict): str to Tensor dictionary of images
        Returns:
            int: Number of examples visualized to TensorBoard.
        """
        imgs = []
        names = "-".join(list(img_dict.keys()))
        for name, img in img_dict.items():
            imgs.append(img)

        self.summary_writer.add_image(names, visualize(imgs), self.global_step)

        return len(img_dict)


class WandbLogger(BaseLogger):
    def __init__(self, args, dataset_len):
        super(WandbLogger, self).__init__(args, dataset_len)
        self.iter_start_time = None
        self.epoch_start_time = None
        self.steps_per_print = args.steps_per_print
        self.num_epochs = args.num_epochs
        wandb.init(
            project=args.name,
            name=args.name + "_" + datetime.now().strftime("%y%m%d_%H%M%S"),
        )

    def log_hparams(self, args):
        hparams = {}
        args_dict = vars(args)
        for key in args_dict:
            hparams.update({"hparams/" + key: args_dict[key]})

        wandb.config.update(hparams)

    def log_iter(self, loss_dict={}):
        if not hasattr(self, "loss_meters"):
            self.loss_meters = {
                loss_name: AverageMeter() for loss_name in loss_dict.keys()
            }
        for loss_name, meter in self.loss_meters.items():
            meter.update(loss_dict[loss_name], self.batch_size)

        # Periodically write to the log and Wandb
        if self.iter % self.steps_per_print == 0:
            # Write a header for the log entry
            avg_time = (time() - self.iter_start_time) / self.batch_size
            message = "(epoch: %d, iter: %d, time: %.3f) " % (
                self.epoch,
                self.iter,
                avg_time,
            )
            for loss_name, meter in self.loss_meters.items():
                message += "%s: %.3f " % (loss_name, meter.avg)

            # Write all errors as scalars to the graph
            wandb.log(
                {loss_name: meter.avg for loss_name, meter in self.loss_meters.items()}
            )

            for _, meter in self.loss_meters.items():
                meter.reset()

            # write to .log file
            self.write(message)

    def log_metrics(self, metrics):
        wandb.log(metrics)

    def log_img(self, img, name):
        wandb.log({name: [wandb.Image(img)]})

    def log_audio(self, audio, name, sampling_rate):
        wandb.log({name: [wandb.Audio(audio, sample_rate=sampling_rate)]})

    def start_iter(self):
        self.iter_start_time = time()

    def end_iter(self):
        self.iter += self.batch_size
        self.global_step += self.batch_size

    def start_epoch(self):
        self.epoch_start_time = time()
        self.iter = 0
        self.write("[start of epoch {}]".format(self.epoch))

    def end_epoch(self, metrics=None):
        self.write(
            "[end of epoch {}/{}, epoch time: {:.2g}]".format(
                self.epoch, self.num_epochs, time() - self.epoch_start_time
            )
        )
        if metrics:
            wandb.log(metrics)
        self.epoch += 1

    def is_finished_training(self):
        return 0 < self.num_epochs < self.epoch

    def visualize_outputs(self, img_dict):
        imgs = []
        names = "-".join(list(img_dict.keys()))
        for name, img in img_dict.items():
            imgs.append(img)

        wandb.log({names: [wandb.Image(visualize(imgs))]})

        return len(img_dict)
