import numpy as np
import os
import sys
from time import time

from datetime import datetime
from tensorboardX import SummaryWriter

from .average_meter import AverageMeter


class TrainLogger(object):
    def __init__(self, save_dir, name, num_epochs, iters_per_print):
        self.save_dir = save_dir
        self.name = name
        self.num_epochs = num_epochs
        self.iters_per_print = iters_per_print
        self.log_path = os.path.join(self.save_dir, '{}.log'.format(self.name))
        log_dir = os.path.join('logs', name + '_' + datetime.now().strftime('%b%d_%H%M'))
        self.summary_writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.global_step = 0

        self.metric_logs = {}
        self.metric_logs['loss'] = []
        self.metric_logs['accuracy'] = []

        self.best_loss = sys.maxsize
        self.loss_meter = AverageMeter()

        self.notImprovedCounter = 0


    def _log_scalars(self, scalar_dict, print_to_stdout=True):
        """Log all values in a dict as scalars to TensorBoard."""
        for k, v in scalar_dict.items():
            if print_to_stdout:
                self.write('[{}: {:.3g}]'.format(k, v))
            k = k.replace('_', '/')  # Group in TensorBoard by phase
            self.summary_writer.add_scalar(k, v, self.global_step) 

            self.metric_logs[k].append(v)

        
    def write(self, message, print_to_stdout=True):
        """Write a message to the log. If print_to_stdout is True, also print to stdout."""
        with open(self.log_path, 'a') as log_file:
            log_file.write(message + '\n')
        if print_to_stdout:
            print(message)


    def start_iter(self):
        """Log info for start of an iteration."""
        self.iter_start_time = time()
        self.loss_meter.reset()


    def log_iter(self, metrics):
        """Log results from a training iteration"""
        if self.iter % self.iters_per_print == 0:

            avg_time = time() - self.iter_start_time
            message = '[epoch: {}, iter: {}, time: {:.2f}, loss: {:.3g}, accuracy {:3g}]' \
                .format(self.epoch, self.iter, avg_time, 
                        metrics['loss'],
                        metrics['accuracy'])

            self.write(message)

        self._log_scalars(metrics, False)

        self.loss_meter.update(metrics['loss'])


    def end_iter(self):
        """Log info for end of an iteration."""
        self.iter += 1
        self.global_step += 1


    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0
        self.write('[start of epoch {}]'.format(self.epoch))


    def end_epoch(self, metrics):
        """Log info for end of an epoch.
        Args:
            metrics: Dictionary of metric values. Items have format '{phase}_{metric}': value.
            optimizer: Optimizer for the model.
        """
        #TODO: record lr
        self.write('[end of epoch {}, epoch time: {:.2g}, average epoch loss: {:.3g}, best epoch loss: {:.3g}, not Improved for {} epochs]'
                   .format(self.epoch, time() - self.epoch_start_time, self.loss_meter.avg, self.best_loss, self.notImprovedCounter))
        if metrics is not None:
            self._log_scalars(metrics)

        self.epoch += 1

    
    def has_improved(self):
        """Reports whether this epochs loss has improved since the last"""
        last_epoch_loss = self.loss_meter.avg
        isBetter = last_epoch_loss < self.best_loss
        if isBetter:
            self.best_loss = last_epoch_loss
            self.notImprovedCounter = 0
        else:
            self.notImprovedCounter += 1

        return last_epoch_loss

    def is_finished_training(self):
        """Return True if finished training, otherwise return False."""
        return 0 < self.num_epochs < self.epoch
