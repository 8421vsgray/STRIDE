import random
import torch
import torch
import math
from torch.backends import cudnn
import numpy as np
import datetime
import time
from collections import defaultdict, deque


class InfoBuffer:
    def __init__(self, args):
        self.n_samples = args.n_sample
        self.n_views = args.n_views
        self.n_classes = args.n_classes
        self.bs = args.batch_size
        self.mmt = 0.97
        self.emp_dist = torch.full((args.n_views,args.n_sample), 1.0)

    def init_Buffer(self,emp_dist_train,i):
        self.emp_dist[i] = torch.full((self.n_samples,) ,emp_dist_train.mean())

    def update_Buffer(self, emp_dist_train, i):
        self.emp_dist[i] = emp_dist_train * (1 - self.mmt) + self.emp_dist[i] * self.mmt

    def get_Buffer(self):
        return self.emp_dist.clone().detach()


def fix_random_seeds(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True
        print("\nenable cudnn.deterministic, seed fixed: {}".format(seed))
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True


def adjust_learning_config(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


class FileLogger:
    def __init__(self, output_file):
        self.output_file = output_file

    def write(self, msg, p=True):
        with open(self.output_file, mode="a", encoding="utf-8") as log_file:
            log_file.writelines(msg + "\n")
        if p:
            print(msg)

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )

class Ratioschedule:
    def __init__(self,warmup_epochs,max_epoch, initial, final, schedule_type='cosine', **kwargs):
        self.warmup_epochs = warmup_epochs
        self.max_epoch = max_epoch
        self.initial = initial
        self.gap = initial - final
        self.sc = schedule_type
        if schedule_type == 'step':
            self.drop_every = kwargs.get('drop_every', 20)
            self.drop_rate = kwargs.get('drop_rate', 0.8)
        elif schedule_type == 'exp':
            self.gamma = kwargs.get('gamma', 0.98)
        elif schedule_type != 'cosine':
            raise 'unknown schedule_type'

    def __call__(self, optimizer,epoch):
        if epoch < self.warmup_epochs:
            ratio = self.initial * epoch / self.warmup_epochs
        else:
            epoch = epoch - self.warmup_epochs
            if self.sc == 'step':
                ratio = self.step_decay(epoch)
            elif self.sc == 'exp':
                ratio = self.exp_decay(epoch)
            else:
                ratio = self.cosine_decay(epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = ratio

        return ratio

    def step_decay(self,epoch):
        return self.initial - (self.drop_rate ** (epoch // self.drop_every))*self.gap

    def exp_decay(self,epoch):
        return self.initial - (1 - self.gamma ** epoch) * self.gap

    def cosine_decay(self,epoch):
        ratio = 0.5 * (1 + math.cos(math.pi * epoch / self.max_epoch))
        return self.initial +  self.gap * (1 - ratio)



