import math
import torch


class Optimizer:
    def __init__(
            self,
            objective,
            x0,
            lr,
            lr_max=float("inf"),
            batch_size=512,
            seed=42,
            lr_decay_coef=0,
            it_start_decay=0,
    ):
        self.objective = objective
        self.x = x0.clone()
        self.lr = lr            # base / constant stepsize
        self.lr_max = lr_max    # ceiling stepsize used by the decaying schedule
        self.batch_size = batch_size
        self.iteration = 0
        self.epoch = 0
        self.n = objective.n
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

        # Constant normalization for permutation-based methods (RR, IG, SO):
        # dividing every batch gradient by n / steps_per_epoch instead of the
        # actual batch length keeps the per-sample weight uniform within an
        # epoch even when the last batch is incomplete. Otherwise the samples
        # of the short last batch are overweighted — for SO and IG these are
        # the SAME samples every epoch, which biases the effective objective.
        self.steps_per_epoch = math.ceil(self.n / batch_size)
        self.epoch_norm = self.n / self.steps_per_epoch

        self.lr_decay_coef = lr_decay_coef    # == c (= mu/3 or mu/2 for RR, SO, IG or SGD for strongly convex objectives respectively)

        self.it_start_decay = it_start_decay  # == k0

    def current_lr(self):
        # lr_max == inverse of smoothness constant (stepsize ceiling)
        # lr_decay_coef == c (mu/3 (RR, SO, IG) or mu/2 (SGD) for the strongly convex case)

        if self.lr_decay_coef == 0:
            decayed = self.lr          # constant stepsize (no decay)
        else:
            decayed = 1.0 / (self.lr_decay_coef * max(1, self.iteration - self.it_start_decay))
        return min(self.lr_max, decayed)


class SGD(Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self):
        idx = torch.randint(
            low=0,
            high=self.n,
            size=(self.batch_size,),
            generator=self.rng
            )

        grad = self.objective.stochastic_gradient(self.x, idx)
        lr = self.current_lr()
        self.x = self.x - lr * grad

        self.iteration += 1


class RR(Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.permutation = torch.randperm(self.n, generator=self.rng)
        self.position = 0

    def new_epoch(self):
        self.permutation = torch.randperm(self.n, generator=self.rng)
        self.position = 0

    def step(self):
        batch_end = min(self.position + self.batch_size, self.n)

        idx = self.permutation[self.position:batch_end]

        grad = self.objective.stochastic_gradient(self.x, idx, normalization=self.epoch_norm)
        lr = self.current_lr()
        self.x = self.x - lr * grad

        self.position = batch_end

        if self.position >= self.n:
            self.new_epoch()
            self.epoch += 1

        self.iteration += 1


class IG(Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.position = 0

    def step(self):
        batch_end = min(self.position + self.batch_size, self.n)

        idx = torch.arange(self.position, batch_end)

        grad = self.objective.stochastic_gradient(self.x, idx, normalization=self.epoch_norm)
        lr = self.current_lr()
        self.x = self.x - lr * grad

        self.position = batch_end

        if self.position >= self.n:
            self.position = 0
            self.epoch += 1

        self.iteration += 1


class SO(Optimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.permutation = torch.randperm(self.n, generator=self.rng)
        self.position = 0

    def step(self):
        batch_end = min(self.position + self.batch_size, self.n)

        idx = self.permutation[self.position:batch_end]

        grad = self.objective.stochastic_gradient(self.x, idx, normalization=self.epoch_norm)
        lr = self.current_lr()
        self.x = self.x - lr * grad

        self.position = batch_end

        if self.position >= self.n:
            self.position = 0
            self.epoch += 1

        self.iteration += 1
