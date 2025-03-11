import torch
from torch.optim import Optimizer


class NoisedProjectedSGD(Optimizer):
    def __init__(self, params, lr=0.01, radius=1.0, noise_scale=0.01):
        """
        Initialize the Noised Projected SGD optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
            radius (float): Radius of the projection ball.
            noise_scale (float): Scale of the noise to be added.
        """
        defaults = dict(lr=lr, radius=radius, noise_scale=noise_scale)
        super(NoisedProjectedSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Gradient descent step
                d_p = p.grad.data
                param_update = -group["lr"] * d_p

                # Add noise
                noise = group["noise_scale"] * torch.randn_like(p.data)
                param_update += noise

                # Project onto the ball
                p.data.add_(param_update)
                p.data = self._project_onto_ball(p.data, group["radius"])

        return loss

    def _project_onto_ball(self, tensor, radius):
        """
        Project the tensor onto an L2 ball of given radius.

        Args:
            tensor (torch.Tensor): The tensor to project.
            radius (float): The radius of the ball.

        Returns:
            torch.Tensor: The projected tensor.
        """
        norm = tensor.norm(p=2)
        if norm > radius:
            return tensor * (radius / norm)
        return tensor
