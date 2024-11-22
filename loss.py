import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss


class JSDLoss(_Loss):
    """Jensen-Shannon divergence loss."""

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        log_target: bool = False,
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        M = 0.5 * (input + target)
        return 0.5 * F.kl_div(
            input, M, reduction=self.reduction, log_target=self.log_target
        ) + 0.5 * F.kl_div(
            target, M, reduction=self.reduction, log_target=self.log_target
        )
