"""
Gradient accumulator class for Abstract Gradient Training.

AGT requires training with large batch sizes to obtain tight guarantees. However, we also need to perform per-sample
computations for each sample in the batch. That is, for each batch, we must perform the following steps:

1. Compute the gradients for each sample in the batch.
2. Accumulate the gradients for each sample in the batch using the corresponding update rule.

Let us define the following and then explain the update rules:

- grad_n[i]: nominal gradients for the i-th training example.
- grad_wp_l[i], grad_wp_u[i]: lower and upper bound gradients for the i-th training example with weight 
  perturbation only.
- grad_iwp_l[i], grad_iwp_u[i]: lower and upper bound gradients for the i-th training example with both input and
  weight perturbation.

============== Nominal update rule ============== 

The nominal update rule is given by 

    d_theta <- 1 / batchsize * sum_{i=1}^{batchsize} grad_n[i]
    
where theta <- theta - alpha * d_theta are the updated model parameters.

============== Unlearning update rule ============== 

The unlearning update rule for removal of up to k samples is given by

    d_theta_l <- 1 / (batchsize - k) * SEMin_{batchsize-k}(grad_wp_l)
    d_theta_u <- 1 / (batchsize - k) * SEMax_{batchsize-k}(grad_wp_u)

where [theta_l, theta_u] <- [theta_l, theta_u] - alpha * [d_theta_l, d_theta_u] is the updated parameter interval.
SEMin/Max are the sum of the bottom/top (batchsize - k) gradients at each index over the batch dimension.

============== Poisoning update rule ==============

The poisoning update rule for bounded modification of up to k samples is given by

    d_theta_l <- 1 / batchsize * (SEMin_{k}(grad_iwp_l - grad_wp_l) + sum(grad_wp_l))
    d_theta_u <- 1 / batchsize * (SEMax_{k}(grad_iwp_u - grad_wp_u) + sum(grad_wp_u))
    
which corresponds to taking the weight perturbed bounds for each gradient, and additionally adding the top/bottom k
differences between the input+weight perturbed and weight perturbed bounds.

============== Privacy update rule ==============

The privacy update rule for substitution of up to k samples is given by

    d_theta_l <- 1 / batchsize * (SEMin_{batchsize - k}(grad_wp_l) - k * gamma)
    d_theta_u <- 1 / batchsize * (SEMax_{batchsize - k}(grad_wp_u) + k * gamma)

where gamma is the gradient clipping parameter.

============== Gradient Accumulation ==============

Each of the operations above must be performed for each sample in the batch, and for each parameter in the model. This
takes a large amount of memory that can easily be infeasible for large batch sizes, and is slow if limited to swapping
memory between the GPU and CPU. To address this, we can use the GradientAccumulator class to process the batch in
chunks called fragments, and gradually accumulate the gradients for each sample. This allows us to perform the
operations above without storing the gradients for all samples in the batch at once. For example, when taking the
SEMax_{k} operation, we can store only the top-k gradients for each fragment of the batch, then accumulate the top-k
from each fragment to obtain the top-k gradients for the entire batch.
"""

import logging
import torch

from abstract_gradient_training import interval_arithmetic

LOGGER = logging.getLogger(__name__)


class PoisoningGradientAccumulator:
    """
    A class that accumulates gradients across a batch in fragments, then computes the poisoning update rule.

        d_theta_l <- 1 / batchsize * (SEMin_{k}(grad_iwp_l - grad_wp_l) + sum(grad_wp_l))
        d_theta_u <- 1 / batchsize * (SEMax_{k}(grad_iwp_u - grad_wp_u) + sum(grad_wp_u))
    """

    def __init__(self, k_poison: int, param_n: list[torch.Tensor]):
        """
        Args:
            k_poison (int): The number of poisoned samples per batch.
            param_n (list[torch.Tensor]): The parameters of the model that the gradients relate to.
        """
        self.k_poison = k_poison
        # initialise containers to store the nominal and bounds on the gradients for each fragment
        self.grads_n = [torch.zeros_like(p) for p in param_n]  # nominal gradients
        self.grads_u = [torch.zeros_like(p) for p in param_n]  # upper bound gradients
        self.grads_l = [torch.zeros_like(p) for p in param_n]  # lower bound gradients
        # store the min-k and max-k differences between the input+weight perturbed and weight perturbed bounds that we
        # have seen so far in each fragment
        self.diffs_l = [torch.empty(0, *p.size(), dtype=p.dtype, device=p.device) for p in param_n]
        self.diffs_u = [torch.empty(0, *p.size(), dtype=p.dtype, device=p.device) for p in param_n]

    def add_clean_fragment_gradients(
        self, frag_grads_n: list[torch.Tensor], frag_grads_wp_l: list[torch.Tensor], frag_grads_wp_u: list[torch.Tensor]
    ) -> None:
        """
        Accumulate the gradients for a fragment of clean data with the given bounds on the weight perturbed gradients.

        Args:
            frag_grads_n (list[torch.Tensor]): The nominal gradients for the fragment.
            frag_grads_wp_l (list[torch.Tensor]): The lower bound gradients for the fragment with weight perturbation.
            frag_grads_wp_u (list[torch.Tensor]): The upper bound gradients for the fragment with weight perturbation.
        """
        # validate the input
        interval_arithmetic.validate_interval(
            frag_grads_wp_l, frag_grads_wp_u, frag_grads_n, msg="grad bounds, clean fragment"
        )
        # clean gradients can be summed directly
        for i in range(len(self.grads_n)):
            self.grads_l[i] = self.grads_l[i] + frag_grads_wp_l[i].sum(dim=0)
            self.grads_n[i] = self.grads_n[i] + frag_grads_n[i].sum(dim=0)
            self.grads_u[i] = self.grads_u[i] + frag_grads_wp_u[i].sum(dim=0)

    def add_poisoned_fragment_gradients(
        self,
        frag_grads_n: list[torch.Tensor],
        frag_grads_wp_l: list[torch.Tensor],
        frag_grads_wp_u: list[torch.Tensor],
        frag_grads_iwp_l: list[torch.Tensor],
        frag_grads_iwp_u: list[torch.Tensor],
    ) -> None:
        """
        Accumulate the gradients for a fragment of potentially poisoned data with the given bounds on the weight perturbed
        and input + weight perturbed gradients.

        Args:
            frag_grads_n (list[torch.Tensor]): The nominal gradients for the fragment.
            frag_grads_wp_l (list[torch.Tensor]): The lower bound gradients with weight perturbation.
            frag_grads_wp_u (list[torch.Tensor]): The upper bound gradients with weight perturbation.
            frag_grads_iwp_l (list[torch.Tensor]): The lower bound gradients with input + weight perturbation.
            frag_grads_iwp_u (list[torch.Tensor]): The upper bound gradients with input + weight perturbation.
        """
        # validate the input
        interval_arithmetic.validate_interval(
            frag_grads_wp_l, frag_grads_wp_u, frag_grads_n, msg="wp grad bounds, poisoned fragment"
        )
        interval_arithmetic.validate_interval(
            frag_grads_iwp_l, frag_grads_iwp_u, frag_grads_n, msg="i+wp grad bounds, poisoned fragment"
        )
        for i in range(len(self.grads_n)):
            # add weight perturbed bounds directly
            self.grads_l[i] = self.grads_l[i] + frag_grads_wp_l[i].sum(dim=0)
            self.grads_n[i] = self.grads_n[i] + frag_grads_n[i].sum(dim=0)
            self.grads_u[i] = self.grads_u[i] + frag_grads_wp_u[i].sum(dim=0)
            # calculate the differences beetween the input+weight perturbed and weight perturbed bounds
            frag_diff_l = frag_grads_iwp_l[i] - frag_grads_wp_l[i]
            frag_diff_u = frag_grads_iwp_u[i] - frag_grads_wp_u[i]
            # calculate the bottom/top-k differences for each parameter
            k = min(frag_diff_l.size(0), self.k_poison)
            frag_diff_l = torch.topk(frag_diff_l, k, dim=0, largest=False, sorted=False).values
            frag_diff_u = torch.topk(frag_diff_u, k, dim=0, sorted=False).values
            # concatenate with the bottom/top-k differences from previous fragments
            frag_diff_l = torch.cat([self.diffs_l[i], frag_diff_l], dim=0)
            frag_diff_u = torch.cat([self.diffs_u[i], frag_diff_u], dim=0)
            # take the new top/bottom k
            k = min(frag_diff_l.size(0), self.k_poison)
            self.diffs_l[i] = torch.topk(frag_diff_l, k, dim=0, largest=False, sorted=False).values
            self.diffs_u[i] = torch.topk(frag_diff_u, k, dim=0, sorted=False)[0]

    def concretize_gradient_update(
        self, batchsize: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Compute the final gradient update rule, given all of the accumulated gradients for each fragment sent to this
        class during training.

            d_theta_l <- 1 / batchsize * (SEMin_{k}(grad_iwp_l - grad_wp_l) + sum(grad_wp_l))
            d_theta_n <- 1 / batchsize * sum(grad_n)
            d_theta_u <- 1 / batchsize * (SEMax_{k}(grad_iwp_u - grad_wp_u) + sum(grad_wp_u))

        Args:
            batchsize (int): The size of the batch.

        Returns:
            update_l: The lower bound on the descent direction.
            update_n: The nominal descent direction.
            update_u: The upper bound on the descent direction.
        """
        # accumulate the top-k diffs from each fragment then add the overall top-k diffs to the gradient bounds
        for i in range(len(self.grads_n)):
            assert self.diffs_u[i].size(0) == self.k_poison
            assert self.diffs_l[i].size(0) == self.k_poison
            self.grads_l[i] += self.diffs_l[i].sum(dim=0)
            self.grads_u[i] += self.diffs_u[i].sum(dim=0)

        # normalise each by the batchsize
        update_l = [g / batchsize for g in self.grads_l]
        update_n = [g / batchsize for g in self.grads_n]
        update_u = [g / batchsize for g in self.grads_u]

        # clear the memory of this batch, and return the previous values
        self.grads_n = [torch.zeros_like(p) for p in self.grads_n]  # nominal gradients
        self.grads_u = [torch.zeros_like(p) for p in self.grads_n]  # upper bound gradients
        self.grads_l = [torch.zeros_like(p) for p in self.grads_n]  # lower bound gradients
        self.diffs_l = [torch.empty(0, *p.size(), dtype=p.dtype, device=p.device) for p in self.grads_n]
        self.diffs_u = [torch.empty(0, *p.size(), dtype=p.dtype, device=p.device) for p in self.grads_n]
        return update_l, update_n, update_u


class PrivacyGradientAccumulator:
    """
    A class that accumulates gradients across a batch in fragments, then computes the privacy update rule.

        d_theta_l <- 1 / batchsize * (SEMin_{batchsize - k}(grad_wp_l) - k * gamma)
        d_theta_u <- 1 / batchsize * (SEMax_{batchsize - k}(grad_wp_u) + k * gamma)
    """

    def __init__(self, k_private: int, clip_gamma: float, param_n: list[torch.Tensor]):
        """
        Args:
            k_private (int): The number of poisoned samples per batch.
            clip_gamma (float): The gradient clipping parameter.
            param_n (list[torch.Tensor]): The parameters of the model that the gradients relate to.
        """
        self.k_private = k_private
        self.clip_gamma = clip_gamma
        # initialise containers to store the nominal and bounds on the gradients for each fragment
        self.grads_n = [torch.zeros_like(p) for p in param_n]  # nominal gradients
        self.grads_l = [torch.zeros_like(p) for p in param_n]  # lower bound gradient
        self.grads_u = [torch.zeros_like(p) for p in param_n]  # upper bound gradient
        # the privacy update rule requires the top/bottom (batchsize - k) gradient bounds.
        # instead of this, we store the top/bottom k gradients we have seen so far, and add the rest to the descent
        # direction bounds directly
        self.grads_l_top_k = [torch.empty(0, *p.size(), dtype=p.dtype, device=p.device) for p in self.grads_n]
        self.grads_u_bottom_k = [torch.empty(0, *p.size(), dtype=p.dtype, device=p.device) for p in self.grads_n]

    def add_public_fragment_gradients(
        self, frag_grads_n: list[torch.Tensor], frag_grads_l: list[torch.Tensor], frag_grads_u: list[torch.Tensor]
    ) -> None:
        """
        Accumulate the gradients for a fragment of public data with the given bounds on the weight perturbed gradients.

        Args:
            frag_grads_n (list[torch.Tensor]): The nominal gradients for the fragment.
            frag_grads_l (list[torch.Tensor]): The lower bound gradients for the fragment with weight perturbation.
            frag_grads_u (list[torch.Tensor]): The upper bound gradients for the fragment with weight perturbation.
        """
        # validate the input
        interval_arithmetic.validate_interval(
            frag_grads_l, frag_grads_u, frag_grads_n, msg="grad bounds, public fragment"
        )
        # clean gradients can be summed directly
        for i in range(len(self.grads_n)):
            self.grads_l[i] = self.grads_l[i] + frag_grads_l[i].sum(dim=0)
            self.grads_n[i] = self.grads_n[i] + frag_grads_n[i].sum(dim=0)
            self.grads_u[i] = self.grads_u[i] + frag_grads_u[i].sum(dim=0)

    def add_private_fragment_gradients(
        self, frag_grads_n: list[torch.Tensor], frag_grads_l: list[torch.Tensor], frag_grads_u: list[torch.Tensor]
    ) -> None:
        """
        Accumulate the gradients for a fragment of private data with the given bounds on the weight perturbed gradients.

        Args:
            frag_grads_n (list[torch.Tensor]): The nominal gradients for the fragment.
            frag_grads_l (list[torch.Tensor]): The lower bound gradients for the fragment with weight perturbation.
            frag_grads_u (list[torch.Tensor]): The upper bound gradients for the fragment with weight perturbation.
        """
        # validate the input
        interval_arithmetic.validate_interval(
            frag_grads_l, frag_grads_u, frag_grads_n, msg="grad bounds, private fragment"
        )
        for i in range(len(self.grads_n)):
            # accumulate the nominal gradients
            self.grads_n[i] += frag_grads_n[i].sum(dim=0)
            # take the top k lower bounds and bottom k upper bounds from the fragment
            k = min(frag_grads_l[i].size(0), self.k_private)
            l_top_k = torch.topk(frag_grads_l[i], k, largest=True, dim=0, sorted=False).values
            u_bottom_k = torch.topk(frag_grads_u[i], k, largest=False, dim=0, sorted=False).values
            # accumulate the rest of the gradients from the fragment to the bounds
            self.grads_l[i] += frag_grads_l[i].sum(dim=0) - l_top_k.sum(dim=0)
            self.grads_u[i] += frag_grads_u[i].sum(dim=0) - u_bottom_k.sum(dim=0)
            # now we concatenate the top/bottom k gradients from the fragment to the top/bottom k gradients we have seen
            frag_grads_l[i] = torch.cat([l_top_k, self.grads_l_top_k[i]], dim=0)
            frag_grads_u[i] = torch.cat([u_bottom_k, self.grads_u_bottom_k[i]], dim=0)
            # take the new top k lower bound gradients and bottom k upper bound gradients
            k = min(frag_grads_l[i].size(0), self.k_private)
            self.grads_l_top_k[i] = torch.topk(frag_grads_l[i], k, largest=True, dim=0, sorted=False).values
            self.grads_u_bottom_k[i] = torch.topk(frag_grads_u[i], k, largest=False, dim=0, sorted=False).values
            # accumulate the required sums to the bounds
            self.grads_l[i] += frag_grads_l[i].sum(dim=0) - self.grads_l_top_k[i].sum(dim=0)
            self.grads_u[i] += frag_grads_u[i].sum(dim=0) - self.grads_u_bottom_k[i].sum(dim=0)

    def concretize_gradient_update(
        self, batchsize: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Compute the final gradient update rule, given all of the accumulated gradients for each fragment sent to this
        class during training.

            d_theta_l <- 1 / batchsize * (SEMin_{batchsize - k}(grad_wp_l) - k * gamma)
            d_theta_n <- 1 / batchsize * sum(grad_n)
            d_theta_u <- 1 / batchsize * (SEMax_{batchsize - k}(grad_wp_u) + k * gamma)

        Args:
            batchsize (int): The size of the batch.

        Returns:
            update_l: The lower bound on the descent direction.
            update_n: The nominal descent direction.
            update_u: The upper bound on the descent direction.
        """
        for i in range(len(self.grads_n)):
            assert self.grads_u_bottom_k[i].size(0) == self.k_private
            assert self.grads_l_top_k[i].size(0) == self.k_private
            self.grads_l[i] -= self.k_private * self.clip_gamma
            self.grads_u[i] += self.k_private * self.clip_gamma

        # normalise each by the batchsize
        update_l = [g / batchsize for g in self.grads_l]
        update_n = [g / batchsize for g in self.grads_n]
        update_u = [g / batchsize for g in self.grads_u]

        # clear the memory of this batch, and return the previous values
        self.grads_n = [torch.zeros_like(p) for p in self.grads_n]  # nominal gradients
        self.grads_l = [torch.zeros_like(p) for p in self.grads_n]  # lower bound gradient
        self.grads_u = [torch.zeros_like(p) for p in self.grads_n]  # upper bound gradient
        self.grads_l_top_k = [torch.empty(0, *p.size(), dtype=p.dtype, device=p.device) for p in self.grads_n]
        self.grads_u_bottom_k = [torch.empty(0, *p.size(), dtype=p.dtype, device=p.device) for p in self.grads_n]
        return update_l, update_n, update_u


class UnlearningGradientAccumulator:
    """
    A class that accumulates gradients across a batch in fragments, then computes the unlearning update rule.

        d_theta_l <- 1 / (batchsize - k) * SEMin_{batchsize-k}(grad_wp_l)
        d_theta_u <- 1 / (batchsize - k) * SEMax_{batchsize-k}(grad_wp_u)

    The fragment accumulation rule is the same as the privacy one, except that we don't add the gamma term at the end,
    and divide by batchsize - k.
    """

    def __init__(self, k_unlearn: int, param_n: list[torch.Tensor]):
        """
        Args:
            k_unlearn (int): The number of poisoned samples per batch.
            param_n (list[torch.Tensor]): The parameters of the model that the gradients relate to.
        """
        self.k_unlearn = k_unlearn

        # initialise containers to store the nominal and bounds on the gradients for each fragment
        self.grads_n = [torch.zeros_like(p) for p in param_n]  # nominal gradient
        self.grads_l = [torch.zeros_like(p) for p in param_n]  # lower bound gradient
        self.grads_u = [torch.zeros_like(p) for p in param_n]  # upper bound gradient
        # the unlearning update rule requires the top/bottom (batchsize - k) gradient bounds.
        # instead of this, we store the top/bottom k gradients we have seen so far, and add the rest to the descent
        # direction bounds directly
        self.grads_l_top_k = [torch.empty(0, *p.size(), dtype=p.dtype, device=p.device) for p in self.grads_n]
        self.grads_u_bottom_k = [torch.empty(0, *p.size(), dtype=p.dtype, device=p.device) for p in self.grads_n]

    def add_fragment_gradients(
        self, frag_grads_n: list[torch.Tensor], frag_grads_l: list[torch.Tensor], frag_grads_u: list[torch.Tensor]
    ) -> None:
        """
        Accumulate the gradients for a fragment of data with the given bounds on the weight perturbed gradients.

        Args:
            frag_grads_n (list[torch.Tensor]): The nominal gradients for the fragment.
            frag_grads_l (list[torch.Tensor]): The lower bound gradients for the fragment with weight perturbation.
            frag_grads_u (list[torch.Tensor]): The upper bound gradients for the fragment with weight perturbation.
        """
        # validate the input
        interval_arithmetic.validate_interval(
            frag_grads_l, frag_grads_u, frag_grads_n, msg="grad bounds, private fragment"
        )
        for i in range(len(self.grads_n)):
            # accumulate the nominal gradients
            self.grads_n[i] += frag_grads_n[i].sum(dim=0)
            # take the top k lower bounds and bottom k upper bounds from the fragment
            k = min(frag_grads_l[i].size(0), self.k_unlearn)
            l_top_k = torch.topk(frag_grads_l[i], k, largest=True, dim=0, sorted=False).values
            u_bottom_k = torch.topk(frag_grads_u[i], k, largest=False, dim=0, sorted=False).values
            # accumulate the rest of the gradients from the fragment to the bounds
            self.grads_l[i] += frag_grads_l[i].sum(dim=0) - l_top_k.sum(dim=0)
            self.grads_u[i] += frag_grads_u[i].sum(dim=0) - u_bottom_k.sum(dim=0)
            # now we concatenate the top/bottom k gradients from the fragment to the top/bottom k gradients we have seen
            frag_grads_l[i] = torch.cat([l_top_k, self.grads_l_top_k[i]], dim=0)
            frag_grads_u[i] = torch.cat([u_bottom_k, self.grads_u_bottom_k[i]], dim=0)
            # take the new top k lower bound gradients and bottom k upper bound gradients
            k = min(frag_grads_l[i].size(0), self.k_unlearn)
            self.grads_l_top_k[i] = torch.topk(frag_grads_l[i], k, largest=True, dim=0, sorted=False).values
            self.grads_u_bottom_k[i] = torch.topk(frag_grads_u[i], k, largest=False, dim=0, sorted=False).values
            # accumulate the required sums to the bounds
            self.grads_l[i] += frag_grads_l[i].sum(dim=0) - self.grads_l_top_k[i].sum(dim=0)
            self.grads_u[i] += frag_grads_u[i].sum(dim=0) - self.grads_u_bottom_k[i].sum(dim=0)

    def concretize_gradient_update(
        self, batchsize: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Compute the final gradient update rule, given all of the accumulated gradients for each fragment sent to this
        class during training.

            d_theta_l <- 1 / (batchsize - k) * SEMin_{batchsize-k}(grad_wp_l)
            d_theta_n <- 1 / batchsize * sum(grad_n)
            d_theta_u <- 1 / (batchsize - k) * SEMax_{batchsize-k}(grad_wp_u)

        Args:
            batchsize (int): The size of the batch.

        Returns:
            update_l: The lower bound on the descent direction.
            update_n: The nominal descent direction.
            update_u: The upper bound on the descent direction.
        """
        for i in range(len(self.grads_n)):
            assert self.grads_u_bottom_k[i].size(0) == self.k_unlearn
            assert self.grads_l_top_k[i].size(0) == self.k_unlearn

        # normalise each by the batchsize
        update_l = [g / (batchsize - self.k_unlearn) for g in self.grads_l]
        update_n = [g / batchsize for g in self.grads_n]
        update_u = [g / (batchsize - self.k_unlearn) for g in self.grads_u]

        # clear the memory of this batch, and return the previous values
        self.grads_n = [torch.zeros_like(p) for p in self.grads_n]  # nominal gradient
        self.grads_l = [torch.zeros_like(p) for p in self.grads_n]  # lower bound gradient
        self.grads_u = [torch.zeros_like(p) for p in self.grads_n]  # upper bound gradient
        self.grads_l_top_k = [torch.empty(0, *p.size(), dtype=p.dtype, device=p.device) for p in self.grads_n]
        self.grads_u_bottom_k = [torch.empty(0, *p.size(), dtype=p.dtype, device=p.device) for p in self.grads_n]
        return update_l, update_n, update_u
