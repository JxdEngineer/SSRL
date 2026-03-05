import torch
import torch.nn.functional as F


def vicreg_loss(
    h1: torch.Tensor,
    h2: torch.Tensor,
    sim_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    var_eps: float = 1e-4,
) -> torch.Tensor:
    """
    VICReg (variance-invariance-covariance regularization)

    h1, h2: (B, D) projected embeddings (NOT necessarily normalized).
            You should pass pairs (same damage, different excitation).
            B should be >= 2 for variance/covariance terms to work.

    Returns a scalar loss:
      sim_weight * invariance + var_weight * variance + cov_weight * covariance
    """

    # invariance (representation similarity)
    sim_loss = F.mse_loss(h1, h2)

    # variance: each dimension should have std >= 1
    def _var_term(x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] < 2:
            return x.new_zeros(())
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0, unbiased=False) + var_eps)
        return F.relu(1.0 - std).mean()

    var_loss = _var_term(h1) + _var_term(h2)

    # covariance: decorrelate dimensions
    def _cov_term(x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] < 2:
            return x.new_zeros(())
        x = x - x.mean(dim=0, keepdim=True)
        B, D = x.shape
        cov = (x.t() @ x) / max(B - 1, 1)  # (D,D)
        off_diag = cov.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
        return (off_diag ** 2).mean()

    cov_loss = _cov_term(h1) + _cov_term(h2)

    return sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss

def vicreg_loss_individual(
    h1: torch.Tensor,
    h2: torch.Tensor,
    var_eps: float = 1e-4,
) -> torch.Tensor:
    """
    VICReg (variance-invariance-covariance regularization)

    h1, h2: (B, D) projected embeddings (NOT necessarily normalized).
            You should pass pairs (same damage, different excitation).
            B should be >= 2 for variance/covariance terms to work.

    Returns a scalar loss:
      sim_weight * invariance + var_weight * variance + cov_weight * covariance
    """

    # invariance (representation similarity)
    sim_loss = F.mse_loss(h1, h2)

    # variance: each dimension should have std >= 1
    def _var_term(x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] < 2:
            return x.new_zeros(())
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0, unbiased=False) + var_eps)
        return F.relu(0.2 - std).mean()

    var_loss = _var_term(h1) + _var_term(h2)

    # covariance: decorrelate dimensions
    def _cov_term(x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] < 2:
            return x.new_zeros(())
        x = x - x.mean(dim=0, keepdim=True)
        B, D = x.shape
        cov = (x.t() @ x) / max(B - 1, 1)  # (D,D)
        off_diag = cov.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
        return (off_diag ** 2).mean()

    cov_loss = _cov_term(h1) + _cov_term(h2)

    return sim_loss, var_loss, cov_loss

def InfoNCE(h: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    h: (M, D) L2-normalized, M>=2
    positives: all non-self (collapse baseline samples together)
    """
    M = h.shape[0]
    if M <= 1:
        return h.new_zeros(())

    logits = (h @ h.t()) / max(float(temperature), 1e-8)
    logits = logits - logits.max(dim=1, keepdim=True).values

    mask = ~torch.eye(M, device=h.device, dtype=torch.bool)  # non-self
    exp_logits = torch.exp(logits) * mask.float()
    denom = exp_logits.sum(dim=1) + 1e-8

    log_prob = logits - torch.log(denom.view(-1, 1))
    mean_log_prob_pos = (log_prob * mask.float()).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    return -mean_log_prob_pos.mean()