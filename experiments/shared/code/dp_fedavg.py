"""Ideia B — Central DP at aggregation (DP-FedAvg).

Mecanismo de DP alternativo ao DP-SGD per-sample do Opacus:
clientes treinam **sem** PrivacyEngine; o servidor (agregador) clipa cada
update Δ_i = w_i - w_global, soma-os, divide pelo número de clientes
(= FedAvg padrão), e adiciona ruído gaussiano calibrado para garantir
(ε, δ)-DP por rodada via RDP.

Comparado a DP-SGD:
- Privacidade é **por rodada de agregação**, não por amostra. Unidade
  de protege é o **cliente**.
- Não exige `GradSampleModule`/`PrivacyEngine` no cliente — cliente
  treina como se não houvesse DP.
- Calibração: `sample_rate = num_clients_selected / total_clients`,
  `steps = num_rounds`.
- Aplicação natural em **HFL**. Em VFL não há agregação de updates
  homólogos — os silos contribuem com features ortogonais. Esta versão
  só implementa o caminho HFL; VFL exige um mecanismo diferente
  (e.g., LDP no embedding antes da fusão), que deverá ser implementado
  separadamente.

Decisão registrada em silo/decisions/2026-05-23_paper-rescue-experiments.md
"""
from __future__ import annotations

from typing import Iterable, Mapping

import torch


def _state_diff(new: Mapping[str, torch.Tensor], old: Mapping[str, torch.Tensor]
                ) -> dict[str, torch.Tensor]:
    """Δ = new - old, key-by-key, em float32 no device do new."""
    return {k: (new[k].detach().float() - old[k].detach().float().to(new[k].device))
            for k in new.keys()}


def _flat_norm(diff: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """L2 norm sobre todos os tensores do dict concatenados."""
    sq = sum((t.detach().pow(2).sum() for t in diff.values()), start=torch.tensor(0.0))
    return sq.sqrt()


def clip_state_diff(diff: Mapping[str, torch.Tensor], max_norm: float
                    ) -> dict[str, torch.Tensor]:
    """Per-client clip: redimensiona o update inteiro pra ter L2 ≤ max_norm."""
    nrm = _flat_norm(diff)
    factor = (max_norm / (nrm + 1e-12)).clamp(max=1.0)
    return {k: v * factor for k, v in diff.items()}


def add_gaussian_noise(
    summed: Mapping[str, torch.Tensor],
    noise_multiplier: float,
    max_grad_norm: float,
) -> dict[str, torch.Tensor]:
    """Adiciona ruído gaussiano N(0, σ·C) a cada tensor do dict somado."""
    out = {}
    for k, v in summed.items():
        noise = torch.randn_like(v) * (noise_multiplier * max_grad_norm)
        out[k] = v + noise
    return out


def dp_fedavg_aggregate(
    global_state: Mapping[str, torch.Tensor],
    client_states: list[Mapping[str, torch.Tensor]],
    max_grad_norm: float,
    noise_multiplier: float,
) -> dict[str, torch.Tensor]:
    """DP-FedAvg de um round.

    Args:
        global_state: pesos do modelo global pré-round.
        client_states: lista de state_dicts treinados localmente pelos K clientes.
        max_grad_norm: bound L2 por-cliente sobre o update.
        noise_multiplier: σ tal que ruído gaussiano = N(0, σ·C).

    Returns:
        Novo state_dict global = global + (Σ clip(Δ_i) + noise) / K.
    """
    if not client_states:
        return dict(global_state)
    K = len(client_states)
    # 1. Per-client clip
    clipped = [clip_state_diff(_state_diff(c, global_state), max_grad_norm)
               for c in client_states]
    # 2. Soma dos updates clipados
    summed = {k: torch.stack([c[k] for c in clipped]).sum(dim=0) for k in clipped[0]}
    # 3. Ruído gaussiano (sobre a soma)
    noisy = add_gaussian_noise(summed, noise_multiplier, max_grad_norm)
    # 4. Média (a unidade canônica de FedAvg) → atualiza global
    return {k: global_state[k] + (v / K).to(global_state[k].device)
            for k, v in noisy.items()}


def calibrate_dp_fedavg_sigma(
    target_epsilon: float,
    num_rounds: int,
    sample_rate: float,
    delta: float,
) -> float:
    """Calibra σ para o cronograma de DP-FedAvg via opacus get_noise_multiplier.

    `sample_rate` = clientes selecionados por round / clientes totais.
    `epochs` no sentido do opacus = `num_rounds` (cada round é uma "amostra").

    Retorna σ tal que composição RDP sobre `num_rounds` passos com Gaussian
    mechanism (sensibilidade `max_grad_norm` absorvida na razão σ/C)
    satisfaz (ε, δ)-DP.
    """
    if target_epsilon <= 0:
        return 0.0
    from opacus.accountants.utils import get_noise_multiplier
    return float(get_noise_multiplier(
        target_epsilon=float(target_epsilon),
        target_delta=float(delta),
        sample_rate=float(sample_rate),
        epochs=int(num_rounds),
        accountant="rdp",
    ))


def composed_epsilon_fedavg(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
) -> float:
    """ε composto após `steps` rounds de DP-FedAvg via RDPAccountant."""
    if noise_multiplier <= 0 or steps <= 0:
        return 0.0
    from opacus.accountants import RDPAccountant
    acc = RDPAccountant()
    for _ in range(int(steps)):
        acc.step(noise_multiplier=float(noise_multiplier),
                 sample_rate=float(sample_rate))
    return float(acc.get_epsilon(delta=float(delta)))


# =========================================================================
# Notas de implementação para integração com os scripts existentes
# =========================================================================
#
# Para usar DP-FedAvg num script de exp01/exp02/exp03 (HFL):
#
# 1. No cliente: NÃO chamar PrivacyEngine.make_private. O cliente treina
#    com SGD/Adam padrão. O atual `FederatedClient.train(use_dp=True)`
#    precisa ganhar uma flag `mechanism="dp_sgd"|"dp_fedavg"`.
#
# 2. No servidor (loop principal):
#    - Antes do loop: σ = calibrate_dp_fedavg_sigma(eps, num_rounds, K/N, δ).
#    - Cada round:
#        client_states = [c.train(global, mechanism="dp_fedavg") for c in clients]
#        new_global = dp_fedavg_aggregate(global_state, client_states,
#                                         max_grad_norm=C, noise_multiplier=σ)
#        eps_spent = composed_epsilon_fedavg(σ, K/N, round_idx+1, δ)
#
# 3. C (max_grad_norm) tipicamente ~ 1.0-5.0; calibrar empiricamente.
#    Valor sugerido inicial: 1.0 (mais conservador que o C=5.0 do DP-SGD).
#
# Para VFL: este módulo NÃO se aplica. VFL precisaria de um mecanismo
# diferente (e.g., ruído LDP no embedding pré-fusão). Deixar como TODO.
