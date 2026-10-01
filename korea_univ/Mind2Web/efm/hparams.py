# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");

"""EFM_TASK.md 3.7 hyperparameter table, as a CLI-overridable dataclass. Ported verbatim from
WebArena/efm/hparams.py -- the algorithm's hyperparameters are domain-agnostic."""

from dataclasses import dataclass


@dataclass
class EFMHParams:
    tau_d: float = 0.85
    tau_c: float = 0.85
    m_cand: int = 10
    k_cand: int = 5
    eta: float = 0.1
    gamma: float = 0.98
    t_stale: int = 60
    grace: int = 20
    r_min: int = 3
    rho: float = 0.34
    quality_delete: bool = False

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument("--efm-tau-d", type=float, default=cls.tau_d)
        parser.add_argument("--efm-tau-c", type=float, default=cls.tau_c)
        parser.add_argument("--efm-m-cand", type=int, default=cls.m_cand)
        parser.add_argument("--efm-k-cand", type=int, default=cls.k_cand)
        parser.add_argument("--efm-eta", type=float, default=cls.eta)
        parser.add_argument("--efm-gamma", type=float, default=cls.gamma)
        parser.add_argument("--efm-t-stale", type=int, default=cls.t_stale)
        parser.add_argument("--efm-grace", type=int, default=cls.grace)
        parser.add_argument("--efm-r-min", type=int, default=cls.r_min)
        parser.add_argument("--efm-rho", type=float, default=cls.rho)
        parser.add_argument("--efm-quality-delete", action="store_true", default=cls.quality_delete)

    @classmethod
    def from_args(cls, args):
        return cls(
            tau_d=args.efm_tau_d, tau_c=args.efm_tau_c, m_cand=args.efm_m_cand,
            k_cand=args.efm_k_cand, eta=args.efm_eta, gamma=args.efm_gamma,
            t_stale=args.efm_t_stale, grace=args.efm_grace, r_min=args.efm_r_min,
            rho=args.efm_rho, quality_delete=args.efm_quality_delete,
        )
