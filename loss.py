import sys
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from typing import List
from forecasting_models.pytorch.distribution_loss import PredictdEGPDLossTrunc
class DictWrapper:
    def __init__(self, d):
        self.d = d
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Sequence
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
def check_finite(name, x):
    if isinstance(x, torch.Tensor):
        if not torch.isfinite(x).all():
            raise RuntimeError(f"[NaN ERROR] {name} contains NaN or Inf")
    else:
        import numpy as np
        if not np.isfinite(x):
            raise RuntimeError(f"[NaN ERROR] {name} contains NaN or Inf")
def check_finite(name, x):
    if isinstance(x, torch.Tensor):
        if not torch.isfinite(x).all():
            raise ValueError(f"[{name}] contains NaN/Inf")
    return True

class ClusterCLMBinnedTransitionLoss(nn.Module):
    def __init__(
        self,

        # ============================================================
        # General
        # ============================================================
        num_classes: int,
        sigma:float,
        id: int = 0,
        iddept: int = 4,
        eps: float = 1e-4,
        
        # ============================================================
        # Group structure
        # ============================================================
        nclusters: int = 1,
        ndepartements: int = 1,

        # ============================================================
        # Ordinal threshold model
        # ============================================================
        alphatype: str = "department",
        learngains: bool = False,
        gainsfloor: float = 2.5,

        # ============================================================
        # Monotonic transition loss
        # ============================================================
        wk=None,
        wkdecay: str = "None",
        wkpower: float = 2.06,
        wklambda: float = 0.35,
        wtrans: float = 1.0,

        # ============================================================
        # Soft class-center estimation
        # ============================================================
        taugate: float = 0.05,
        gatetemp: float = 0.11,
        massupdate: float = 0.5,

        # ============================================================
        # EMA priors for class centers
        # ============================================================
        mumomentum: float = 0.95,
        mulambdag: float = 0.18,
        mulambdac: float = 1.61,
        mulambdad: float = 0.75,
        muinit=None,

        # ============================================================
        # Coverage loss
        # ============================================================
        coverageagg: str = "global",          # "cluster", "department", "global"
        coveragedistance: str = "cdf_l1",     # "cdf_l2", "cdf_l1", "cdf_linf", "l2"
        coveragewarmupupdates: int = 0,
        warmupes: int = 2,
        wcoverage: float = 1.73,
        shift: float = 0.0,

        # ============================================================
        # Auxiliary regularization
        # ============================================================
        wmu0: float = 1.94,
        wmid: float = 0.2,

        ):
        super().__init__()

        self.C = int(num_classes)
        self.id = int(id)
        self.idept = int(iddept)
        self.nclusters = int(nclusters)
        self.ndepartements = int(ndepartements)
        self.alphatype = str(alphatype)

        self.beta = 0.0
        self.t = 0.0
        self.eps = float(eps)

        self.wcoverage = float(wcoverage)
        self.warmupes = int(warmupes)
        self.wmu0 = float(wmu0)
        self.wtrans = float(wtrans)

        self.taugate = float(taugate)
        self.gatetemp = float(gatetemp)

        self.mu_momentum = float(mumomentum)
        self.mu_lambda_g = float(mulambdag)
        self.mu_lambda_c = float(mulambdac)
        self.mu_lambda_d = float(mulambdad)
        
        self.sigma = sigma
        
        clustersequaldept = self.id == self.idept
        
        if clustersequaldept:
            self.mu_lambda_c = 0
        if self.ndepartements == 1:
            self.mu_lambda_d = 0
        
        self.massupdate = float(massupdate)
        
        self.wmid = wmid          # poids de Lmid
        self.mid_detach_mu = True

        _buf_size = self.nclusters
        _dept_buf_size = self.ndepartements

        self.cluster_raw_to_slot = {}
        self.departement_raw_to_slot = {}
        self.cluster_next_free_slot = 0
        self.departement_next_free_slot = 0

        self.register_buffer(
            "cluster_slot_to_raw",
            torch.full((_buf_size,), -1, dtype=torch.long)
        )
        self.register_buffer(
            "departement_slot_to_raw",
            torch.full((_dept_buf_size,), -1, dtype=torch.long)
        )

        if muinit is not None:
            _mu = torch.as_tensor(muinit, dtype=torch.float32)
            if _mu.shape == (self.nclusters, self.C):
                self.register_buffer("mu_prior", _mu.clone())
            else:
                raise ValueError(
                    f"muinit shape {tuple(_mu.shape)} != ({self.nclusters}, {self.C})"
                )
        else:
            self.register_buffer(
                "mu_prior",
                torch.full((_buf_size, self.C), float("nan"), dtype=torch.float32)
            )

        self.register_buffer(
            "mu_prior_global",
            torch.full((self.C,), float("nan"), dtype=torch.float32)
        )
        
        self.register_buffer(
            "mu_prior_departement",
            torch.full((_dept_buf_size, self.C), float("nan"), dtype=torch.float32)
        )

        self.wkdecay = wkdecay
        self.wkpower = wkpower
        self.wklambda = wklambda
        self.wkmin = 0.0

        self.P = {k: [(a, a + k) for a in range(0, self.C - k)] for k in range(1, self.C)}
        if wk is None:
            self.wk = self._build_wk_monotone()
        else:
            self.wk = wk
            
        self.all_pairs = [(a, b) for k in range(1, self.C) for (a, b) in self.P[k]]
        self.pair_to_idx = {(a, b): i for i, (a, b) in enumerate(self.all_pairs)}
        num_pairs = len(self.all_pairs)

        if self.alphatype == "cluster":
            self.alpha = nn.Parameter(torch.zeros(_buf_size, self.C - 1))
        elif self.alphatype == "department":
            self.alpha = nn.Parameter(torch.zeros(_dept_buf_size, self.C - 1))
        else:
            self.alpha = nn.Parameter(torch.zeros(self.C - 1))

        self.learn_gains = bool(learngains)
        self.gains_floor = float(gainsfloor)
        if self.learn_gains:
            self.g_raw = nn.Parameter(torch.zeros(self.C - 1))
        else:
            self.g_raw = torch.zeros(self.C - 1)

        _default_scale = torch.ones(num_pairs)
        self.register_buffer("delta_scale_ema", _default_scale)

        self.scale_momentum = mumomentum
        self.scale_min = 1e-3
        self.scale_max = 1e3

        self.tau_loss = 1.5
        self.loss_ema_momentum = mumomentum
        self.register_buffer(
            "loss_ema",
            torch.zeros(_buf_size, dtype=torch.float32)
        )
        
        # --------------------------------------------------
        # Coverage distributionnelle
        # --------------------------------------------------
        self.wcoverage = float(wcoverage) if wcoverage is not None else float(wfocal)
        self.coverage_momentum = float(mumomentum)
        self.coverage_warmup_updates = int(coveragewarmupupdates)
        self.coverage_distance = str(coveragedistance)
        self.coverageagg = str(coverageagg)
        self.shift = shift

        if self.coverage_distance not in {"cdf_l2", "cdf_l1", "cdf_linf", "l2"}:
            raise ValueError(
                "coveragedistance must be one of "
                "{'cdf_l2', 'cdf_l1', 'cdf_linf', 'l2'}"
            )

        if self.coverageagg not in {"cluster", "department", "global"}:
            raise ValueError(
                "coverageagg must be one of {'cluster', 'department', 'global'}"
            )

        self.register_buffer(
            "coverage_target_global",
            torch.full((self.C,), float("nan"), dtype=torch.float32),
        )

        self.register_buffer(
            "coverage_target_cluster",
            torch.full((_buf_size, self.C), float("nan"), dtype=torch.float32),
        )

        self.register_buffer(
            "coverage_target_departement",
            torch.full((_dept_buf_size, self.C), float("nan"), dtype=torch.float32),
        )

        self.register_buffer(
            "coverage_pred_ema_cluster",
            torch.full((_buf_size, self.C), float("nan"), dtype=torch.float32),
        )

        self.register_buffer(
            "coverage_pred_ema_departement",
            torch.full((_dept_buf_size, self.C), float("nan"), dtype=torch.float32),
        )

        self.register_buffer(
            "coverage_pred_ema_global",
            torch.full((self.C,), float("nan"), dtype=torch.float32),
        )

        self.register_buffer(
            "coverage_update_count_cluster",
            torch.zeros((_buf_size,), dtype=torch.long),
        )

        self.register_buffer(
            "coverage_update_count_departement",
            torch.zeros((_dept_buf_size,), dtype=torch.long),
        )

        self.register_buffer(
            "coverage_update_count_global",
            torch.zeros((), dtype=torch.long),
        )

    def _build_wk_monotone(self):
        wk = {}
        decay = getattr(self, "wkdecay", "power")
        raw = {}

        for k in range(1, self.C):
            if decay == "exp":
                w = float(math.exp(-self.wklambda * (k - 1)))
            elif decay == "None" or decay is None:
                w = 1.0
            else:
                w = 1.0 / (float(k) ** float(self.wkpower))
            raw[k] = max(w, float(self.wkmin))

        ks = sorted(raw.keys())
        vals = [raw[k] for k in ks]
        for k, v in zip(ks, reversed(vals)):
            wk[k] = v
        return wk

    def _compute_thresholds(self):
        alpha = self.alpha

        if alpha.dim() == 1:
            theta0 = alpha[0:1]
            if alpha.numel() > 1:
                incr = F.softplus(alpha[1:])
                theta = torch.cat([theta0, incr], dim=0).cumsum(dim=0)
            else:
                theta = theta0
            return theta
        else:
            theta0 = alpha[:, 0:1]
            if alpha.size(1) > 1:
                incr = F.softplus(alpha[:, 1:])
                theta = torch.cat([theta0, incr], dim=1).cumsum(dim=1)
            else:
                theta = theta0
            return theta

    def _compute_gains(self):
        if hasattr(self, "g_raw"):
            gains = []
            floor = float(getattr(self, "gains_floor", 0.0))
            cur = F.softplus(self.g_raw[0]) + floor
            gains.append(cur)
            for i in range(1, len(self.g_raw)):
                cur = cur + F.softplus(self.g_raw[i])
                gains.append(cur)
            return torch.stack(gains)
        return None

    def _class_probs_from_score(self, s, clusters_ids=None, departement_ids=None):
        theta = self._compute_thresholds().to(s.device)

        if theta.dim() == 1:
            Fk = torch.sigmoid(theta[None, :] - s[:, None])
        else:
            if self.alphatype == "cluster":
                if clusters_ids is None:
                    raise ValueError("clusters_ids is required when alphatype='cluster'")
                chosen_ids = clusters_ids.clamp(0, theta.shape[0] - 1)
            elif self.alphatype == "department":
                if departement_ids is None:
                    raise ValueError("departement_ids is required when alphatype='department'")
                chosen_ids = departement_ids.clamp(0, theta.shape[0] - 1)
            else:
                raise ValueError(f"Unknown alphatype: {self.alphatype}")

            thr = theta.index_select(0, chosen_ids.to(device=s.device).long())
            Fk = torch.sigmoid(thr - s[:, None])

        p = s.new_zeros((s.size(0), self.C))
        p[:, 0] = Fk[:, 0]
        if self.C > 2:
            p[:, 1:-1] = Fk[:, 1:] - Fk[:, :-1]
        p[:, -1] = 1.0 - Fk[:, -1]
        return p

    def _softmin(self, x):
        return -(1.0 / self.beta) * torch.logsumexp(-self.beta * x, dim=0)

    def _soft_median(self, deltas):
        alpha = 20.0
        if deltas.dim() == 1:
            c = deltas.mean()
            w = torch.softmax(-alpha * (deltas - c).abs(), dim=0)
            return (w * deltas).sum()
        else:
            c = deltas.mean(dim=0, keepdim=True)
            w = torch.softmax(-alpha * (deltas - c).abs(), dim=0)
            return (w * deltas).sum(dim=0)

            new_buf[:len(old_buf)] = old_buf
            self.register_buffer("coverage_update_count_departement", new_buf)
            
            if self.alphatype == "department":
                raise RuntimeError("Cannot dynamically resize buffer when alphatype='department' because alpha is an nn.Parameter.")

            self.ndepartements = new_size

    def _remap_ids(self, raw_ids: torch.Tensor, buf_size: int, kind: str):
        if raw_ids.dim() != 1:
            raw_ids = raw_ids.view(-1)

        raw_ids = raw_ids.long()
        device = raw_ids.device

        if kind == "cluster":
            raw_to_slot = self.cluster_raw_to_slot
            slot_to_raw = self.cluster_slot_to_raw
            next_free_attr = "cluster_next_free_slot"
        elif kind == "department":
            raw_to_slot = self.departement_raw_to_slot
            slot_to_raw = self.departement_slot_to_raw
            next_free_attr = "departement_next_free_slot"
        else:
            raise ValueError(f"Unknown kind: {kind}")

        if not raw_to_slot and (slot_to_raw != -1).any():
            max_slot = -1
            for slot_idx, raw_val in enumerate(slot_to_raw.tolist()):
                if raw_val != -1:
                    raw_to_slot[raw_val] = slot_idx
                    if slot_idx > max_slot:
                        max_slot = slot_idx
            setattr(self, next_free_attr, max_slot + 1)

        local_ids = torch.empty_like(raw_ids, dtype=torch.long, device=device)
        next_free_slot = getattr(self, next_free_attr)

        for i in range(raw_ids.numel()):
            rid = int(raw_ids[i].item())

            if rid in raw_to_slot:
                slot = raw_to_slot[rid]
            else:
                if next_free_slot >= buf_size:
                    raise ValueError(
                        f"No free slot left for kind='{kind}'. "
                        f"Encountered new raw id {rid}, but buf_size={buf_size}."
                    )

                slot = next_free_slot
                raw_to_slot[rid] = slot
                slot_to_raw[slot] = rid
                next_free_slot += 1

            local_ids[i] = slot

        setattr(self, next_free_attr, next_free_slot)

        valid_mask = torch.ones_like(raw_ids, dtype=torch.bool, device=device)
        return slot_to_raw.clone(), local_ids, valid_mask
    
    def _group_centers_from_weights(self, y, weights, group_ids_local, Z):
        """
        Compute robust group-wise centers from weights.

        Parameters
        ----------
        y : (N,)
        weights : (N,)
        group_ids_local : (N,) in [0..Z-1]
        Z : int
            Number of active groups

        Returns
        -------
        m_k : (Z,)
            Effective mass per group
        mu_hat : (Z,)
            Robust center per group
        """
        """device = y.device
        dtype = y.dtype
        eps = self.eps
        alpha = 20.0

        m_k = torch.zeros(Z, device=device, dtype=dtype)
        m_k.scatter_add_(0, group_ids_local, weights)

        wy = weights * y
        sum_wy = torch.zeros(Z, device=device, dtype=dtype)
        sum_wy.scatter_add_(0, group_ids_local, wy)
        c_loc = sum_wy / m_k.clamp_min(eps)

        logits = -alpha * (y - c_loc[group_ids_local]).abs()

        max_per_group = torch.full((Z,), -float("inf"), device=device, dtype=dtype)
        max_per_group.scatter_reduce_(0, group_ids_local, logits, reduce="amax", include_self=True)

        logits_shift = logits - max_per_group[group_ids_local]
        exp_logits = torch.exp(logits_shift)

        den = torch.zeros(Z, device=device, dtype=dtype)
        den.scatter_add_(0, group_ids_local, exp_logits)
        w_loc = exp_logits / den[group_ids_local].clamp_min(eps)

        mu_hat = torch.zeros(Z, device=device, dtype=dtype)
        mu_hat.scatter_add_(0, group_ids_local, w_loc * y)"""
        
    def _group_centers_from_weights(self, y, weights, group_ids_local, Z):
        device = y.device
        dtype = y.dtype
        eps = self.eps

        m_k = torch.zeros(Z, device=device, dtype=dtype)
        m_k.scatter_add_(0, group_ids_local, weights)

        wy = weights * y
        sum_wy = torch.zeros(Z, device=device, dtype=dtype)
        sum_wy.scatter_add_(0, group_ids_local, wy)

        mu_hat = torch.full((Z,), float('nan'), device=device, dtype=dtype)
        valid = m_k > eps
        mu_hat[valid] = sum_wy[valid] / m_k[valid]

        return m_k, mu_hat
    
    def _mu_soft(
        self,
        p,
        y_cont,
        clusters_ids_local,
        current_epoch,
        departement_ids_local=None,
        sw=None,
        active_cluster_slots=None,
        active_dept_slots=None,
    ):
        y = y_cont.to(dtype=p.dtype)
        device = p.device

        if self.mu_prior.device != device:
            self.mu_prior = self.mu_prior.to(device)
        if self.mu_prior_global.device != device:
            self.mu_prior_global = self.mu_prior_global.to(device)
        if hasattr(self, "mu_prior_departement") and self.mu_prior_departement.device != device:
            self.mu_prior_departement = self.mu_prior_departement.to(device)

        # ---- Log raw soft mass BEFORE gate ----
        if not hasattr(self, "epoch_stats"):
            self.epoch_stats = {}
        if "p_raw_mean" not in self.epoch_stats:
            self.epoch_stats["p_raw_mean"] = []
        self.epoch_stats["p_raw_mean"].append(p.detach().mean(dim=0).cpu().numpy())

        gate = torch.sigmoid((p - self.taugate) / max(self.gatetemp, 1e-6))
        p = p * gate

        # ---- Log gated soft mass AFTER gate ----
        if "p_gated_mean" not in self.epoch_stats:
            self.epoch_stats["p_gated_mean"] = []
        self.epoch_stats["p_gated_mean"].append(p.detach().mean(dim=0).cpu().numpy())

        gamma = getattr(self, "gamma", 1.0)
        if gamma != 1.0:
            p = p.clamp_min(self.eps).pow(gamma)
            p = p / p.sum(dim=1, keepdim=True).clamp_min(self.eps)

        Zc = len(active_cluster_slots) if active_cluster_slots is not None else (
            int(clusters_ids_local.max().item()) + 1 if clusters_ids_local.numel() > 0 else 1
        )

        Zd = len(active_dept_slots) if active_dept_slots is not None else (
            int(departement_ids_local.max().item()) + 1
            if departement_ids_local is not None and departement_ids_local.numel() > 0
            else 0
        )

        mu_clusters = torch.zeros(Zc, self.C, device=device, dtype=p.dtype)
        mass_clusters = torch.zeros(Zc, self.C, device=device, dtype=p.dtype)

        mu_departements = torch.zeros(Zd, self.C, device=device, dtype=p.dtype) if Zd > 0 else None
        mass_departements = torch.zeros(Zd, self.C, device=device, dtype=p.dtype) if Zd > 0 else None

        warmup = current_epoch < getattr(self, "warmupes", 0)

        lambda_g = 0.0 if warmup else self.mu_lambda_g
        lambda_c = 0.0 if warmup else self.mu_lambda_c
        lambda_d = 0.0 if warmup else self.mu_lambda_d
        
        #print('global', lambda_g)
        #print('cluster', lambda_c)
        #print('departement', lambda_d)
        
        min_mass_update = self.massupdate
        
        # Mapping local cluster -> local department
        if Zd > 0 and departement_ids_local is not None:
            cluster_to_dept_local = torch.empty(Zc, dtype=torch.long, device=device)
            for z in range(Zc):
                dept_z = torch.unique(departement_ids_local[clusters_ids_local == z])

                if dept_z.numel() == 0:
                    raise ValueError(
                        f"Empty cluster {z}: no department associated with this cluster in the batch."
                    )

                if dept_z.numel() > 1:
                    raise ValueError(
                        f"Cluster {z} is associated with multiple departments in the same batch: "
                        f"{dept_z.tolist()}"
                    )

                cluster_to_dept_local[z] = dept_z[0]
        else:
            cluster_to_dept_local = None

        for k in range(self.C):
            pk = p[:, k]
            if sw is not None:
                swk = sw.to(device=device, dtype=p.dtype).clamp_min(self.eps)
                weights = pk * swk
            else:
                weights = pk

            # ---------------------------
            # Global prior
            # ---------------------------
            m_k_global = weights.sum()
            if m_k_global > 0:
                mu_hat_k_global = (weights * y).sum() / m_k_global.clamp_min(self.eps)
            else:
                mu_hat_k_global = y.mean()

            with torch.no_grad():
                if not warmup:
                    if not torch.isfinite(self.mu_prior_global[k]):
                        self.mu_prior_global[k] = mu_hat_k_global.detach()
                    elif m_k_global > min_mass_update:
                        self.mu_prior_global[k] = (
                            self.mu_momentum * self.mu_prior_global[k]
                            + (1.0 - self.mu_momentum) * mu_hat_k_global.detach()
                        )

            prior_global_k = self.mu_prior_global[k].to(device=device, dtype=p.dtype)

            # ---------------------------
            # Department-level mu
            # ---------------------------
            if Zd > 0 and departement_ids_local is not None:
                m_k_d, mu_hat_d = self._group_centers_from_weights(
                    y=y,
                    weights=weights,
                    group_ids_local=departement_ids_local,
                    Z=Zd,
                )
                mass_departements[:, k] = m_k_d.detach()

                valid_d = m_k_d > min_mass_update
                
                with torch.no_grad():
                    if not warmup and active_dept_slots is not None:
                        for li, slot_t in enumerate(active_dept_slots):
                            slot = int(slot_t.item())
                            old_val = self.mu_prior_departement[slot, k]
                            if not torch.isfinite(old_val):
                                self.mu_prior_departement[slot, k] = mu_hat_d[li].detach()
                            elif valid_d[li]:
                                self.mu_prior_departement[slot, k] = (
                                    self.mu_momentum * old_val
                                    + (1.0 - self.mu_momentum) * mu_hat_d[li].detach()
                                )
                if active_dept_slots is not None:
                    active_dept_slots = active_dept_slots.to(device)
                    prior_dept_k = self.mu_prior_departement[active_dept_slots, k].to(device=device, dtype=p.dtype)
                    prior_dept_k = torch.where(
                        torch.isfinite(prior_dept_k),
                        prior_dept_k,
                        torch.full_like(prior_dept_k, prior_global_k),
                    )
                else:
                    prior_dept_k = torch.full((Zd,), prior_global_k, device=device, dtype=p.dtype)

                if warmup:
                    mu_k_d = torch.where(
                        m_k_d <= self.eps,
                        torch.tensor(float('nan'), device=device, dtype=p.dtype),
                        mu_hat_d
                    )
                else:
                    mu_hat_d_safe = torch.nan_to_num(mu_hat_d, nan=0.0)
                    mu_k_d = torch.where(
                        m_k_d <= self.eps,
                        prior_dept_k,
                        (m_k_d * mu_hat_d_safe + lambda_d * prior_dept_k + lambda_g * prior_global_k)
                        / (m_k_d + lambda_d + lambda_g),
                    )
                mu_departements[:, k] = mu_k_d
            else:
                prior_dept_k = None

            # ---------------------------
            # Cluster-level mu
            # ---------------------------
            m_k_c, mu_hat_c = self._group_centers_from_weights(
                y=y,
                weights=weights,
                group_ids_local=clusters_ids_local,
                Z=Zc,
            )
            mass_clusters[:, k] = m_k_c.detach()

            valid_c = m_k_c > min_mass_update

            with torch.no_grad():
                if not warmup and active_cluster_slots is not None:
                    for li, slot_t in enumerate(active_cluster_slots):
                        slot = int(slot_t.item())
                        old_val = self.mu_prior[slot, k]
                        if not torch.isfinite(old_val):
                            self.mu_prior[slot, k] = mu_hat_c[li].detach()
                        elif valid_c[li]:
                            self.mu_prior[slot, k] = (
                                self.mu_momentum * old_val
                                + (1.0 - self.mu_momentum) * mu_hat_c[li].detach()
                            )
                            
            if cluster_to_dept_local is not None and prior_dept_k is not None:
                dept_prior_for_cluster = prior_dept_k.index_select(0, cluster_to_dept_local)
            else:
                dept_prior_for_cluster = torch.full(
                    (Zc,),
                    prior_global_k,
                    device=device,
                    dtype=p.dtype,
                )

            if active_cluster_slots is not None:
                active_cluster_slots = active_cluster_slots.to(device)
                prior_cluster_k = self.mu_prior[active_cluster_slots, k].to(device=device, dtype=p.dtype)
                prior_cluster_k = torch.where(
                    torch.isfinite(prior_cluster_k),
                    prior_cluster_k,
                    dept_prior_for_cluster,
                )
            else:
                prior_cluster_k = dept_prior_for_cluster

            if warmup:
                mu_k_c = torch.where(
                    m_k_c <= self.eps,
                    torch.tensor(float('nan'), device=device, dtype=p.dtype),
                    mu_hat_c
                )
            else:
                mu_hat_c_safe = torch.nan_to_num(mu_hat_c, nan=0.0)
                if prior_dept_k is not None and cluster_to_dept_local is not None:
                    prior_dept_for_cluster = prior_dept_k.index_select(0, cluster_to_dept_local)
                    mu_k_c = torch.where(
                        m_k_c <= self.eps,
                        prior_cluster_k,
                        (m_k_c * mu_hat_c_safe + lambda_c * prior_cluster_k + lambda_d * prior_dept_for_cluster + lambda_g * prior_global_k)
                        / (m_k_c + lambda_c + lambda_d + lambda_g),
                    )
                else:
                    mu_k_c = torch.where(
                        m_k_c <= self.eps,
                        prior_cluster_k,
                        (m_k_c * mu_hat_c_safe + lambda_c * prior_cluster_k + lambda_g * prior_global_k)
                        / (m_k_c + lambda_c + lambda_g),
                    )

            mu_clusters[:, k] = mu_k_c

        return mu_clusters, mass_clusters, mu_departements, mass_departements
    
    def _score_centers_soft(
        self,
        scores,
        p,
        clusters_ids_local,
        departement_ids_local=None,
        sw=None,
        active_cluster_slots=None,
        active_dept_slots=None,
    ):
        """
        Compute class centers in score space (predicted score s), not in target space.

        Returns
        -------
        score_clusters : (Zc, C)
            Soft class centers per cluster in score space.
        mass_clusters : (Zc, C)
            Effective mass per cluster and class.
        score_departements : (Zd, C) or None
            Soft class centers per department in score space.
        mass_departements : (Zd, C) or None
            Effective mass per department and class.
        """
        scores = scores.view(-1).to(dtype=p.dtype, device=p.device)
        device = p.device

        gate = torch.sigmoid((p - self.taugate) / max(self.gatetemp, 1e-6))
        p_eff = p * gate

        gamma = getattr(self, "gamma", 1.0)
        if gamma != 1.0:
            p_eff = p_eff.clamp_min(self.eps).pow(gamma)
            p_eff = p_eff / p_eff.sum(dim=1, keepdim=True).clamp_min(self.eps)

        Zc = len(active_cluster_slots) if active_cluster_slots is not None else (
            int(clusters_ids_local.max().item()) + 1 if clusters_ids_local.numel() > 0 else 1
        )

        Zd = len(active_dept_slots) if active_dept_slots is not None else (
            int(departement_ids_local.max().item()) + 1
            if departement_ids_local is not None and departement_ids_local.numel() > 0
            else 0
        )
        
        score_clusters = torch.zeros(Zc, self.C, device=device, dtype=p.dtype)
        mass_clusters = torch.zeros(Zc, self.C, device=device, dtype=p.dtype)

        score_departements = (
            torch.zeros(Zd, self.C, device=device, dtype=p.dtype) if Zd > 0 else None
        )                        

        mass_departements = (
            torch.zeros(Zd, self.C, device=device, dtype=p.dtype) if Zd > 0 else None
        )

        sw_eff = sw.to(device=device, dtype=p.dtype).clamp_min(self.eps) if sw is not None else None

        for k in range(self.C):
            weights = p_eff[:, k]
            if sw_eff is not None:
                weights = weights * sw_eff

            m_k_c, s_hat_c = self._group_centers_from_weights(
                y=scores,
                weights=weights,
                group_ids_local=clusters_ids_local,
                Z=Zc,
            )
            score_clusters[:, k] = s_hat_c
            mass_clusters[:, k] = m_k_c

            if Zd > 0 and departement_ids_local is not None:
                m_k_d, s_hat_d = self._group_centers_from_weights(
                    y=scores,
                    weights=weights,
                    group_ids_local=departement_ids_local,
                    Z=Zd,
                )
                score_departements[:, k] = s_hat_d
                mass_departements[:, k] = m_k_d

        return score_clusters, mass_clusters, score_departements, mass_departements
            
    def _loss_mid(self, mu_ref: torch.Tensor, theta_ref: torch.Tensor) -> torch.Tensor:
        """
        Align each threshold theta_k with the midpoint between consecutive centers:
            theta_k ~ 0.5 * (mu_k + mu_{k+1})

        Parameters
        ----------
        mu_ref : tensor (..., C)
            Class centers to use for the alignment.
            Typical choice: mu_active or self.mu_prior[active_slots].
        theta_ref : tensor (..., C-1)
            Threshold rows aligned with mu_ref.
            Must have the same leading dimension as mu_ref, or be broadcastable.

        Returns
        -------
        scalar tensor
        """
        if mu_ref.dim() == 1:
            mu_ref = mu_ref.unsqueeze(0)          # (1, C)
        if theta_ref.dim() == 1:
            theta_ref = theta_ref.unsqueeze(0)    # (1, C-1)

        if mu_ref.shape[-1] != self.C:
            raise ValueError(f"mu_ref last dim must be {self.C}, got {mu_ref.shape}")
        if theta_ref.shape[-1] != self.C - 1:
            raise ValueError(f"theta_ref last dim must be {self.C - 1}, got {theta_ref.shape}")

        # Broadcast if one side has one row
        if theta_ref.shape[0] == 1 and mu_ref.shape[0] > 1:
            theta_ref = theta_ref.expand(mu_ref.shape[0], -1)
        elif mu_ref.shape[0] == 1 and theta_ref.shape[0] > 1:
            mu_ref = mu_ref.expand(theta_ref.shape[0], -1)
            
        if mu_ref.shape[0] != theta_ref.shape[0]:
            raise ValueError(
                f"mu_ref and theta_ref are not aligned: {mu_ref.shape} vs {theta_ref.shape}"
            )

        target_mid = 0.5 * (mu_ref[:, :-1] + mu_ref[:, 1:])
        target_mid = target_mid.clone().detach()

        # Important: detach mu so Lmid mainly calibrates thresholds/bins
        if getattr(self, "mid_detach_mu", True):
            target_mid = target_mid.detach()

        valid = torch.isfinite(target_mid) & torch.isfinite(theta_ref)
        if not valid.any():
            return theta_ref.new_tensor(0.0)

        # Robust regression, better than plain MSE if some centers move abruptly
        return F.smooth_l1_loss(theta_ref[valid], target_mid[valid], reduction="mean")

    def forward(self, score, y_cont, clusters_ids, departement_ids, current_epoch, sample_weight=None):
        s = score.view(-1)
        y = y_cont.view(-1).to(device=s.device)
        
        if s.numel() == 0:
            zero = s.sum() * 0.0
            return {
                "total_loss": zero,
                "trans": zero,
                "coverage": zero,
                "mu0_term": zero,
                "lmid": zero,
            }

        s = s / self.sigma
        y = y / self.sigma

        check_finite("score", s)
        check_finite("y_cont", y_cont)

        clusters_ids = clusters_ids.view(-1).long().to(device=s.device)
        if departement_ids is not None:
            departement_ids = departement_ids.view(-1).long().to(device=s.device)

        sw = sample_weight.view(-1).to(device=s.device) if sample_weight is not None else None
        if sw is not None:
            check_finite("sample_weight", sw)
            assert (sw >= 0).all(), "Negative sample_weight detected"

        cluster_slot_to_raw, cluster_slot_ids, c_valid = self._remap_ids(
            clusters_ids, self.nclusters, kind="cluster"
        )

        if departement_ids is not None:
            dept_slot_to_raw, dept_slot_ids, d_valid = self._remap_ids(
                departement_ids, self.ndepartements, kind="department"
            )
        else:
            dept_slot_to_raw = dept_slot_ids = d_valid = None

        device = s.device

        probs = self._class_probs_from_score(
            s,
            clusters_ids=cluster_slot_ids if self.alphatype == "cluster" else None,
            departement_ids=dept_slot_ids if self.alphatype == "department" else None,
        )
        check_finite("probs", probs)
        
        probs = torch.nan_to_num(probs, nan=self.eps, posinf=1.0, neginf=0.0)
        probs = probs.clamp_min(0.0)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(self.eps)

        active_cluster_slots, local_cluster_ids = torch.unique(cluster_slot_ids, return_inverse=True)

        if dept_slot_ids is not None:
            active_dept_slots, local_dept_ids = torch.unique(dept_slot_ids, return_inverse=True)
        else:
            active_dept_slots = local_dept_ids = None

        mu, mass, mu_departement, mass_departement = self._mu_soft(
            probs,
            y,
            local_cluster_ids,
            current_epoch,
            departement_ids_local=local_dept_ids,
            sw=sw,
            active_cluster_slots=active_cluster_slots,
            active_dept_slots=active_dept_slots,
        )
        warmup = current_epoch < getattr(self, "warmupes", 0)
        if not warmup:
            check_finite("mu", mu)
        check_finite("mass", mass)
        if not warmup:
            check_finite("mu_prior", self.mu_prior_global)

        gains = self._compute_gains()
        active_local_clusters = torch.arange(len(active_cluster_slots), device=device)

        loss = s.new_tensor(0.0)
        wsum = s.new_tensor(0.0)
        
        mu_active = mu[active_local_clusters]
        mass_active = mass[active_local_clusters]
        if not warmup:
            check_finite("mu_active", mu_active)
        
        theta_all = self._compute_thresholds().to(device)

        # --------------------------------------------------
        # Score-space centers for Lmid
        # theta_k is pushed toward 0.5 * (s_k + s_{k+1})
        # --------------------------------------------------
        score_centers_cluster, score_mass_cluster, score_centers_dept, score_mass_dept = self._score_centers_soft(
            s,
            probs,
            local_cluster_ids,
            departement_ids_local=local_dept_ids,
            sw=sw,
            active_cluster_slots=active_cluster_slots,
            active_dept_slots=active_dept_slots,
        )

        if theta_all.dim() == 1:
            # global thresholds -> target = average class-score centers over active clusters
            score_ref = score_centers_cluster.mean(dim=0, keepdim=True)
            theta_mid = theta_all.unsqueeze(0)

        elif self.alphatype == "department":
            theta_mid = theta_all.index_select(0, active_dept_slots)

            if score_centers_dept is not None and score_centers_dept.shape[0] > 0:
                score_ref = score_centers_dept
            else:
                score_ref = score_centers_cluster.mean(dim=0, keepdim=True).expand(theta_mid.shape[0], -1)

        elif self.alphatype == "cluster":
            theta_mid = theta_all.index_select(0, active_cluster_slots)
            score_ref = score_centers_cluster

        else:
            raise ValueError(f"Unknown alphatype: {self.alphatype}")

        Lmid = self._loss_mid(score_ref, theta_mid)

        num_active = len(active_local_clusters)
        loss_accum = torch.zeros(num_active, device=device, dtype=s.dtype)
        waccum = 0.0

        tau = float(getattr(self, "tau_loss", 1.5))
        beta = float(getattr(self, "loss_ema_momentum", 0.99))

        loss_ema_dev = self.loss_ema.to(device=device)
        ema_active = loss_ema_dev[active_cluster_slots.clamp(0, loss_ema_dev.shape[0] - 1)].to(dtype=mu_active.dtype)
        w_viol = torch.softmax(tau * ema_active, dim=0)
        w_viol = w_viol.detach() + self.eps

        for k, pairs in self.P.items():
            raw = torch.stack([mu_active[:, b] - mu_active[:, a] for (a, b) in pairs], dim=0).to(s.device)
            margins = torch.stack([gains[a:b].sum() for (a, b) in pairs], dim=0).to(device)
            deltas = raw - margins.unsqueeze(1)
            
            MEDk = s.new_tensor(0.0)
            MINk = s.new_tensor(0.0)

            loss_med = 0.0
            loss_min = 0.0
            
            valid_mask = torch.isfinite(deltas)
            deltas_safe = torch.where(valid_mask, deltas, torch.zeros_like(deltas))
            
            # Évaluation type RankNet : BCEWithLogitsLoss(deltas, target=1)
            # deltas > 0 signifie que mu_active_b > mu_active_a + margin
            target_ones = torch.ones_like(deltas_safe)
            loss_neg_matrix = F.binary_cross_entropy_with_logits(deltas_safe, target_ones, reduction='none')
            
            loss_neg_matrix = torch.where(valid_mask, loss_neg_matrix, torch.zeros_like(loss_neg_matrix))
            
            valid_count = valid_mask.sum(dim=0).clamp_min(1.0)
            Lk_clusters = loss_neg_matrix.sum(dim=0) / valid_count

            check_finite("Lk_clusters", Lk_clusters)
            check_finite("MEDk", MEDk)
            check_finite("MINk", MINk)

            w = float(self.wk.get(k, 1.0))
            loss_accum = loss_accum + w * Lk_clusters.detach()
            waccum = waccum + w

            # Lk = (Lk_clusters * w_viol).sum() / w_viol.sum()
            Lk = Lk_clusters.mean()  # Commented out w_viol for now per user request

            if not hasattr(self, "epoch_stats"):
                self.epoch_stats = {}
            if "deltas" not in self.epoch_stats:
                self.epoch_stats["deltas"] = {}
            if k not in self.epoch_stats["deltas"]:
                self.epoch_stats["deltas"][k] = {"median": [], "min": [], "viol": [], "neg": []}

            self.epoch_stats["deltas"][k]["median"].append(MEDk.mean().item())
            self.epoch_stats["deltas"][k]["min"].append(MINk.mean().item())
            self.epoch_stats["deltas"][k]["viol"].append((deltas < 0).float().mean().item())
            self.epoch_stats["deltas"][k]["neg"].append(Lk_clusters.mean().item())

            loss = loss + w * Lk
            wsum = wsum + w

        if waccum > 0:
            loss_mean_per_cluster = loss_accum / waccum
            check_finite("loss_mean_per_cluster", loss_mean_per_cluster)
            with torch.no_grad():
                if self.loss_ema.device != device:
                    self.loss_ema = self.loss_ema.to(device=device)
                for ci, slot_t in enumerate(active_cluster_slots):
                    slot = int(slot_t.item())
                    if 0 <= slot < self.loss_ema.shape[0]:
                        self.loss_ema[slot] = beta * self.loss_ema[slot] + (1.0 - beta) * loss_mean_per_cluster[ci]

        w_mu = mass_active.sum(dim=1).clamp_min(1e-6)
        w_mu_sum = w_mu.sum().clamp_min(1e-6)
        mu_log = (mu_active * w_mu.unsqueeze(1)).sum(dim=0) / w_mu_sum
        if "mu" not in self.epoch_stats:
            self.epoch_stats["mu"] = []
        self.epoch_stats["mu"].append(mu_log.detach().cpu().numpy())

        if not hasattr(self, "epoch_stats"):
            self.epoch_stats = {}
        if "cluster_weights" not in self.epoch_stats:
            self.epoch_stats["cluster_weights"] = []
        cw_full = torch.zeros(self.loss_ema.shape[0], dtype=w_viol.dtype, device=device)
        for ci, slot_t in enumerate(active_cluster_slots):
            slot = int(slot_t.item())
            if 0 <= slot < cw_full.shape[0]:
                cw_full[slot] = w_viol[ci].to(device=device)
        self.epoch_stats["cluster_weights"].append(cw_full.detach().cpu().numpy())

        if "mass_active" not in self.epoch_stats:
            self.epoch_stats["mass_active"] = []
        ma_full = torch.zeros(self.mu_prior.shape[0], self.C, device=device, dtype=mass_active.dtype)
        for ci, slot_t in enumerate(active_cluster_slots):
            slot = int(slot_t.item())
            ma_full[slot] = mass_active[ci].to(device=device).detach()
        self.epoch_stats["mass_active"].append(ma_full.detach().cpu().numpy())

        transition_loss = loss / wsum.clamp_min(1e-6)
        
        # --------------------------------------------------
        # Coverage loss à la place de la focal loss
        # --------------------------------------------------
        coverage_loss = self._coverage_loss(
            probs=probs,
            cluster_slot_ids=cluster_slot_ids,
            dept_slot_ids=dept_slot_ids,
            sample_weight=sw,
        )

        check_finite("coverage_loss", coverage_loss)

        if "coverage" not in self.epoch_stats:
            self.epoch_stats["coverage"] = []
        self.epoch_stats["coverage"].append(float(coverage_loss.detach().cpu().item()))
        if "transition" not in self.epoch_stats:
            self.epoch_stats["transition"] = []
        self.epoch_stats["transition"].append(transition_loss.item())
        
        if "mid" not in self.epoch_stats:
            self.epoch_stats["mid"] = []
        self.epoch_stats["mid"].append(Lmid.item())

        mu0_val = mu_log[0]
        if not torch.isfinite(mu0_val):
            mu0_val = mu0_val.new_tensor(0.0)
        else:
            mu0_val = mu0_val.clamp(-100.0, 100.0)

        mu0_term = F.softplus(mu0_val)
        
        if "mu0_term" not in self.epoch_stats:
            self.epoch_stats["mu0_term"] = []
        self.epoch_stats["mu0_term"].append(mu0_term.item())

        check_finite("transition_loss", transition_loss)
        check_finite("focal_loss", coverage_loss)
        check_finite("mu0_term", mu0_term)
        check_finite("lmid", Lmid)

        try:
            
            total_loss = \
                self.wtrans * transition_loss \
                + self.wcoverage * coverage_loss \
                + self.wmu0 * mu0_term \
                + self.wmid * Lmid
                
        except Exception as e:
            print("DEBUG NAN SOURCE:")
            print("score:", s)
            print("probs:", probs)
            print("mu:", mu)
            print("scale:", self.delta_scale_ema)
            raise e

        return {
                "total_loss": total_loss,
                "trans": transition_loss,
                "coverage": coverage_loss,
                "mu0_term": mu0_term,
                "lmid": Lmid,
            }

    def get_learnable_parameters(self):
        params = {"alpha": self.alpha}

        if getattr(self, "learn_gains", False):
            if hasattr(self, "g_raw"):
                params["g_raw"] = self.g_raw
            elif hasattr(self, "gain_raw"):
                params["gain_raw"] = self.gain_raw
        return params

    @torch.no_grad()
    def score_to_class(self, scores: torch.Tensor, clusters_ids: torch.Tensor = None, departement_ids: torch.Tensor = None) -> torch.Tensor:
        s = scores.detach().to(dtype=self.alpha.dtype).flatten().unsqueeze(1)
        s = s / self.sigma
        
        device = s.device

        thr = self._compute_thresholds().detach().to(device=device)

        if thr.dim() == 1:
            return torch.bucketize(scores.flatten(), thr, right=True)
        else:
            if self.alphatype == "cluster":
                chosen_ids = clusters_ids
            elif self.alphatype == "department":
                chosen_ids = departement_ids
            else:
                chosen_ids = (clusters_ids if clusters_ids is not None else departement_ids)

            if chosen_ids is None:
                raise ValueError("IDs are required when thresholds are cluster/department-specific.")
            else:
                chosen_ids = chosen_ids.view(-1).long().to(device=device)
                if self.alphatype == "cluster":
                    _, idx, _ = self._remap_ids(chosen_ids, self.nclusters, kind="cluster")
                elif self.alphatype == "department":
                    _, idx, _ = self._remap_ids(chosen_ids, self.ndepartements, kind="department")
                else:
                    raise ValueError(f"Unknown alphatype: {self.alphatype}")

            thr_s = thr.index_select(0, idx)
            return (s > thr_s).sum(dim=1)

    def get_attribute(self):
        payload = {
            "alpha": self.alpha.detach().cpu().numpy(),
            "thresholds": self._compute_thresholds().detach().cpu().numpy(),
            "mu_prior": self.mu_prior.detach().cpu().numpy(),
            "mu_prior_global": self.mu_prior_global.detach().cpu().numpy(),
            "cluster_slot_to_raw": self.cluster_slot_to_raw.detach().cpu().numpy(),
            "departement_slot_to_raw": self.departement_slot_to_raw.detach().cpu().numpy(),
            "mu_prior_departement": self.mu_prior_departement.detach().cpu().numpy(),
        }

        if getattr(self, "learn_gains", False):
            g = self._compute_gains().detach().cpu().numpy() if hasattr(self, "_compute_gains") else None
            if g is None:
                g = self.gains.detach().cpu().numpy() if hasattr(self, "gains") and self.gains is not None else None
            if g is None and hasattr(self, "g_raw"):
                floor = float(getattr(self, "gains_floor", 0.0))
                g = (F.softplus(self.g_raw) + floor).detach().cpu().numpy()
            if g is not None:
                payload["gains"] = g

        if hasattr(self, "epoch_stats") and "deltas" in self.epoch_stats:
            agg_deltas = {}
            for k, dstats in self.epoch_stats["deltas"].items():
                agg_deltas[k] = {
                    "median": np.mean(dstats["median"]) if dstats["median"] else 0.0,
                    "min": np.mean(dstats["min"]) if dstats["min"] else 0.0,
                    "viol": np.mean(dstats["viol"]) if dstats["viol"] else 0.0,
                    "neg": np.mean(dstats["neg"]) if dstats["neg"] else 0.0,
                }
            payload["deltas"] = agg_deltas

        if hasattr(self, "epoch_stats") and "mu" in self.epoch_stats and len(self.epoch_stats["mu"]) > 0:
            mu_stack = np.stack(self.epoch_stats["mu"])
            payload["mu"] = np.mean(mu_stack, axis=0)

        if hasattr(self, "epoch_stats"):
            for _lkey in ("transition", "coverage"):
                vals = self.epoch_stats.get(_lkey, [])
                if vals:
                    payload[_lkey] = [float(np.mean(vals))]

            vals = self.epoch_stats.get("mid", [])
            if vals:
                payload["mid"] = [float(np.mean(vals))]

        payload["delta_scale_ema"] = self.delta_scale_ema.detach().cpu().numpy()

        if hasattr(self, "epoch_stats") and self.epoch_stats.get("cluster_weights"):
            cw_stack = np.stack(self.epoch_stats["cluster_weights"])
            payload["cluster_weights"] = np.mean(cw_stack, axis=0)

        if hasattr(self, "epoch_stats") and self.epoch_stats.get("mass_active"):
            ma_stack = np.stack(self.epoch_stats["mass_active"])
            payload["mass_active"] = np.max(ma_stack, axis=0)

        if hasattr(self, "epoch_stats") and self.epoch_stats.get("p_raw_mean"):
            payload["p_raw_mean"] = np.mean(np.stack(self.epoch_stats["p_raw_mean"]), axis=0)

        if hasattr(self, "epoch_stats") and self.epoch_stats.get("p_gated_mean"):
            payload["p_gated_mean"] = np.mean(np.stack(self.epoch_stats["p_gated_mean"]), axis=0)

        return [("ordinal_params", DictWrapper(payload))]
    
        # ============================================================
    # Coverage target computation
    # ============================================================

    @staticmethod
    def _normalize_distribution_np(x, eps=1e-8):
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, 0.0, None)
        s = float(x.sum())
        if s <= eps:
            return np.ones_like(x, dtype=np.float64) / float(len(x))
        return x / s


    def _shift_distribution_np(self, dist, shift: float):
        """
        Décale une distribution ordinale vers la droite ou vers la gauche.

        shift > 0 : pousse la masse vers les classes plus hautes.
        shift < 0 : pousse la masse vers les classes plus basses.

        Exemple :
            shift=0.25 : 75% reste sur la classe c, 25% va vers c+1.
        """
        dist = self._normalize_distribution_np(dist, eps=self.eps)

        shifted = np.zeros_like(dist, dtype=np.float64)

        for c in range(self.C):
            mass = dist[c]
            pos = float(c) + float(shift)

            if pos <= 0.0:
                shifted[0] += mass

            elif pos >= float(self.C - 1):
                shifted[self.C - 1] += mass

            else:
                low = int(math.floor(pos))
                high = low + 1

                w_high = pos - float(low)
                w_low = 1.0 - w_high

                shifted[low] += mass * w_low
                shifted[high] += mass * w_high

        return self._normalize_distribution_np(shifted, eps=self.eps)


    def _shift_distribution_torch(self, dist: torch.Tensor, shift: float):
        """
        Version torch du shift ordinal, utilisée si besoin pendant le training.
        """
        dist = dist.clamp_min(0.0)
        dist = dist / dist.sum().clamp_min(self.eps)

        shifted = torch.zeros_like(dist)

        for c in range(self.C):
            mass = dist[c]
            pos = float(c) + float(shift)

            if pos <= 0.0:
                shifted[0] = shifted[0] + mass

            elif pos >= float(self.C - 1):
                shifted[self.C - 1] = shifted[self.C - 1] + mass

            else:
                low = int(math.floor(pos))
                high = low + 1

                w_high = pos - float(low)
                w_low = 1.0 - w_high

                shifted[low] = shifted[low] + mass * w_low
                shifted[high] = shifted[high] + mass * w_high

        shifted = shifted.clamp_min(0.0)
        shifted = shifted / shifted.sum().clamp_min(self.eps)

        return shifted


    def _register_cluster_slot_from_raw(self, raw_id: int) -> int:
        """
        Enregistre un cluster raw id dans les slots internes,
        de manière cohérente avec _remap_ids.
        """
        raw_id = int(raw_id)

        if raw_id in self.cluster_raw_to_slot:
            return self.cluster_raw_to_slot[raw_id]

        if self.cluster_next_free_slot >= self.nclusters:
            raise ValueError(
                f"No free cluster slot left for raw cluster id {raw_id}. "
                f"nclusters={self.nclusters}"
            )

        slot = self.cluster_next_free_slot
        self.cluster_raw_to_slot[raw_id] = slot
        self.cluster_slot_to_raw[slot] = raw_id
        self.cluster_next_free_slot += 1

        return slot


    def _register_departement_slot_from_raw(self, raw_id: int) -> int:
        """
        Enregistre un département raw id dans les slots internes,
        de manière cohérente avec _remap_ids.
        """
        raw_id = int(raw_id)

        if raw_id in self.departement_raw_to_slot:
            return self.departement_raw_to_slot[raw_id]

        if self.departement_next_free_slot >= self.ndepartements:
            raise ValueError(
                f"No free department slot left for raw department id {raw_id}. "
                f"ndepartements={self.ndepartements}"
            )

        slot = self.departement_next_free_slot
        self.departement_raw_to_slot[raw_id] = slot
        self.departement_slot_to_raw[slot] = raw_id
        self.departement_next_free_slot += 1

        return slot


    def calculate_class_coverage(
        self,
        df,
        target_col: str,
        cluster_col: str='cluster-encoder',
        departement_col: Optional[str] = 'departement',
        shrinkage: float = 30.0,
        reset: bool = True,
        dir_output=None,
    ):
        """
        Calcule les distributions cibles de coverage.

        Contrairement à ContextualOrdinalUncertaintyFocalWKLoss, cette fonction
        ne crée pas de nouveaux samples. Elle calcule seulement les histogrammes
        de classes cibles.

        Pour chaque groupe g :

            q_g(c) = P(y=c | g)

        avec shrinkage vers la distribution globale :

            q_g = (counts_g + shrinkage * q_global) / (n_g + shrinkage)

        Puis un shift ordinal optionnel est appliqué :

            shift > 0 : pousse la distribution vers les classes hautes.
            shift < 0 : pousse la distribution vers les classes basses.

        Paramètres
        ----------
        df:
            DataFrame d'entraînement.

        cluster_col:
            Colonne des clusters.

        target_col:
            Colonne cible contenant les classes 0..C-1.

        departement_col:
            Colonne des départements. Nécessaire si coverageagg='department'
            ou si tu veux aussi stocker les priors départementaux.

        shrinkage:
            Lissage vers la distribution globale.

        reset:
            Si True, réinitialise les buffers de coverage.

        Retour
        ------
        self
        """
        
        shift = self.shift

        if cluster_col not in df.columns:
            raise ValueError(f"Missing cluster_col={cluster_col}")

        if target_col not in df.columns:
            raise ValueError(f"Missing target_col={target_col}")

        if departement_col is not None and departement_col not in df.columns:
            raise ValueError(f"Missing departement_col={departement_col}")

        if self.coverageagg == "department" and departement_col is None:
            raise ValueError(
                "departement_col is required when coverageagg='department'"
            )

        d = df.copy()
        d = d[[c for c in [cluster_col, departement_col, target_col] if c is not None]].dropna()
        d[target_col] = d[target_col].astype(int)

        if reset:
            with torch.no_grad():
                self.coverage_target_global.fill_(float("nan"))
                self.coverage_target_cluster.fill_(float("nan"))
                self.coverage_target_departement.fill_(float("nan"))

                self.coverage_pred_ema_cluster.fill_(float("nan"))
                self.coverage_pred_ema_departement.fill_(float("nan"))
                self.coverage_pred_ema_global.fill_(float("nan"))

                self.coverage_update_count_cluster.zero_()
                self.coverage_update_count_departement.zero_()
                self.coverage_update_count_global.zero_()

        # ---------------------------
        # Global distribution
        # ---------------------------
        global_counts = (
            d[target_col]
            .value_counts()
            .reindex(range(self.C), fill_value=0)
            .sort_index()
            .values
            .astype(np.float64)
        )

        global_dist = self._normalize_distribution_np(global_counts, eps=self.eps)
        global_dist = self._shift_distribution_np(global_dist, shift=shift)

        with torch.no_grad():
            self.coverage_target_global.copy_(
                torch.tensor(
                    global_dist,
                    dtype=self.coverage_target_global.dtype,
                    device=self.coverage_target_global.device,
                )
            )

        d = d.loc[:,~d.columns.duplicated()].copy()

        # ---------------------------
        # Cluster distributions
        # ---------------------------
        for raw_cluster, dfg in d.groupby(cluster_col):
            
            slot = self._register_cluster_slot_from_raw(int(raw_cluster))

            counts = (
                dfg[target_col]
                .value_counts()
                .reindex(range(self.C), fill_value=0)
                .sort_index()
                .values
                .astype(np.float64)
            )

            if shrinkage > 0.0:
                q = (
                    counts + float(shrinkage) * global_dist
                ) / max(float(counts.sum()) + float(shrinkage), self.eps)
            else:
                q = self._normalize_distribution_np(counts, eps=self.eps)

            q = self._normalize_distribution_np(q, eps=self.eps)
            q = self._shift_distribution_np(q, shift=shift)

            with torch.no_grad():
                self.coverage_target_cluster[slot].copy_(
                    torch.tensor(
                        q,
                        dtype=self.coverage_target_cluster.dtype,
                        device=self.coverage_target_cluster.device,
                    )
                )

        # ---------------------------
        # Department distributions
        # ---------------------------
        if departement_col is not None:
            for raw_dept, dfd in d.groupby(departement_col):
                slot = self._register_departement_slot_from_raw(int(raw_dept))

                counts = (
                    dfd[target_col]
                    .value_counts()
                    .reindex(range(self.C), fill_value=0)
                    .sort_index()
                    .values
                    .astype(np.float64)
                )

                if shrinkage > 0.0:
                    q = (
                        counts + float(shrinkage) * global_dist
                    ) / max(float(counts.sum()) + float(shrinkage), self.eps)
                else:
                    q = self._normalize_distribution_np(counts, eps=self.eps)

                q = self._normalize_distribution_np(q, eps=self.eps)
                q = self._shift_distribution_np(q, shift=shift)

                with torch.no_grad():
                    self.coverage_target_departement[slot].copy_(
                        torch.tensor(
                            q,
                            dtype=self.coverage_target_departement.dtype,
                            device=self.coverage_target_departement.device,
                        )
                    )

        # ---------------------------
        # Optional CSV export
        # ---------------------------
        if dir_output is not None:
            os.makedirs(dir_output, exist_ok=True)

            with open(os.path.join(dir_output, "coverage_target_global.csv"), "w") as f:
                f.write("class,probability\n")
                for c, p in enumerate(global_dist):
                    f.write(f"{c},{p:.8f}\n")

            with open(os.path.join(dir_output, "coverage_target_cluster.csv"), "w") as f:
                f.write("slot,raw_cluster,class,probability\n")
                arr = self.coverage_target_cluster.detach().cpu().numpy()
                raw = self.cluster_slot_to_raw.detach().cpu().numpy()
                for slot in range(arr.shape[0]):
                    if raw[slot] < 0 or not np.isfinite(arr[slot]).all():
                        continue
                    for c in range(self.C):
                        f.write(f"{slot},{int(raw[slot])},{c},{arr[slot, c]:.8f}\n")

            if departement_col is not None:
                with open(os.path.join(dir_output, "coverage_target_departement.csv"), "w") as f:
                    f.write("slot,raw_departement,class,probability\n")
                    arr = self.coverage_target_departement.detach().cpu().numpy()
                    raw = self.departement_slot_to_raw.detach().cpu().numpy()
                    for slot in range(arr.shape[0]):
                        if raw[slot] < 0 or not np.isfinite(arr[slot]).all():
                            continue
                        for c in range(self.C):
                            f.write(f"{slot},{int(raw[slot])},{c},{arr[slot, c]:.8f}\n")
                            
            plot_dir = os.path.join(dir_output, "coverage_target_plots")
            os.makedirs(plot_dir, exist_ok=True)

            # ---------------------------
            # Global distribution
            # ---------------------------
            plt.figure(figsize=(8, 5))
            plt.bar(np.arange(self.C), global_dist)
            plt.xlabel("Class")
            plt.ylabel("Probability")
            plt.title("Coverage target distribution — global")
            plt.xticks(np.arange(self.C))
            plt.ylim(0.0, max(1.0, float(np.max(global_dist)) * 1.1))
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "coverage_target_global.png"), dpi=200)
            plt.close()

            # ---------------------------
            # Cluster distributions
            # ---------------------------
            arr = self.coverage_target_cluster.detach().cpu().numpy()
            raw = self.cluster_slot_to_raw.detach().cpu().numpy()

            valid_slots = [
                slot for slot in range(arr.shape[0])
                if raw[slot] >= 0 and np.isfinite(arr[slot]).all()
            ]

            if len(valid_slots) > 0:
                cluster_plot_dir = os.path.join(plot_dir, "clusters")
                os.makedirs(cluster_plot_dir, exist_ok=True)

                # Heatmap cluster x class
                mat = arr[valid_slots]
                y_labels = [str(int(raw[slot])) for slot in valid_slots]

                fig, ax = plt.subplots(
                    figsize=(8, max(4, 0.35 * len(valid_slots)))
                )
                im = ax.imshow(mat, aspect="auto")
                plt.colorbar(im, ax=ax, label="Probability")

                ax.set_xticks(np.arange(self.C))
                ax.set_xticklabels([str(c) for c in range(self.C)])
                ax.set_yticks(np.arange(len(valid_slots)))
                ax.set_yticklabels(y_labels)

                ax.set_xlabel("Class")
                ax.set_ylabel("Raw cluster")
                ax.set_title("Coverage target distributions — clusters")

                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        ax.text(
                            j,
                            i,
                            f"{mat[i, j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                        )

                plt.tight_layout()
                plt.savefig(
                    os.path.join(plot_dir, "coverage_target_cluster_heatmap.png"),
                    dpi=200,
                )
                plt.close()

                # One barplot per cluster
                for slot in valid_slots:
                    q = arr[slot]
                    raw_cluster = int(raw[slot])

                    plt.figure(figsize=(8, 5))
                    plt.bar(np.arange(self.C), q)
                    plt.xlabel("Class")
                    plt.ylabel("Probability")
                    plt.title(f"Coverage target distribution — cluster {raw_cluster}")
                    plt.xticks(np.arange(self.C))
                    plt.ylim(0.0, max(1.0, float(np.max(q)) * 1.1))
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            cluster_plot_dir,
                            f"coverage_target_cluster_{raw_cluster}.png",
                        ),
                        dpi=200,
                    )
                    plt.close()

            # ---------------------------
            # Department distributions
            # ---------------------------
            if departement_col is not None:
                arr = self.coverage_target_departement.detach().cpu().numpy()
                raw = self.departement_slot_to_raw.detach().cpu().numpy()

                valid_slots = [
                    slot for slot in range(arr.shape[0])
                    if raw[slot] >= 0 and np.isfinite(arr[slot]).all()
                ]

                if len(valid_slots) > 0:
                    dept_plot_dir = os.path.join(plot_dir, "departements")
                    os.makedirs(dept_plot_dir, exist_ok=True)

                    # Heatmap department x class
                    mat = arr[valid_slots]
                    y_labels = [str(int(raw[slot])) for slot in valid_slots]

                    fig, ax = plt.subplots(
                        figsize=(8, max(4, 0.35 * len(valid_slots)))
                    )
                    im = ax.imshow(mat, aspect="auto")
                    plt.colorbar(im, ax=ax, label="Probability")

                    ax.set_xticks(np.arange(self.C))
                    ax.set_xticklabels([str(c) for c in range(self.C)])
                    ax.set_yticks(np.arange(len(valid_slots)))
                    ax.set_yticklabels(y_labels)

                    ax.set_xlabel("Class")
                    ax.set_ylabel("Raw department")
                    ax.set_title("Coverage target distributions — departments")

                    for i in range(mat.shape[0]):
                        for j in range(mat.shape[1]):
                            ax.text(
                                j,
                                i,
                                f"{mat[i, j]:.2f}",
                                ha="center",
                                va="center",
                                fontsize=7,
                            )

                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            plot_dir,
                            "coverage_target_departement_heatmap.png",
                        ),
                        dpi=200,
                    )
                    plt.close()

                    # One barplot per department
                    for slot in valid_slots:
                        q = arr[slot]
                        raw_dept = int(raw[slot])

                        plt.figure(figsize=(8, 5))
                        plt.bar(np.arange(self.C), q)
                        plt.xlabel("Class")
                        plt.ylabel("Probability")
                        plt.title(
                            f"Coverage target distribution — department {raw_dept}"
                        )
                        plt.xticks(np.arange(self.C))
                        plt.ylim(0.0, max(1.0, float(np.max(q)) * 1.1))
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(
                                dept_plot_dir,
                                f"coverage_target_departement_{raw_dept}.png",
                            ),
                            dpi=200,
                        )
                        plt.close()

        return self

    def _coverage_distance(self, pred_dist: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
        """
        Distance entre distributions de classes.
        """
        pred_dist = pred_dist.clamp_min(0.0)
        target_dist = target_dist.clamp_min(0.0)

        pred_dist = pred_dist / pred_dist.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        if self.coverage_distance == "l2":
            return (pred_dist - target_dist).pow(2).mean(dim=-1)

        pred_cdf = torch.cumsum(pred_dist, dim=-1)
        target_cdf = torch.cumsum(target_dist, dim=-1)
        diff = pred_cdf - target_cdf

        if self.coverage_distance == "cdf_l2":
            return diff.pow(2).mean(dim=-1)

        if self.coverage_distance == "cdf_l1":
            return diff.abs().mean(dim=-1)

        if self.coverage_distance == "cdf_linf":
            return diff.abs().max(dim=-1).values

        raise ValueError(f"Unknown coverage_distance={self.coverage_distance}")


    def _get_valid_target_or_global(self, target_bank: torch.Tensor, slots: torch.Tensor):
        """
        Récupère les distributions cibles pour les slots actifs.
        Si une cible locale est absente, fallback vers coverage_target_global.
        """
        device = slots.device
        dtype = target_bank.dtype

        target_bank = target_bank.to(device=device)
        target = target_bank.index_select(0, slots.long())

        global_target = self.coverage_target_global.to(device=device, dtype=dtype)

        if not torch.isfinite(global_target).all():
            global_target = torch.ones(self.C, device=device, dtype=dtype) / float(self.C)

        global_target = global_target / global_target.sum().clamp_min(self.eps)

        invalid = ~torch.isfinite(target).all(dim=1)
        if invalid.any():
            target[invalid] = global_target.to(dtype=target.dtype)

        target = target.clamp_min(0.0)
        target = target / target.sum(dim=1, keepdim=True).clamp_min(self.eps)

        return target


    def _coverage_loss(
        self,
        probs: torch.Tensor,
        cluster_slot_ids: torch.Tensor,
        dept_slot_ids: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Coverage loss avec EMA.

        Pour chaque groupe g du batch :

            p_batch_g = mean_i p_i

            ema_hat_g = beta * EMA_old_g + (1-beta) * p_batch_g

            L_g = distance(ema_hat_g, q_g)

        où q_g vient de calculate_class_coverage.
        """

        device = probs.device
        dtype = probs.dtype

        probs = probs.clamp_min(0.0)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(self.eps)

        if sample_weight is not None:
            sw = sample_weight.view(-1).to(device=device, dtype=dtype).clamp_min(self.eps)
        else:
            sw = None

        beta = float(self.coverage_momentum)

        if self.coverageagg == "global":
            if sw is None:
                batch_dist = probs.mean(dim=0)
            else:
                batch_dist = (probs * sw[:, None]).sum(dim=0) / sw.sum().clamp_min(self.eps)

            batch_dist = batch_dist / batch_dist.sum().clamp_min(self.eps)

            old = self.coverage_pred_ema_global.to(device=device, dtype=dtype)

            if not torch.isfinite(old).all():
                old = self.coverage_target_global.to(device=device, dtype=dtype)
                if not torch.isfinite(old).all():
                    old = batch_dist.detach()

            old = old / old.sum().clamp_min(self.eps)

            ema_hat = beta * old.detach() + (1.0 - beta) * batch_dist
            ema_hat = ema_hat / ema_hat.sum().clamp_min(self.eps)

            target = self.coverage_target_global.to(device=device, dtype=dtype)
            if not torch.isfinite(target).all():
                target = batch_dist.detach()

            target = target / target.sum().clamp_min(self.eps)

            loss = self._coverage_distance(
                ema_hat.unsqueeze(0),
                target.unsqueeze(0),
            ).mean()

            if self.training:
                with torch.no_grad():
                    self.coverage_pred_ema_global.copy_(ema_hat.detach().cpu())
                    self.coverage_update_count_global += 1

            return loss

        if self.coverageagg == "cluster":
            group_ids = cluster_slot_ids.view(-1).long().to(device)
            target_bank = self.coverage_target_cluster
            ema_bank = self.coverage_pred_ema_cluster
            count_bank = self.coverage_update_count_cluster

        elif self.coverageagg == "department":
            if dept_slot_ids is None:
                raise ValueError("dept_slot_ids is required when coverageagg='department'")
            group_ids = dept_slot_ids.view(-1).long().to(device)
            target_bank = self.coverage_target_departement
            ema_bank = self.coverage_pred_ema_departement
            count_bank = self.coverage_update_count_departement

        else:
            raise ValueError(f"Unknown coverageagg={self.coverageagg}")

        active_slots = torch.unique(group_ids)
        losses = []
        ema_updates = []

        target_bank_dev = target_bank.to(device=device, dtype=dtype)
        ema_bank_dev = ema_bank.to(device=device, dtype=dtype)
        count_bank_dev = count_bank.to(device=device)

        targets = self._get_valid_target_or_global(target_bank_dev, active_slots)

        for local_idx, slot_t in enumerate(active_slots):
            slot = int(slot_t.item())
            mask = group_ids == slot_t

            if not mask.any():
                continue

            if sw is None:
                batch_dist = probs[mask].mean(dim=0)
            else:
                sw_g = sw[mask]
                batch_dist = (probs[mask] * sw_g[:, None]).sum(dim=0) / sw_g.sum().clamp_min(self.eps)

            batch_dist = batch_dist / batch_dist.sum().clamp_min(self.eps)

            old = ema_bank_dev[slot]

            if not torch.isfinite(old).all():
                old = targets[local_idx].detach()

            old = old / old.sum().clamp_min(self.eps)

            ema_hat = beta * old.detach() + (1.0 - beta) * batch_dist
            ema_hat = ema_hat / ema_hat.sum().clamp_min(self.eps)

            target = targets[local_idx]

            dist_loss = self._coverage_distance(
                ema_hat.unsqueeze(0),
                target.unsqueeze(0),
            ).mean()

            update_count = int(count_bank_dev[slot].item())
            if self.coverage_warmup_updates > 0:
                warmup = min(
                    1.0,
                    float(update_count + 1) / float(self.coverage_warmup_updates),
                )
            else:
                warmup = 1.0

            losses.append(dist_loss * probs.new_tensor(warmup))
            ema_updates.append((slot, ema_hat.detach()))

        if self.training and len(ema_updates) > 0:
            with torch.no_grad():
                if ema_bank.device != device:
                    ema_bank.data = ema_bank.data.to(device)
                if count_bank.device != device:
                    count_bank.data = count_bank.data.to(device)

                for slot, ema_hat in ema_updates:
                    ema_bank[slot].copy_(ema_hat)
                    count_bank[slot] += 1

        if len(losses) == 0:
            return probs.new_tensor(0.0)

        return torch.stack(losses).mean()

    def update_params(self, new_dict, epoch=None):
        """
        Accepte soit :
        1) directement un payload dict
        2) un DictWrapper(payload)
        3) un dict externe du type:
            {"epoch": ..., "ordinal_params": DictWrapper(payload)}
        et met à jour alpha, mu_prior, mu_prior_global si présents.
        """

        # --------------------------------------------------
        # 1) Déplier la structure externe
        # --------------------------------------------------
        payload = new_dict

        # Cas: {"epoch": ..., "ordinal_params": DictWrapper(...)}
        if isinstance(payload, dict) and "ordinal_params" in payload:
            if epoch is None and "epoch" in payload:
                epoch = payload["epoch"]
            payload = payload["ordinal_params"]

        # Cas: DictWrapper(payload)
        if hasattr(payload, "numpy") and not isinstance(payload, dict):
            payload = payload.numpy()

        # Sécurité finale
        if not isinstance(payload, dict):
            raise TypeError(
                f"update_params expected a dict-like payload after unwrapping, got {type(payload)}"
            )

        # --------------------------------------------------
        # 2) alpha
        # --------------------------------------------------
        print('Old alpha', self.alpha)
        if "alpha" in payload and payload["alpha"] is not None:
            alpha_new = torch.as_tensor(
                payload["alpha"],
                dtype=self.alpha.dtype,
                device=self.alpha.device,
            )
            if alpha_new.shape != self.alpha.shape:
                raise ValueError(
                    f"alpha shape mismatch: got {tuple(alpha_new.shape)}, "
                    f"expected {tuple(self.alpha.shape)}"
                )
            with torch.no_grad():
                self.alpha.copy_(alpha_new)
        else:
            if "alpha" not in payload or payload["alpha"] is None:
                raise ValueError(
                    f"alpha missing from payload, {payload}"
                )

        # --------------------------------------------------
        # 3) mu_prior
        # --------------------------------------------------
        if "mu_prior" in payload and payload["mu_prior"] is not None:
            mu_prior_new = torch.as_tensor(
                payload["mu_prior"],
                dtype=self.mu_prior.dtype,
                device=self.mu_prior.device,
            )
            if mu_prior_new.shape != self.mu_prior.shape:
                raise ValueError(
                    f"mu_prior shape mismatch: got {tuple(mu_prior_new.shape)}, "
                    f"expected {tuple(self.mu_prior.shape)}"
                )
            with torch.no_grad():
                self.mu_prior.copy_(mu_prior_new)

        # --------------------------------------------------
        # 4) mu_prior_global
        # --------------------------------------------------
        if "mu_prior_global" in payload and payload["mu_prior_global"] is not None:
            mu_prior_global_new = torch.as_tensor(
                payload["mu_prior_global"],
                dtype=self.mu_prior_global.dtype,
                device=self.mu_prior_global.device,
            )
            if mu_prior_global_new.shape != self.mu_prior_global.shape:
                raise ValueError(
                    f"mu_prior_global shape mismatch: got {tuple(mu_prior_global_new.shape)}, "
                    f"expected {tuple(self.mu_prior_global.shape)}"
                )
            with torch.no_grad():
                self.mu_prior_global.copy_(mu_prior_global_new)
                
        # --------------------------------------------------
        # 4) mu_prior_departement
        # --------------------------------------------------
        if "mu_prior_departement" in payload and payload["mu_prior_departement"] is not None:
            mu_prior_departement_new = torch.as_tensor(
                payload["mu_prior_departement"],
                dtype=self.mu_prior_departement.dtype,
                device=self.mu_prior_departement.device,
            )
            if mu_prior_departement_new.shape != self.mu_prior_departement.shape:
                raise ValueError(
                    f"mu_prior_departement shape mismatch: got {tuple(mu_prior_departement_new.shape)}, "
                    f"expected {tuple(self.mu_prior_departement.shape)}"
                )
            with torch.no_grad():
                self.mu_prior_departement.copy_(mu_prior_departement_new)

        # --------------------------------------------------
        # 5) resynchronisation des seuils dérivés
        # --------------------------------------------------
        self.thresholds = self._compute_thresholds().detach()

        if getattr(self, "learn_gains", False):
            if hasattr(self, "_compute_gains"):
                self.gains = self._compute_gains().detach()
            elif hasattr(self, "g_raw"):
                floor = float(getattr(self, "gains_floor", 0.0))
                self.gains = (F.softplus(self.g_raw) + floor).detach()

        print('New alpha', self.alpha)
        
    def plot_params(self, params_history, log_dir, best_epoch=None):
        import matplotlib.pyplot as plt

        root_dir = log_dir / "ordinal_params"
        root_dir.mkdir(parents=True, exist_ok=True)

        epochs = []
        thresholds_list = []
        gains_list = []
        deltas_list = []
        mu_list = []
        delta_scale_ema_list = []
        mu_prior_list = [] 
        mu_prior_global_list = []
        cluster_weights_list = []
        mass_active_list = []
        cluster_slot_to_raw_list = []
        mu_prior_departement_list = []
        departement_slot_to_raw_list = []

        iterator = []
        if isinstance(params_history, dict):
            iterator = sorted(params_history.items())
        else:
            for entry in params_history:
                if isinstance(entry, dict) and ("epoch" in entry):
                    iterator.append((entry["epoch"], entry))
            iterator.sort(key=lambda x: x[0])

        for ep, entry in iterator:
            if "ordinal_params" not in entry:
                continue

            stats_container = entry["ordinal_params"]
            p = stats_container.d if hasattr(stats_container, "d") else stats_container

            if not isinstance(p, dict):
                continue

            if "thresholds" not in p:
                continue
            
            mu_prior_departement_list.append(p.get("mu_prior_departement", None))
            departement_slot_to_raw_list.append(p.get("departement_slot_to_raw", None))
            epochs.append(ep)
            thresholds_list.append(p["thresholds"])
            gains_list.append(p.get("gains", None))
            deltas_list.append(p.get("deltas", None))
            mu_list.append(p.get("mu", None))
            delta_scale_ema_list.append(p.get("delta_scale_ema", None))
            mu_prior_list.append(p.get("mu_prior", None))
            mu_prior_global_list.append(p.get("mu_prior_global", None))
            cluster_weights_list.append(p.get("cluster_weights", None))
            mass_active_list.append(p.get("mass_active", None))
            cluster_slot_to_raw_list.append(p.get("cluster_slot_to_raw", None))

        if not epochs:
            return

        thresholds_arr = np.array(thresholds_list)

        alphatype = getattr(self, "alphatype", "global")

        if thresholds_arr.ndim == 2:
            # Global alphatype: one threshold set, shape (epochs, C-1)
            fig, ax = plt.subplots(figsize=(8, 6))
            for i in range(thresholds_arr.shape[1]):
                ax.plot(epochs, thresholds_arr[:, i], label=f"theta_{i}")
            ax.set_title(f"{self.__class__.__name__} Thresholds Evolution (global)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Threshold Value")
            ax.grid(True, alpha=0.3)
            if best_epoch is not None:
                ax.axvline(best_epoch, color="r", linestyle="--", label="Best Epoch")
            ax.legend()
            plt.tight_layout()
            plt.savefig(root_dir / "thresholds_evolution.png")
            plt.close()
        else:
            # Per-group alphatype: shape (epochs, n_groups, C-1)
            # Determine the slot→raw ID map for labelling
            if alphatype == "cluster":
                slot_maps = [np.asarray(x) if x is not None else None for x in cluster_slot_to_raw_list]
                group_label = "cluster"
            elif alphatype == "department":
                slot_maps = [np.asarray(x) if x is not None else None for x in departement_slot_to_raw_list]
                group_label = "dept"
            else:
                slot_maps = [None] * len(epochs)
                group_label = "group"

            last_slot_map = slot_maps[-1] if slot_maps else None
            n_groups = thresholds_arr.shape[1]
            n_thresholds = thresholds_arr.shape[2]

            # Filter groups that have at least one finite value
            active_groups = [
                g for g in range(n_groups)
                if np.isfinite(thresholds_arr[:, g, :]).any()
            ]
            n_plot = len(active_groups)

            if n_plot == 0:
                plt.close("all")
            else:
                cols = min(4, n_plot)
                rows = (n_plot + cols - 1) // cols
                fig, axes = plt.subplots(
                    rows, cols,
                    figsize=(5 * cols, 3.5 * rows),
                    sharex=True, sharey=False,
                    squeeze=False,
                )
                axes_flat = axes.flatten()
                cmap = plt.cm.tab10

                for plot_idx, g in enumerate(active_groups):
                    ax = axes_flat[plot_idx]
                    for i in range(n_thresholds):
                        ax.plot(
                            epochs,
                            thresholds_arr[:, g, i],
                            color=cmap(i / max(n_thresholds - 1, 1)),
                            label=f"theta_{i}",
                        )
                    if best_epoch is not None:
                        ax.axvline(best_epoch, color="r", linestyle="--", linewidth=0.8)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Threshold")

                    # Build meaningful title using slot→raw map
                    if (
                        last_slot_map is not None
                        and g < len(last_slot_map)
                        and last_slot_map[g] >= 0
                    ):
                        ax.set_title(f"{group_label} slot {g} / raw {int(last_slot_map[g])}")
                    else:
                        ax.set_title(f"{group_label} slot {g}")

                for j in range(n_plot, len(axes_flat)):
                    axes_flat[j].set_visible(False)

                axes_flat[min(n_plot - 1, len(axes_flat) - 1)].legend(
                    fontsize=7, loc="best"
                )
                fig.suptitle(
                    f"{self.__class__.__name__} Thresholds Evolution (per {group_label})",
                    fontsize=12,
                )
                plt.tight_layout()
                plt.savefig(root_dir / "thresholds_evolution.png")
                plt.close()

        try:
            valid_gains = [(ep, g) for ep, g in zip(epochs, gains_list) if g is not None]
            if valid_gains:
                g_epochs, g_vals = zip(*valid_gains)
                gains_arr = np.array(g_vals)
                fig, ax = plt.subplots(figsize=(8, 6))
                for i in range(gains_arr.shape[1]):
                    ax.plot(list(g_epochs), gains_arr[:, i], label=f"gain_{i}")
                ax.set_title(f"{self.__class__.__name__} Gains Evolution")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Gain Value")
                ax.grid(True, alpha=0.3)
                if best_epoch is not None:
                    ax.axvline(best_epoch, color="r", linestyle="--", label="Best Epoch")
                ax.legend()
                plt.tight_layout()
                plt.savefig(root_dir / "gains_evolution.png")
                plt.close()
        except Exception:
            plt.close("all")

        try:
            valid_deltas = [(ep, d) for ep, d in zip(epochs, deltas_list) if d is not None]
            if valid_deltas:
                d_epochs, d_vals = zip(*valid_deltas)
                d_epochs = list(d_epochs)
                ks = sorted(d_vals[0].keys())
                if ks:
                    fig, axes = plt.subplots(len(ks), 4, figsize=(20, 3 * len(ks)), sharex=True)
                    if len(ks) == 1:
                        axes = axes[None, :]
                    for i, k in enumerate(ks):
                        axes[i, 0].plot(d_epochs, [d[k]["median"] for d in d_vals if k in d], color="blue")
                        axes[i, 0].set_title(f"k={k} Median Delta")
                        axes[i, 1].plot(d_epochs, [d[k]["min"] for d in d_vals if k in d], color="red")
                        axes[i, 1].set_title(f"k={k} Min Delta")
                        axes[i, 2].plot(d_epochs, [d[k]["viol"] for d in d_vals if k in d], color="orange")
                        axes[i, 2].set_title(f"k={k} Viol Rate")
                        axes[i, 2].set_ylim(-0.1, 1.1)
                        axes[i, 3].plot(d_epochs, [d[k]["neg"] for d in d_vals if k in d], color="purple")
                        axes[i, 3].set_title(f"k={k} Mean NEG")
                    plt.tight_layout()
                    plt.savefig(root_dir / "deltas_stats.png")
                    plt.close()
        except Exception:
            plt.close("all")

        try:
            valid_scales = [(ep, s) for ep, s in zip(epochs, delta_scale_ema_list) if s is not None]
            if valid_scales:
                s_epochs, s_vals = zip(*valid_scales)
                scales_arr = np.array(s_vals)
                ep_list = list(s_epochs)

                def _pair_label(i):
                    if i < len(self.all_pairs):
                        a, b = self.all_pairs[i]
                        return f"{a}→{b}"
                    return f"pair_{i}"

                if scales_arr.ndim == 2:
                    num_p = scales_arr.shape[1]
                    fig, ax = plt.subplots(figsize=(10, 5))
                    for i in range(num_p):
                        ax.plot(ep_list, scales_arr[:, i], label=_pair_label(i))
                    ax.set_title(f"{self.__class__.__name__} – Delta Scale EMA per pair")
                    ax.set_yscale("log")
                    ax.set_xlabel("Epoch")
                    ax.grid(True, alpha=0.3)
                    if best_epoch is not None:
                        ax.axvline(best_epoch, color="r", linestyle="--", label="Best Epoch")
                    ax.legend(fontsize=6, ncol=max(1, num_p // 8))
                    plt.tight_layout()
                    plt.savefig(root_dir / "delta_scale_ema_evolution.png")
                    plt.close()
                else:
                    E, ncl, num_p = scales_arr.shape
                    mean_arr = scales_arr.mean(axis=1)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    for i in range(num_p):
                        ax.plot(ep_list, mean_arr[:, i], label=_pair_label(i))
                    ax.set_title(f"{self.__class__.__name__} – Delta Scale EMA (mean over groups)")
                    ax.set_yscale("log")
                    ax.set_xlabel("Epoch")
                    ax.grid(True, alpha=0.3)
                    if best_epoch is not None:
                        ax.axvline(best_epoch, color="r", linestyle="--", label="Best Epoch")
                    ax.legend(fontsize=6, ncol=max(1, num_p // 8))
                    plt.tight_layout()
                    plt.savefig(root_dir / "delta_scale_ema_evolution.png")
                    plt.close()

                    last = scales_arr[-1]
                    pair_labels = [_pair_label(i) for i in range(num_p)]
                    fig, ax = plt.subplots(figsize=(max(8, num_p * 0.4), max(4, ncl * 0.4)))
                    im = ax.imshow(np.log10(last + 1e-12), aspect="auto", origin="upper")
                    ax.set_title(f"log10(Delta Scale EMA) heatmap – epoch {ep_list[-1]}")
                    ax.set_xlabel("(a,b) pair")
                    ax.set_ylabel("Cluster slot")
                    ax.set_xticks(range(num_p))
                    ax.set_xticklabels(pair_labels, rotation=90, fontsize=6)
                    ax.set_yticks(range(ncl))
                    fig.colorbar(im, ax=ax, label="log10(scale)")
                    plt.tight_layout()
                    plt.savefig(root_dir / "delta_scale_ema_heatmap_last.png")
                    plt.close()
        except Exception:
            plt.close("all")

        try:
            valid_mu_priors = [(ep, m) for ep, m in zip(epochs, mu_prior_list) if m is not None]
            if valid_mu_priors:
                m_epochs, m_vals = zip(*valid_mu_priors)
                mu_arr = np.array(m_vals)
                if mu_arr.ndim == 3:
                    mu_arr_mean = np.nanmean(mu_arr, axis=1)
                else:
                    mu_arr_mean = mu_arr
                fig, ax = plt.subplots(figsize=(8, 6))
                for i in range(mu_arr_mean.shape[1]):
                    ax.plot(list(m_epochs), mu_arr_mean[:, i], label=f"mu_prior_avg_c={i}", alpha=0.4)

                valid_globals = [(ep_g, mg) for ep_g, mg in zip(epochs, mu_prior_global_list) if mg is not None]
                if valid_globals:
                    eg, m_vals_g = zip(*valid_globals)
                    mu_g_arr = np.array(m_vals_g)
                    for i in range(mu_g_arr.shape[1]):
                        ax.plot(list(eg), mu_g_arr[:, i], label=f"mu_prior_global_c={i}", linewidth=2, linestyle="--")
                ax.set_title("Mu Prior Evolution (Global vs Avg-Cluster)")
                ax.legend(fontsize="x-small", ncol=2)
                plt.tight_layout()
                plt.savefig(root_dir / "mu_prior_evolution.png")
                plt.close()
        except Exception:
            plt.close("all")

        try:
            valid_locals = [(ep, mp) for ep, mp in zip(epochs, mu_prior_list) if mp is not None]
            if valid_locals:
                epl, mp_vals = zip(*valid_locals)
                mp_arr = np.stack(mp_vals)
                fig, ax = plt.subplots(figsize=(10, 6))
                C = mp_arr.shape[2]
                for c in range(C):
                    series = mp_arr[:, :, c]
                    mean_c = np.nanmean(series, axis=1)
                    p10 = np.nanpercentile(series, 10, axis=1)
                    p90 = np.nanpercentile(series, 90, axis=1)
                    ax.plot(list(epl), mean_c, label=f"local_mean_c={c}")
                    ax.fill_between(list(epl), p10, p90, alpha=0.15)
                if best_epoch is not None:
                    ax.axvline(best_epoch, linestyle="--", alpha=0.5, label="Best Epoch")
                ax.set_title("Local mu_prior summary (mean + 10-90% band)")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("mu_prior value")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize="x-small", ncol=2)
                plt.tight_layout()
                plt.savefig(root_dir / "mu_prior_local_summary.png")
                plt.close()

                last = mp_arr[-1]
                fig, ax = plt.subplots(figsize=(8, 6))
                fill_val = np.nanmin(last[np.isfinite(last)]) if np.isfinite(last).any() else 0.0
                im = ax.imshow(np.nan_to_num(last, nan=fill_val), aspect="auto")
                ax.set_title(f"Local mu_prior heatmap (last epoch={epl[-1]})")
                ax.set_xlabel("Class")
                ax.set_ylabel("Cluster slot")
                ax.set_xticks(range(last.shape[1]))
                fig.colorbar(im, ax=ax, shrink=0.8)
                plt.tight_layout()
                plt.savefig(root_dir / "mu_prior_local_heatmap_last.png")
                plt.close()
        except Exception:
            plt.close("all")

        try:
            valid_mu = [(ep, m) for ep, m in zip(epochs, mu_list) if m is not None]
            if valid_mu:
                m_epochs, m_vals = zip(*valid_mu)
                mu_arr = np.stack(m_vals)
                fig, ax = plt.subplots(figsize=(10, 6))
                for c in range(mu_arr.shape[1]):
                    ax.plot(list(m_epochs), mu_arr[:, c], label=f"mu(class {c})")
                ax.set_title("Mu evolution per class")
                ax.legend()
                ax.grid(True, alpha=0.3)
                if best_epoch is not None:
                    ax.axvline(best_epoch, color="r", linestyle="--", label="Best Epoch")
                plt.tight_layout()
                plt.savefig(root_dir / "mu_s.png")
                plt.close()
        except Exception:
            plt.close("all")

        try:
            valid_mup = [(ep, m) for ep, m in zip(epochs, mu_prior_list) if m is not None]
            if valid_mup:
                mp_epochs, mp_vals = zip(*valid_mup)
                mp_arr = np.array(mp_vals)
                slot_maps = [np.asarray(x) if x is not None else None for x in cluster_slot_to_raw_list]

                if mp_arr.ndim == 3:
                    n_buf = mp_arr.shape[1]
                    n_classes = mp_arr.shape[2]
                    cluster_slots_to_plot = [cl for cl in range(n_buf) if np.isfinite(mp_arr[:, cl, :]).any()]
                    n_plot = len(cluster_slots_to_plot)

                    if n_plot > 0:
                        cols = min(4, n_plot)
                        rows = (n_plot + cols - 1) // cols
                        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), sharey=True, sharex=True)
                        if rows == 1 and cols == 1:
                            axes = np.array([[axes]])
                        elif rows == 1:
                            axes = axes[None, :]
                        axes_flat = axes.flatten()
                        cmap = plt.cm.plasma

                        last_slot_map = slot_maps[-1] if len(slot_maps) > 0 else None

                        for plot_idx, cl in enumerate(cluster_slots_to_plot):
                            ax = axes_flat[plot_idx]
                            for c in range(n_classes):
                                ax.plot(list(mp_epochs), mp_arr[:, cl, c], color=cmap(c / max(n_classes - 1, 1)), label=f"class {c}")
                            if best_epoch is not None:
                                ax.axvline(best_epoch, color="r", linestyle="--", linewidth=0.8)

                            if last_slot_map is not None and cl < len(last_slot_map) and last_slot_map[cl] >= 0:
                                ax.set_title(f"slot {cl} / raw {int(last_slot_map[cl])}")
                            else:
                                ax.set_title(f"slot {cl}")

                            ax.set_xlabel("Epoch")
                            ax.grid(True, alpha=0.3)

                        for j in range(n_plot, len(axes_flat)):
                            axes_flat[j].set_visible(False)

                        axes_flat[n_plot - 1].legend(fontsize=7, loc="best")
                        fig.suptitle(f"{self.__class__.__name__} — Mu Prior per cluster slot", fontsize=12)
                        plt.tight_layout()
                        plt.savefig(root_dir / "mu_per_cluster.png")
                        plt.close()
        except Exception as _e:
            plt.close("all")
            print(f"[plot_params] mu_per_cluster error: {_e}")

        try:
            valid_cw = [(ep, m) for ep, m in zip(epochs, cluster_weights_list) if m is not None]
            if valid_cw:
                cw_epochs, cw_vals = zip(*valid_cw)
                cw_arr = np.array(cw_vals)
                slot_maps = [np.asarray(x) if x is not None else None for x in cluster_slot_to_raw_list]

                if cw_arr.ndim == 2:
                    n_buf = cw_arr.shape[1]
                    cluster_slots_to_plot = [cl for cl in range(n_buf) if cw_arr[:, cl].any()]
                    n_cl = len(cluster_slots_to_plot)
                    if n_cl > 0:
                        cols = min(4, n_cl)
                        rows = (n_cl + cols - 1) // cols
                        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharey=True, sharex=True)
                        if rows == 1 and cols == 1:
                            axes = np.array([[axes]])
                        elif rows == 1:
                            axes = axes[None, :]
                        axes_flat = axes.flatten()

                        last_slot_map = slot_maps[-1] if len(slot_maps) > 0 else None

                        for plot_idx, cl in enumerate(cluster_slots_to_plot):
                            ax = axes_flat[plot_idx]
                            ax.plot(list(cw_epochs), cw_arr[:, cl], color="steelblue")
                            if best_epoch is not None:
                                ax.axvline(best_epoch, color="r", linestyle="--", linewidth=0.8)

                            if last_slot_map is not None and cl < len(last_slot_map) and last_slot_map[cl] >= 0:
                                ax.set_title(f"slot {cl} / raw {int(last_slot_map[cl])}")
                            else:
                                ax.set_title(f"slot {cl}")

                            ax.set_xlabel("Epoch")
                            ax.set_ylabel("w_z")
                            ax.grid(True, alpha=0.3)

                        for j in range(n_cl, len(axes_flat)):
                            axes_flat[j].set_visible(False)

                        fig.suptitle(f"{self.__class__.__name__} — Cluster EMA weights (softmax)", fontsize=11)
                        plt.tight_layout()
                        plt.savefig(root_dir / "cluster_weights_evolution.png")
                        plt.close()
        except Exception as _e:
            plt.close("all")
            print(f"[plot_params] cluster_weights_evolution error: {_e}")

        try:
            valid_ma = [(ep, m) for ep, m in zip(epochs, mass_active_list) if m is not None]
            if valid_ma:
                ma_epochs, ma_vals = zip(*valid_ma)
                ma_arr = np.array(ma_vals)
                slot_maps = [np.asarray(x) if x is not None else None for x in cluster_slot_to_raw_list]

                if ma_arr.ndim == 3:
                    n_buf = ma_arr.shape[1]
                    n_classes = ma_arr.shape[2]
                    cluster_slots_to_plot = [cl for cl in range(n_buf) if ma_arr[:, cl, :].any()]
                    n_cl = len(cluster_slots_to_plot)
                    if n_cl > 0:
                        cols = min(4, n_cl)
                        rows = (n_cl + cols - 1) // cols
                        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), sharey=False, sharex=True)
                        if rows == 1 and cols == 1:
                            axes = np.array([[axes]])
                        elif rows == 1:
                            axes = axes[None, :]
                        axes_flat = axes.flatten()
                        cmap = plt.cm.plasma

                        last_slot_map = slot_maps[-1] if len(slot_maps) > 0 else None

                        for plot_idx, cl in enumerate(cluster_slots_to_plot):
                            ax = axes_flat[plot_idx]
                            for c in range(n_classes):
                                ax.plot(list(ma_epochs), ma_arr[:, cl, c], color=cmap(c / max(n_classes - 1, 1)), alpha=0.7, label=f"class {c}")
                            total_mass = ma_arr[:, cl, :].sum(axis=1)
                            ax.plot(list(ma_epochs), total_mass, color="black", linewidth=1.5, linestyle="--", label="total")
                            if best_epoch is not None:
                                ax.axvline(best_epoch, color="r", linestyle="--", linewidth=0.8)

                            if last_slot_map is not None and cl < len(last_slot_map) and last_slot_map[cl] >= 0:
                                ax.set_title(f"slot {cl} / raw {int(last_slot_map[cl])}")
                            else:
                                ax.set_title(f"slot {cl}")

                            ax.set_xlabel("Epoch")
                            ax.set_ylabel("mass")
                            ax.grid(True, alpha=0.3)

                        for j in range(n_cl, len(axes_flat)):
                            axes_flat[j].set_visible(False)

                        axes_flat[n_cl - 1].legend(fontsize=7, loc="best")
                        fig.suptitle(f"{self.__class__.__name__} — Mass per cluster slot (per class)", fontsize=11)
                        plt.tight_layout()
                        plt.savefig(root_dir / "mass_active_per_cluster.png")
                        plt.close()
        except Exception as _e:
            plt.close("all")
            print(f"[plot_params] mass_active_per_cluster error: {_e}")

        try:
            tr_list, f_list, ep_list = [], [], []
            for ep, entry in iterator:
                p = entry["ordinal_params"].d if hasattr(entry["ordinal_params"], "d") else entry["ordinal_params"]
                if "transition" in p and "coverage" in p:
                    ep_list.append(ep)
                    tr_list.append(np.mean(p["transition"]))
                    f_list.append(np.mean(p["coverage"]))
            if ep_list:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(ep_list, tr_list, label="transition")
                ax.plot(ep_list, f_list, label="coverage", linestyle="--")
                ax.set_title(f"{self.__class__.__name__} – Loss components")
                ax.grid(True, alpha=0.3)
                if best_epoch is not None:
                    ax.axvline(best_epoch, color="r", linestyle="--", label="Best Epoch")
                ax.legend()
                plt.tight_layout()
                plt.savefig(root_dir / "loss_components.png")
                plt.close()
        except Exception:
            plt.close("all")
            
        # --- Plot Mu per department (depuis mu_prior_departement : shape (ndepartements, C)) ---
        try:
            valid_mupd = [(ep, m) for ep, m in zip(epochs, mu_prior_departement_list) if m is not None]
            if valid_mupd:
                md_epochs, md_vals = zip(*valid_mupd)
                md_arr = np.array(md_vals)  # (epochs, ndepartements, C)

                dept_maps = [np.asarray(x) if x is not None else None for x in departement_slot_to_raw_list]

                if md_arr.ndim == 3:
                    n_buf = md_arr.shape[1]
                    n_classes = md_arr.shape[2]

                    dept_slots_to_plot = [
                        d for d in range(n_buf)
                        if np.isfinite(md_arr[:, d, :]).any()
                    ]
                    n_depts_plot = len(dept_slots_to_plot)

                    if n_depts_plot > 0:
                        cols = min(4, n_depts_plot)
                        rows = (n_depts_plot + cols - 1) // cols

                        fig, axes = plt.subplots(
                            rows,
                            cols,
                            figsize=(5 * cols, 3.5 * rows),
                            sharey=False,
                            sharex=True
                        )

                        if rows == 1 and cols == 1:
                            axes = np.array([[axes]])
                        elif rows == 1:
                            axes = axes[None, :]

                        axes_flat = axes.flatten()
                        cmap = plt.cm.plasma

                        last_dept_map = dept_maps[-1] if len(dept_maps) > 0 else None

                        for plot_idx, d in enumerate(dept_slots_to_plot):
                            ax = axes_flat[plot_idx]

                            for c in range(n_classes):
                                ax.plot(
                                    list(md_epochs),
                                    md_arr[:, d, c],
                                    color=cmap(c / max(n_classes - 1, 1)),
                                    label=f"class {c}"
                                )

                            if best_epoch is not None:
                                ax.axvline(best_epoch, color="r", linestyle="--", linewidth=0.8)

                            if last_dept_map is not None and d < len(last_dept_map) and last_dept_map[d] >= 0:
                                ax.set_title(f"dept slot {d} / raw {int(last_dept_map[d])}")
                            else:
                                ax.set_title(f"dept slot {d}")

                            ax.set_xlabel("Epoch")
                            ax.grid(True, alpha=0.3)

                            # Échelle propre à chaque sous-figure
                            yvals = md_arr[:, d, :]
                            finite_mask = np.isfinite(yvals)

                            if finite_mask.any():
                                ymin = np.nanmin(yvals)
                                ymax = np.nanmax(yvals)

                                if np.isclose(ymin, ymax):
                                    pad = 0.1 if np.isclose(ymin, 0.0) else 0.05 * abs(ymin)
                                    ax.set_ylim(ymin - pad, ymax + pad)
                                else:
                                    pad = 0.05 * (ymax - ymin)
                                    ax.set_ylim(ymin - pad, ymax + pad)

                        for j in range(n_depts_plot, len(axes_flat)):
                            axes_flat[j].set_visible(False)

                        axes_flat[n_depts_plot - 1].legend(fontsize=7, loc="best")
                        fig.suptitle(f"{self.__class__.__name__} — Mu Prior per department", fontsize=12)
                        plt.tight_layout()
                        plt.savefig(root_dir / "mu_per_departement.png")
                        plt.close()

        except Exception as _e:
            plt.close("all")
            print(f"[plot_params] mu_per_departement error: {_e}")

        # --- Plot soft mass per class (raw vs gated) ---
        try:
            p_raw_list   = []
            p_gated_list = []
            ep_sm        = []

            for ep, entry in iterator:
                p = entry["ordinal_params"].d if hasattr(entry["ordinal_params"], "d") else entry["ordinal_params"]
                if not isinstance(p, dict):
                    continue
                has_raw   = "p_raw_mean"   in p and p["p_raw_mean"]   is not None
                has_gated = "p_gated_mean" in p and p["p_gated_mean"] is not None
                if has_raw or has_gated:
                    ep_sm.append(ep)
                    p_raw_list.append(p.get("p_raw_mean",   None))
                    p_gated_list.append(p.get("p_gated_mean", None))

            if ep_sm:
                # determine C from first valid entry
                first_raw   = next((x for x in p_raw_list   if x is not None), None)
                first_gated = next((x for x in p_gated_list if x is not None), None)
                C = (first_raw if first_raw is not None else first_gated).shape[0]

                cmap = plt.cm.plasma
                fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

                for c in range(C):
                    color = cmap(c / max(C - 1, 1))

                    # raw
                    if first_raw is not None:
                        vals = [x[c] if x is not None else float("nan") for x in p_raw_list]
                        axes[0].plot(ep_sm, vals, color=color, label=f"class {c}")

                    # gated
                    if first_gated is not None:
                        vals = [x[c] if x is not None else float("nan") for x in p_gated_list]
                        axes[1].plot(ep_sm, vals, color=color, label=f"class {c}")

                axes[0].set_title("Soft mass per class — RAW (before gate)")
                axes[1].set_title("Soft mass per class — GATED (after gate)")
                for ax in axes:
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("mean p[:, c]")
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=7, ncol=2)
                    if best_epoch is not None:
                        ax.axvline(best_epoch, color="r", linestyle="--", linewidth=0.8, label="Best")

                plt.tight_layout()
                plt.savefig(root_dir / "soft_mass_per_class.png")
                plt.close()
        except Exception as _e:
            plt.close("all")
            print(f"[plot_params] soft_mass_per_class error: {_e}")
class ClusterDepartmentRankNetLoss(nn.Module):
    """
    RankNet pairwise loss + thresholds learned per department or per cluster.

    Idée
    ----
    - Le terme principal est un RankNet pairwise :
          y_i > y_j  =>  s_i > s_j
      avec BCEWithLogits sur sigma * (s_i - s_j).

    - Les seuils ne servent PAS au ranking lui-même.
      Ils servent à :
        1) convertir un score latent en classe ordinale
        2) éventuellement régulariser la géométrie des seuils via Lmid

    - On peut apprendre :
        * un seul jeu de seuils globaux
        * un jeu de seuils par cluster
        * un jeu de seuils par department

    Paramètres clés
    ---------------
    num_classes : int
        Nombre de classes ordinales finales.
    sigma : float
        Pente du RankNet logit : logits = sigma * (s_i - s_j).
    num_pairs_per_group : Optional[int]
        Nombre de paires échantillonnées par groupe de ranking.
        Si None, utilise toutes les paires du groupe.
    tie_epsilon : float
        Ignore les paires où |y_i - y_j| <= tie_epsilon.
    use_soft_targets : bool
        Si True, cible pairwise douce :
            target_ij = sigmoid((y_i - y_j)/T)
        au lieu de {0,1}.
    soft_target_temperature : float
        Température pour les cibles douces.
    weight_by_delta : bool
        Pondère les paires par |y_i - y_j|^delta_power.
    delta_power : float
        Exposant de la pondération pairwise.
    wrank : float
        Poids du terme RankNet.
    wmid : float
        Poids du terme Lmid de calibration des seuils.
    alphatype : str
        "global", "cluster", ou "department"
        -> où l'on apprend les seuils.
    pair_scope : str
        "global", "cluster", ou "department"
        -> dans quel groupe on forme les paires RankNet.
    nclusters : int
        Nombre maximal de clusters attendus.
    ndepartements : int
        Nombre maximal de départements attendus.
    id : int
        Identifiant optionnel, comme dans ta loss actuelle.
    """

    def __init__(
        self,
        num_classes: int,
        sigmasig: float = 1.0,
        sigma: float = 1.0,
        num_pairs_per_group: Optional[int] = 2048,
        tie_epsilon: float = 0.0,
        use_soft_targets: bool = False,
        soft_target_temperature: float = 1.0,
        weight_by_delta: bool = True,
        delta_power: float = 1.0,
        wrank: float = 1.0,
        wmid: float = 0.1,
        alphatype: str = "department",     # where thresholds are learned
        pair_scope: str = "department",    # where ranking comparisons are formed
        nclusters: int = 1,
        ndepartements: int = 1,
        id: int = 0,
    ):
        super().__init__()

        self.C = int(num_classes)
        self.id = int(id)

        self.sigmasig = float(sigmasig)
        self.sigma = float(sigma)
        self.num_pairs_per_group = num_pairs_per_group
        self.tie_epsilon = float(tie_epsilon)
        self.use_soft_targets = bool(use_soft_targets)
        self.soft_target_temperature = float(soft_target_temperature)
        self.weight_by_delta = bool(weight_by_delta)
        self.delta_power = float(delta_power)

        self.wrank = float(wrank)
        self.wmid = float(wmid)

        self.alphatype = str(alphatype).lower()
        self.pair_scope = str(pair_scope).lower()

        self.nclusters = int(nclusters)
        self.ndepartements = int(ndepartements)

        if self.C < 2:
            raise ValueError("num_classes must be >= 2")
        if self.alphatype not in {"global", "cluster", "department"}:
            raise ValueError("alphatype must be one of: global, cluster, department")
        if self.pair_scope not in {"global", "cluster", "department"}:
            raise ValueError("pair_scope must be one of: global, cluster, department")

        # -----------------------------
        # Raw-id -> local slot mapping
        # -----------------------------
        self.cluster_raw_to_slot = {}
        self.departement_raw_to_slot = {}
        self.cluster_next_free_slot = 0
        self.departement_next_free_slot = 0

        self.register_buffer(
            "cluster_slot_to_raw",
            torch.full((self.nclusters,), -1, dtype=torch.long)
        )
        self.register_buffer(
            "departement_slot_to_raw",
            torch.full((self.ndepartements,), -1, dtype=torch.long)
        )

        # -----------------------------
        # Threshold parameters
        # alpha -> thresholds monotones via cumulative softplus
        # -----------------------------
        if self.alphatype == "global":
            self.alpha = nn.Parameter(torch.zeros(self.C - 1))
        elif self.alphatype == "cluster":
            self.alpha = nn.Parameter(torch.zeros(self.nclusters, self.C - 1))
        else:  # department
            self.alpha = nn.Parameter(torch.zeros(self.ndepartements, self.C - 1))

        # buffer de confort, mis à jour par update_params()
        init_thr = torch.linspace(-1.0, 1.0, self.C - 1)
        self.register_buffer("thresholds", init_thr.clone())

        self.epoch_stats: Dict[str, list] = {
            "rank": [],
            "mid": [],
            "n_pairs": [],
        }

    # =========================================================
    # Utilities
    # =========================================================
    @staticmethod
    def _validate_1d(name: str, x: torch.Tensor):
        if x.dim() != 1:
            raise ValueError(f"{name} must be 1D")

    def _remap_ids(self, raw_ids: torch.Tensor, buf_size: int, kind: str):
        """
        Remap raw ids -> contiguous local slots in [0 .. buf_size-1]
        """
        if raw_ids.dim() != 1:
            raw_ids = raw_ids.view(-1)
        raw_ids = raw_ids.long()
        device = raw_ids.device

        if kind == "cluster":
            raw_to_slot = self.cluster_raw_to_slot
            slot_to_raw = self.cluster_slot_to_raw
            next_free_attr = "cluster_next_free_slot"
        elif kind == "department":
            raw_to_slot = self.departement_raw_to_slot
            slot_to_raw = self.departement_slot_to_raw
            next_free_attr = "departement_next_free_slot"
        else:
            raise ValueError(f"Unknown kind: {kind}")

        if not raw_to_slot and (slot_to_raw != -1).any():
            max_slot = -1
            for slot_idx, raw_val in enumerate(slot_to_raw.tolist()):
                if raw_val != -1:
                    raw_to_slot[raw_val] = slot_idx
                    if slot_idx > max_slot:
                        max_slot = slot_idx
            setattr(self, next_free_attr, max_slot + 1)

        local_ids = torch.empty_like(raw_ids, dtype=torch.long, device=device)
        next_free_slot = getattr(self, next_free_attr)

        for i in range(raw_ids.numel()):
            rid = int(raw_ids[i].item())
            if rid in raw_to_slot:
                slot = raw_to_slot[rid]
            else:
                if next_free_slot >= buf_size:
                    raise ValueError(
                        f"No free slot left for kind='{kind}'. "
                        f"Encountered new raw id {rid}, but buf_size={buf_size}."
                    )
                slot = next_free_slot
                raw_to_slot[rid] = slot
                slot_to_raw[slot] = rid
                next_free_slot += 1
            local_ids[i] = slot

        setattr(self, next_free_attr, next_free_slot)
        valid_mask = torch.ones_like(local_ids, dtype=torch.bool, device=device)
        return slot_to_raw.clone(), local_ids, valid_mask

    def _compute_thresholds(self):
        """
        Enforce strictly increasing thresholds by cumulative softplus increments.
        """
        alpha = self.alpha

        if alpha.dim() == 1:
            theta0 = alpha[0:1]
            if alpha.numel() > 1:
                incr = F.softplus(alpha[1:])
                theta = torch.cat([theta0, incr], dim=0).cumsum(dim=0)
            else:
                theta = theta0
            return theta
        
        theta0 = alpha[:, 0:1]
        if alpha.size(1) > 1:
            incr = F.softplus(alpha[:, 1:])
            theta = torch.cat([theta0, incr], dim=1).cumsum(dim=1)
        else:
            theta = theta0
        return theta

    def _threshold_rows_for_samples(
        self,
        s: torch.Tensor,
        cluster_slot_ids: Optional[torch.Tensor],
        dept_slot_ids: Optional[torch.Tensor],
    ):
        """
        Retourne les thresholds par échantillon selon alphatype.
        """
        theta = self._compute_thresholds().to(device=s.device, dtype=s.dtype)

        if theta.dim() == 1:
            return theta[None, :].expand(s.numel(), -1)

        if self.alphatype == "cluster":
            if cluster_slot_ids is None:
                raise ValueError("cluster_slot_ids is required when alphatype='cluster'")
            return theta.index_select(0, cluster_slot_ids.long())

        if self.alphatype == "department":
            if dept_slot_ids is None:
                raise ValueError("dept_slot_ids is required when alphatype='department'")
            return theta.index_select(0, dept_slot_ids.long())

        raise ValueError(f"Unknown alphatype: {self.alphatype}")

    def _group_ids_for_pairwise(
        self,
        s: torch.Tensor,
        cluster_slot_ids: Optional[torch.Tensor],
        dept_slot_ids: Optional[torch.Tensor],
    ):
        """
        Détermine dans quel scope on forme les paires RankNet.
        """
        if self.pair_scope == "global":
            return torch.zeros(s.numel(), device=s.device, dtype=torch.long)

        if self.pair_scope == "cluster":
            if cluster_slot_ids is None:
                raise ValueError("cluster ids required when pair_scope='cluster'")
            _, local_group_ids = torch.unique(cluster_slot_ids, return_inverse=True)
            return local_group_ids

        if self.pair_scope == "department":
            if dept_slot_ids is None:
                raise ValueError("department ids required when pair_scope='department'")
            _, local_group_ids = torch.unique(dept_slot_ids, return_inverse=True)
            return local_group_ids

        raise ValueError(f"Unknown pair_scope: {self.pair_scope}")

    # =========================================================
    # RankNet loss
    # =========================================================
    def _build_pairs_for_group(
        self,
        idx: torch.Tensor,
        scores: torch.Tensor,
        y: torch.Tensor,
        sample_weight: Optional[torch.Tensor],
    ):
        """
        Build pairwise logits / targets / weights inside one group.
        """
        yg = y[idx]
        sg = scores[idx]
        wg = sample_weight[idx] if sample_weight is not None else None

        n = idx.numel()
        if n <= 1:
            return None

        # -----------------------------
        # all pairs or sampled pairs
        # -----------------------------
        if self.num_pairs_per_group is None:
            ii, jj = torch.triu_indices(n, n, offset=1, device=idx.device)
        else:
            ii = torch.randint(0, n, (self.num_pairs_per_group,), device=idx.device)
            jj = torch.randint(0, n, (self.num_pairs_per_group,), device=idx.device)
            mask = ii != jj
            ii, jj = ii[mask], jj[mask]
            if ii.numel() == 0:
                return None
            
        yi, yj = yg[ii], yg[jj]
        si, sj = sg[ii], sg[jj]

        dy = yi - yj
        abs_dy = dy.abs()
        
        # ignore ties / quasi-ties
        valid = abs_dy > self.tie_epsilon
        if not valid.any():
            return None

        yi, yj = yi[valid], yj[valid]
        si, sj = si[valid], sj[valid]
        dy = yi - yj
        abs_dy = abs_dy[valid]

        # target pairwise
        if self.use_soft_targets:
            target = torch.sigmoid(
                dy / max(self.soft_target_temperature, 1e-8)
            )
        else:
            target = (dy > 0).to(dtype=si.dtype)

        logits = self.sigmasig * (si - sj)
        
        # weights
        if self.weight_by_delta:
            pair_weight = abs_dy.clamp_min(1e-12).pow(self.delta_power)
        else:
            pair_weight = torch.ones_like(abs_dy, dtype=si.dtype)

        if wg is not None:
            wi, wj = wg[ii[valid]], wg[jj[valid]]
            pair_weight = pair_weight * wi * wj

        return logits, target.to(logits.dtype), pair_weight.to(logits.dtype)

    def _ranknet_loss(
        self,
        scores: torch.Tensor,
        y: torch.Tensor,
        group_ids: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ):
        """
        RankNet pairwise loss aggregated over groups.
        """
        unique_groups = torch.unique(group_ids)
        all_losses = []
        all_weights = []
        total_pairs = 0

        for g in unique_groups:
            idx = torch.where(group_ids == g)[0]
            out = self._build_pairs_for_group(idx, scores, y, sample_weight)
            if out is None:
                continue

            logits, target, pair_weight = out

            # RankNet = BCE sur logits = sigma*(s_i - s_j)
            loss_ij = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )

            all_losses.append(loss_ij)
            all_weights.append(pair_weight)
            total_pairs += int(loss_ij.numel())

        if len(all_losses) == 0:
            return scores.new_tensor(0.0), 0

        losses = torch.cat(all_losses, dim=0)
        weights = torch.cat(all_weights, dim=0)

        rank_loss = (losses * weights).sum() / weights.sum().clamp_min(1e-12)
        return rank_loss, total_pairs

    # =========================================================
    # Midpoint calibration for thresholds
    # =========================================================
    def _loss_mid_score_from_bins(
        self,
        s: torch.Tensor,
        theta_rows: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
    ):
        """
        Geometry-only threshold calibration in SCORE space.

        For each group g and class k:
            center_s[g, k] = mean score of samples currently assigned to hard bin k

        Then enforce:
            theta[g, k] ~ 0.5 * (center_s[g, k] + center_s[g, k+1])
        """
        device = s.device
        dtype = s.dtype
        s_det = s.detach()

        if theta_rows.dim() == 1:
            theta_rows = theta_rows.unsqueeze(0)  # (1, C-1)

        if group_ids is None:
            group_ids = torch.zeros_like(s_det, dtype=torch.long, device=device)
            G = 1
        else:
            group_ids = group_ids.to(device=device, dtype=torch.long).view(-1)
            G = int(group_ids.max().item()) + 1 if group_ids.numel() > 0 else theta_rows.shape[0]

        if theta_rows.shape[0] == 1 and G > 1:
            theta_rows = theta_rows.expand(G, -1)

        if theta_rows.shape[0] != G:
            raise ValueError(
                f"theta_rows and group_ids mismatch: "
                f"theta_rows.shape={theta_rows.shape}, G={G}"
            )

        thr_s = theta_rows.index_select(0, group_ids)  # (N, C-1)
        hard_bins = (s_det.unsqueeze(1) > thr_s.detach()).sum(dim=1)  # (N,)

        centers_s = torch.full((G, self.C), float("nan"), device=device, dtype=dtype)

        for k in range(self.C):
            mask = (hard_bins == k)
            if not mask.any():
                continue

            count_k = torch.zeros(G, device=device, dtype=dtype)
            sum_k = torch.zeros(G, device=device, dtype=dtype)

            ones_k = torch.ones(mask.sum(), device=device, dtype=dtype)
            count_k.scatter_add_(0, group_ids[mask], ones_k)
            sum_k.scatter_add_(0, group_ids[mask], s_det[mask])

            valid_g = (count_k > 0)
            centers_s[valid_g, k] = sum_k[valid_g] / count_k[valid_g]

        target_mid = 0.5 * (centers_s[:, :-1] + centers_s[:, 1:])
        valid = torch.isfinite(target_mid) & torch.isfinite(theta_rows)

        if not valid.any():
            return s.new_tensor(0.0), centers_s, hard_bins

        Lmid = F.smooth_l1_loss(theta_rows[valid], target_mid[valid], reduction="mean")
        return Lmid, centers_s, hard_bins

    # =========================================================
    # Forward
    # =========================================================
    def forward(
        self,
        score: torch.Tensor,
        y_cont: torch.Tensor,
        clusters_ids: Optional[torch.Tensor],
        departement_ids: Optional[torch.Tensor],
        sample_weight: Optional[torch.Tensor] = None,
    ):
        """
        score          : (N,) raw model score
        y_cont         : (N,) continuous/discrete relevance target for ranking
        clusters_ids   : (N,) raw cluster ids
        departement_ids: (N,) raw department ids
        """
        score = score / self.sigma
        y_cont = y_cont / self.sigma

        s = score.view(-1)
        y = y_cont.view(-1).to(device=s.device, dtype=s.dtype)

        self._validate_1d("score", s)
        self._validate_1d("y_cont", y)

        if clusters_ids is not None:
            clusters_ids = clusters_ids.view(-1).long().to(device=s.device)
            _, cluster_slot_ids, _ = self._remap_ids(
                clusters_ids, self.nclusters, kind="cluster"
            )
        else:
            cluster_slot_ids = None

        if departement_ids is not None:
            departement_ids = departement_ids.view(-1).long().to(device=s.device)
            _, dept_slot_ids, _ = self._remap_ids(
                departement_ids, self.ndepartements, kind="department"
            )
        else:
            dept_slot_ids = None

        if sample_weight is not None:
            sw = sample_weight.view(-1).to(device=s.device, dtype=s.dtype)
            self._validate_1d("sample_weight", sw)
        else:
            sw = None

        if not (s.numel() == y.numel()):
            raise ValueError("score and y_cont must have the same length")

        # -----------------------------------
        # 1) RankNet term
        # -----------------------------------
        pair_group_ids = self._group_ids_for_pairwise(
            s=s,
            cluster_slot_ids=cluster_slot_ids,
            dept_slot_ids=dept_slot_ids,
        )
        rank_loss, n_pairs = self._ranknet_loss(
            scores=s,
            y=y,
            group_ids=pair_group_ids,
            sample_weight=sw,
        )

        # -----------------------------------
        # 2) Optional threshold calibration
        # -----------------------------------
        theta_all = self._compute_thresholds().to(device=s.device, dtype=s.dtype)

        if self.wmid > 0.0:
            if self.alphatype == "global":
                Lmid, _, _ = self._loss_mid_score_from_bins(
                    s=s,
                    theta_rows=theta_all,
                    group_ids=None,
                )

            elif self.alphatype == "cluster":
                if cluster_slot_ids is None:
                    raise ValueError("clusters_ids required when alphatype='cluster'")
                active_cluster_slots, local_cluster_ids = torch.unique(
                    cluster_slot_ids, return_inverse=True
                )
                theta_mid = theta_all.index_select(0, active_cluster_slots)
                Lmid, _, _ = self._loss_mid_score_from_bins(
                    s=s,
                    theta_rows=theta_mid,
                    group_ids=local_cluster_ids,
                )

            elif self.alphatype == "department":
                if dept_slot_ids is None:
                    raise ValueError("departement_ids required when alphatype='department'")
                active_dept_slots, local_dept_ids = torch.unique(
                    dept_slot_ids, return_inverse=True
                )
                theta_mid = theta_all.index_select(0, active_dept_slots)
                Lmid, _, _ = self._loss_mid_score_from_bins(
                    s=s,
                    theta_rows=theta_mid,
                    group_ids=local_dept_ids,
                )
            else:
                raise ValueError(f"Unknown alphatype: {self.alphatype}")
        else:
            Lmid = s.new_tensor(0.0)

        total_loss = self.wrank * rank_loss + self.wmid * Lmid

        # logging
        self.epoch_stats.setdefault("rank", []).append(float(rank_loss.detach().cpu()))
        self.epoch_stats.setdefault("mid", []).append(float(Lmid.detach().cpu()))
        self.epoch_stats.setdefault("n_pairs", []).append(int(n_pairs))

        return total_loss

    # =========================================================
    # Inference helpers
    # =========================================================
    @torch.no_grad()
    def score_to_class(
        self,
        scores: torch.Tensor,
        clusters_ids: Optional[torch.Tensor] = None,
        departement_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert score -> class using the learned thresholds.
        """
        s = scores.detach().flatten()
        s = s / self.sigma
        device = s.device
        dtype = self.alpha.dtype

        thr = self._compute_thresholds().detach().to(device=device, dtype=dtype)

        if thr.dim() == 1:
            return torch.bucketize(s, thr, right=True)

        if self.alphatype == "cluster":
            if clusters_ids is None:
                raise ValueError("clusters_ids required when alphatype='cluster'")
            chosen_ids = clusters_ids.view(-1).long().to(device=device)
            _, idx, _ = self._remap_ids(chosen_ids, self.nclusters, kind="cluster")

        elif self.alphatype == "department":
            if departement_ids is None:
                raise ValueError("departement_ids required when alphatype='department'")
            chosen_ids = departement_ids.view(-1).long().to(device=device)
            _, idx, _ = self._remap_ids(chosen_ids, self.ndepartements, kind="department")

        else:
            raise ValueError(f"Unknown alphatype: {self.alphatype}")

        thr_s = thr.index_select(0, idx)  # (N, C-1)
        return (s.unsqueeze(1) > thr_s).sum(dim=1)

    def get_learnable_parameters(self):
        return {"alpha": self.alpha}

    def get_attribute(self):
        payload: Dict[str, Any] = {
            "alpha": self.alpha.detach().cpu().numpy(),
            "thresholds": self._compute_thresholds().detach().cpu().numpy(),
            "cluster_slot_to_raw": self.cluster_slot_to_raw.detach().cpu().numpy(),
            "departement_slot_to_raw": self.departement_slot_to_raw.detach().cpu().numpy(),
        }

        if self.epoch_stats.get("rank"):
            payload["rank"] = [float(np.mean(self.epoch_stats["rank"]))]
        if self.epoch_stats.get("mid"):
            payload["mid"] = [float(np.mean(self.epoch_stats["mid"]))]
        if self.epoch_stats.get("n_pairs"):
            payload["n_pairs"] = [int(np.mean(self.epoch_stats["n_pairs"]))]

        return [("ranknet_params", DictWrapper(payload))]

    def update_params(self, new_dict, epoch=None):
        payload = new_dict

        # Cas: {"epoch": ..., "ranknet_params": DictWrapper(...)}
        if isinstance(payload, dict):
            if "ranknet_params" in payload:
                if epoch is None and "epoch" in payload:
                    epoch = payload["epoch"]
                payload = payload["ranknet_params"]
            elif "ordinal_params" in payload:
                if epoch is None and "epoch" in payload:
                    epoch = payload["epoch"]
                payload = payload["ordinal_params"]

        # Cas: DictWrapper(...)
        if hasattr(payload, "numpy") and not isinstance(payload, dict):
            payload = payload.numpy()

        if not isinstance(payload, dict):
            raise TypeError(
                f"update_params expected a dict-like payload after unwrapping, got {type(payload)}"
            )
            
        print('Old alpha', self.alpha)

        if "alpha" not in payload or payload["alpha"] is None:
            raise ValueError(
                f"alpha missing from payload, {payload}"
            )

        alpha_new = torch.as_tensor(
            payload["alpha"],
            dtype=self.alpha.dtype,
            device=self.alpha.device,
        )

        if alpha_new.shape != self.alpha.shape:
            raise ValueError(
                f"alpha shape mismatch: got {tuple(alpha_new.shape)}, "
                f"expected {tuple(self.alpha.shape)}"
            )

        with torch.no_grad():
            self.alpha.copy_(alpha_new)
        
        print('New alpha', self.alpha)
-