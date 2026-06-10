"""Quality / machine-adaptivity policy for splatreg, *full by default, dial down to fit*.

Registration here is memory-bounded by the **Sim(3) autodiff Jacobian** in
:mod:`splatreg.solvers.lm`: its vmap'd reverse pass holds, per Jacobian row-chunk, one SDF forward
*block*'s graph, so peak ``~ jac_row_chunk x min(n_points, sdf_chunk) x M_target`` (measured). All
three of ``jac_row_chunk``, ``sdf_chunk`` and ``n_points`` therefore divide the peak, but only
``n_points`` changes the *result*; the two chunk knobs are numerically lossless.

This module turns ``n_points`` (how many source anchors the residuals sample), the chunk knobs,
the normal-estimation ``knn`` and the LM ``max_iters`` into a single **quality policy** so the same
code runs full-quality on a big GPU and (at reduced fidelity) without OOM on a small GPU or a
CPU-only box, by trading ``n_points`` for footprint only when asked, and *always* auto-fitting the
lossless chunk knobs to the available memory.

Policy
------
* **Default is FULL.** Out of the box nothing is capped: ``n_points`` is the residual's own full
  default and the Jacobian is row-chunked purely to bound *peak* memory with **zero** quality
  loss (same Jacobian, numerically identical, chunking only changes the live-graph size).
* A user may pick a named **quality**, ``"full"`` / ``"balanced"`` / ``"low"``, or a **0..1
  scale** (1.0 == full, smaller == fewer points / iterations), to trade accuracy for footprint
  explicitly.
* ``quality="auto"`` **detects the hardware at runtime** (``torch.cuda.mem_get_info`` on CUDA,
  ``psutil`` / ``os.sysconf`` RAM on CPU) and picks the *largest* sizes that fit the available
  memory: a 32 GB GPU lands at (or near) full, a 6 GB GPU or a CPU-only laptop scales itself down
  so it *runs* rather than OOMing. It never silently degrades a machine that can afford full.

Nothing here is heavy: torch + (optional) psutil, with an ``os.sysconf`` RAM fallback.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import Optional, Union

import torch

_log = logging.getLogger("splatreg")

__all__ = ["QualityConfig", "resolve_quality", "FULL", "BALANCED", "LOW"]

# --- Full-quality reference knobs ---------------------------------------------------------------
# QUALITY vs MEMORY-SAFETY — two different kinds of knob:
#   * ``n_points`` / ``knn`` / ``max_iters`` set ACCURACY. Dialling them down genuinely trades
#     fidelity for footprint; FULL keeps them maximal (``n_points=None`` -> ALL source anchors).
#   * ``jac_row_chunk`` / ``sdf_chunk_size`` set only PEAK MEMORY of the Sim(3) autodiff and are
#     numerically NEUTRAL (the chunked Jacobian is bit-for-bit the unchunked one). So they are
#     ALWAYS auto-fitted to the available memory — even in FULL mode — which is what lets full
#     quality run on a small GPU without OOM and without throwing away a single point.
# These ``_FULL_*`` chunk values are only the *ceiling* the fitter starts from.
_FULL_N_POINTS: Optional[int] = None  # None -> residuals sample ALL source anchors (full).
_FULL_JAC_ROW_CHUNK = 256  # autodiff row-chunk ceiling (memory-fitted down per run).
_FULL_SDF_CHUNK = 2048  # gaussian_sdf per-query block ceiling (memory-fitted down).
_FULL_KNN = 50  # normal-estimation neighbourhood.
_FULL_MAX_ITERS = 20  # LM iterations (api/Tracker default; callers may override).
_FULL_REFINE_ITERS = 10  # photometric-refine LM iterations (the opt-in refine="photometric" stage).

# Floors so even the smallest machine still does something sane.
_MIN_N_POINTS = 128
_MIN_JAC_ROW_CHUNK = 8
_MIN_SDF_CHUNK = 64
_MIN_MAX_ITERS = 8
_MIN_REFINE_ITERS = 3

# Empirical peak-memory model for the row-chunked Sim(3) SDF autodiff (measured on this box,
# torch 2.12 / float32): the vmap'd reverse pass holds, per Jacobian row-chunk, one SDF forward
# BLOCK's (block x M) graph, so
#     peak_bytes ~= _PEAK_BYTES_PER_TRIPLE * jac_row_chunk * min(n_points, sdf_chunk_size) * M
# where M is the target anchor count. Both chunk knobs therefore divide the peak linearly and
# losslessly. Measured constant was ~37 B; rounded UP to 48 for headroom so the fitter under-shoots
# memory rather than over-shoots (it never wants to be the thing that OOMs).
_PEAK_BYTES_PER_TRIPLE = 48.0


@dataclass(frozen=True)
class QualityConfig:
    """Resolved per-run sizing knobs (the output of :func:`resolve_quality`).

    Attributes
    ----------
    n_points : source-anchor sample size for the SDF/ICP residuals, or ``None`` for *all* anchors
        (full quality). An explicit ``residuals=[...]`` overrides this; it only fills the default set.
    jac_row_chunk : row-chunk for the Sim(3) autodiff Jacobian (``jacrev(chunk_size=...)``). Bounds
        peak autodiff memory with **no** effect on the result.
    sdf_chunk_size : per-query row block for :func:`splatreg.geometry.gaussian_sdf.gaussian_sdf`.
    knn : neighbourhood size for anchor-normal estimation.
    max_iters : default LM iteration count (an explicit ``max_iters=`` to ``register`` wins).
    refine_iters : default LM iteration count for the opt-in photometric refinement stage
        (``register(..., refine="photometric")``); an explicit ``max_iters`` in ``refine_kwargs``
        wins. Each iteration renders the splats from the camera ring, so this is the knob that
        sizes the refine's render budget.
    label : human-readable provenance (e.g. ``"full"``, ``"auto:cuda 6.0GiB-free -> 0.50"``).
    """

    n_points: Optional[int] = _FULL_N_POINTS
    jac_row_chunk: int = _FULL_JAC_ROW_CHUNK
    sdf_chunk_size: int = _FULL_SDF_CHUNK
    knn: int = _FULL_KNN
    max_iters: int = _FULL_MAX_ITERS
    refine_iters: int = _FULL_REFINE_ITERS
    label: str = "full"


# Named ACCURACY presets — only the quality knobs (n_points / knn / max_iters) differ; the chunk
# knobs here are just ceilings that the memory fitter lowers per run. FULL keeps n_points uncapped.
FULL = QualityConfig(label="full")
BALANCED = QualityConfig(
    n_points=4096,
    knn=50,
    max_iters=_FULL_MAX_ITERS,
    refine_iters=8,
    label="balanced",
)
LOW = QualityConfig(
    n_points=1024,
    knn=24,
    max_iters=_FULL_MAX_ITERS,
    refine_iters=5,
    label="low",
)

_NAMED = {"full": FULL, "balanced": BALANCED, "low": LOW}


# --- helpers ------------------------------------------------------------------------------------
def _scale_config(scale: float) -> QualityConfig:
    """A 0..1 ACCURACY scale (1.0 == full, smaller == fewer points / shallower fit).

    Only the accuracy knobs move: ``n_points`` is a fraction of a full reference sample (so any
    scale below 1 yields a concrete cap; full itself stays uncapped at ``scale == 1``) and ``knn``
    interpolates toward the LOW value. The quality-neutral chunk knobs are left at their ceilings
    here and fitted to memory later by :func:`_fit_chunks`.
    """
    s = float(min(1.0, max(0.0, scale)))
    if s >= 1.0:
        return replace(FULL, label="scale=1.00")
    ref_points = 16384  # full-reference sample to take a fraction of when asked for < full.
    n_points = max(_MIN_N_POINTS, int(round(ref_points * s)))
    knn = int(round(_FULL_KNN * s + LOW.knn * (1.0 - s)))
    refine_iters = max(_MIN_REFINE_ITERS, int(round(_FULL_REFINE_ITERS * s + LOW.refine_iters * (1.0 - s))))
    return QualityConfig(
        n_points=n_points,
        knn=knn,
        max_iters=_FULL_MAX_ITERS,
        refine_iters=refine_iters,
        label=f"scale={s:.2f}",
    )


def _free_bytes(device: torch.device) -> tuple[Optional[int], str]:
    """Available memory (bytes) on ``device`` and a short source tag, or ``(None, ...)`` if unknown.

    CUDA uses ``torch.cuda.mem_get_info`` (free VRAM on that device). CPU uses ``psutil`` when
    importable, else ``os.sysconf`` (available physical RAM); either way only a *fraction* of it is
    handed to the budget by the caller so we never lean on the last byte.
    """
    if device.type == "cuda":
        try:
            free, total = torch.cuda.mem_get_info(device)
            return int(free), f"cuda free={free / 2**30:.1f}GiB/{total / 2**30:.1f}GiB"
        except Exception as exc:  # pragma: no cover - exotic CUDA build
            _log.info("quality='auto': torch.cuda.mem_get_info failed (%s); using FULL.", exc)
            return None, "cuda(mem_get_info-failed)"
    # CPU / other: physical RAM available.
    try:
        import psutil  # optional

        avail = int(psutil.virtual_memory().available)
        return avail, f"cpu psutil avail={avail / 2**30:.1f}GiB"
    except Exception:
        pass
    try:
        avail = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        return avail, f"cpu sysconf avail={avail / 2**30:.1f}GiB"
    except (ValueError, OSError):  # pragma: no cover - platform without sysconf
        return None, "cpu(unknown-ram)"


# When the target anchor count is unknown (no Gaussians handed in), assume a mid-size splat so a
# memory bound can still be derived.
_ASSUMED_TARGET_ANCHORS = 50_000
# Fraction of FREE memory used as the working budget for the autodiff peak (the rest is headroom for
# params, the ICP cdist, framework overhead, and other processes). CPU is given less (shared RAM).
_BUDGET_FRAC_CUDA = 0.55
_BUDGET_FRAC_CPU = 0.25
# Largest source sample 'auto' will ever pick (beyond this the marginal accuracy is nil and the ICP
# (n_points x M) cdist — which the chunk fit does NOT cover — starts to dominate).
_AUTO_MAX_N_POINTS = 16_384


def _fit_chunks(
    cfg: QualityConfig,
    device: torch.device,
    target_anchors: Optional[int],
    source_anchors: Optional[int] = None,
) -> QualityConfig:
    """Lower ``cfg``'s quality-neutral chunk knobs until the Sim(3) autodiff peak fits memory.

    The chunked autodiff peaks at ``~_PEAK_BYTES_PER_TRIPLE * jac_row_chunk * min(n_points,
    sdf_chunk) * M`` (measured). Both chunks divide that peak losslessly, so we shrink them, NOT
    the accuracy knobs, to fit a fraction of the free memory. This is what makes even ``"full"``
    (all ``n_points``) run on a small GPU without OOM and bit-for-bit identically to a big one.

    Strategy: keep ``jac_row_chunk`` at its ceiling and pull ``sdf_chunk`` down first (it is the
    cheaper loop), then pull ``jac_row_chunk`` down if still over budget, clamped to the floors. If
    memory is undetectable, the ceilings are kept (the row-chunk already bounds peak; full is safe).
    ``source_anchors`` (when known) sharpens the estimate for ``n_points=None`` (full): the residual
    then samples exactly the source count, not the assumed ceiling.
    """
    free, tag = _free_bytes(device)
    # Effective sample size the SDF residual will actually use: the explicit cap, else the source
    # count (full uses ALL source anchors), else a conservative ceiling.
    if cfg.n_points and cfg.n_points > 0:
        n_pts_eff = cfg.n_points
    elif source_anchors and source_anchors > 0:
        n_pts_eff = int(source_anchors)
    else:
        n_pts_eff = _AUTO_MAX_N_POINTS
    M = int(target_anchors) if target_anchors and target_anchors > 0 else _ASSUMED_TARGET_ANCHORS
    M = max(M, 1)

    jac = int(cfg.jac_row_chunk)
    sdf = min(int(cfg.sdf_chunk_size), n_pts_eff)  # a block bigger than the sample is wasted
    if free is None:
        return replace(
            cfg, sdf_chunk_size=max(_MIN_SDF_CHUNK, sdf), label=f"{cfg.label} (chunks={jac}/{sdf}, mem={tag})"
        )

    frac = _BUDGET_FRAC_CUDA if device.type == "cuda" else _BUDGET_FRAC_CPU
    budget = free * frac

    def peak(j, s):
        return _PEAK_BYTES_PER_TRIPLE * j * min(n_pts_eff, s) * M

    # 1) shrink the SDF forward block first (down to its floor).
    while peak(jac, sdf) > budget and sdf > _MIN_SDF_CHUNK:
        sdf = max(_MIN_SDF_CHUNK, sdf // 2)
    # 2) then shrink the Jacobian row-chunk (down to its floor).
    while peak(jac, sdf) > budget and jac > _MIN_JAC_ROW_CHUNK:
        jac = max(_MIN_JAC_ROW_CHUNK, jac // 2)

    est = peak(jac, sdf) / 2**30
    fitted = replace(
        cfg,
        jac_row_chunk=jac,
        sdf_chunk_size=max(_MIN_SDF_CHUNK, sdf),
        label=f"{cfg.label} (chunks jac={jac}/sdf={sdf}, est_peak~{est:.2f}GiB, {tag}, M={M})",
    )
    _log.info("splatreg quality: %s", fitted.label)
    return fitted


def _auto_config(device: torch.device, target_anchors: Optional[int]) -> QualityConfig:
    """ACCURACY policy for ``quality='auto'``: keep ``n_points`` full when the (n_points x M) ICP
    cdist comfortably fits, else cap it to what fits, the chunk knobs are fitted separately.

    The Sim(3) autodiff peak is handled losslessly by :func:`_fit_chunks` (chunking never costs
    accuracy), so the only reason ``auto`` ever *lowers* the accuracy dial is the ONE working set
    chunking can't divide: the ICP residual's dense ``(n_points x M)`` ``torch.cdist``. We size
    ``n_points`` against that. A roomy GPU stays full (``n_points=None``); a small GPU / CPU caps it.
    """
    free, tag = _free_bytes(device)
    if free is None:
        return replace(FULL, label=f"auto:{tag} -> full(no-detect)")

    frac = _BUDGET_FRAC_CUDA if device.type == "cuda" else _BUDGET_FRAC_CPU
    budget = free * frac
    M = int(target_anchors) if target_anchors and target_anchors > 0 else _ASSUMED_TARGET_ANCHORS
    M = max(M, 1)

    # ICP cdist working set ~ n_points * M * (a few float32 buffers); ~32 B/pair is generous.
    icp_bytes_per_pair = 32.0
    fit_points = budget / (icp_bytes_per_pair * M)
    if fit_points >= _AUTO_MAX_N_POINTS:
        return replace(FULL, label=f"auto:{tag} frac={frac:g} -> full n_points")
    n_points = int(max(_MIN_N_POINTS, min(fit_points, _AUTO_MAX_N_POINTS)))
    knn = _FULL_KNN if n_points >= 1024 else 24
    return QualityConfig(
        n_points=n_points,
        knn=knn,
        max_iters=_FULL_MAX_ITERS,
        label=f"auto:{tag} frac={frac:g} -> n_points={n_points}",
    )


def resolve_quality(
    quality: Union[str, float, QualityConfig, None],
    device: Optional[torch.device] = None,
    *,
    target_anchors: Optional[int] = None,
    source_anchors: Optional[int] = None,
) -> QualityConfig:
    """Turn a user ``quality`` request into a concrete, memory-fitted :class:`QualityConfig`.

    Two stages: (1) pick the ACCURACY knobs (``n_points`` / ``knn`` / ``max_iters``) from the
    requested policy, then (2) ALWAYS fit the quality-neutral CHUNK knobs (``jac_row_chunk`` /
    ``sdf_chunk_size``) to the available memory via :func:`_fit_chunks` so the Sim(3) autodiff peak
    can never OOM, for *every* mode, full included. Chunking is numerically lossless, so stage 2
    changes footprint, not the result.

    Parameters
    ----------
    quality :
        * ``None`` or ``"full"`` -> full ACCURACY (all source anchors). **This is the default.** The
          chunk knobs are still memory-fitted so full runs even on a small GPU.
        * ``"balanced"`` / ``"low"`` -> named accuracy presets (bounded ``n_points``).
        * a float in ``[0, 1]`` -> scaled accuracy (``1.0`` == full, smaller == fewer points).
        * ``"auto"`` -> detect free GPU/CPU memory and pick the largest ``n_points`` that fits
          (full on a roomy machine; capped on a small one), chunks fitted on top.
        * a :class:`~splatreg.quality.QualityConfig` -> returned UNCHANGED (advanced manual override;
          no fitting, so you own the memory budget).
    device : the device the run will execute on (read for memory in ``"auto"`` and in the chunk
        fit). Defaults to CUDA-current when available else CPU.
    target_anchors : the target splat's Gaussian count, if known, sharpens both the ``"auto"``
        ``n_points`` budget and the chunk fit (peak ``~ jac_row_chunk x min(n_points,sdf) x M``).
    source_anchors : the source splat's Gaussian count, if known, sharpens the chunk fit for
        ``full`` (``n_points=None`` samples ALL source anchors, so the true sample is the source size).

    Returns
    -------
    QualityConfig : the resolved, memory-fitted sizing for this run.
    """
    if isinstance(quality, QualityConfig):
        return quality  # explicit override: user owns the budget.
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    if quality is None:
        base = FULL
    elif isinstance(quality, (int, float)) and not isinstance(quality, bool):
        base = _scale_config(float(quality))
    elif isinstance(quality, str):
        key = quality.strip().lower()
        if key == "auto":
            base = _auto_config(device, target_anchors)
        elif key in _NAMED:
            base = _NAMED[key]
        else:
            raise ValueError(
                f"quality must be 'full' | 'balanced' | 'low' | 'auto' | a 0..1 float | "
                f"QualityConfig; got {quality!r}."
            )
    else:
        raise TypeError(f"quality must be str | float | QualityConfig | None, got {type(quality).__name__}.")
    return _fit_chunks(base, device, target_anchors, source_anchors)
