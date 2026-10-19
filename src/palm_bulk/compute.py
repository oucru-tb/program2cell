"""Compute backend: CPU vs GPU and running MCMC chains at the same time.

Neither option changes the model or the posterior; they only change where and how the same
sampler runs. Call `configure_compute` once, at the start of a session, before any fit.
"""
import warnings
import jax
import numpyro


def _backends_initialized():
    try:
        from jax._src import xla_bridge
        return bool(xla_bridge._backends)
    except Exception:  # private API; if it moves, skip the check
        return False


def configure_compute(device="cpu", parallel_chains=1, precision="float32"):
    """Select the JAX device and allow chains to run in parallel.

    device: 'cpu', 'gpu', or 'auto' (GPU if JAX can see one, else CPU). A GPU needs a CUDA build
        of JAX (Linux/WSL: pip install "palm-bulk[gpu]").
    parallel_chains: on CPU, split the machine into this many XLA devices so up to this many
        chains run at once (`chain_method="parallel"`). Set it to the number of chains. Each chain
        then gets a share of the CPU threads, so the speed-up is real but below linear.
        On a single GPU, chains run batched (`chain_method="vectorized"`) and this is not needed.
    precision: 'float32' or 'float64'. With hundreds of samples, deep libraries and ~1000+ genes
        the total log-likelihood is ~1e7 and float32 rounding puts errors of several units on
        the energy differences NUTS uses to accept steps. 'float64' removes that at ~2x CPU cost
        per gradient (much more on consumer GPUs; data-centre GPUs such as A100/H100 are fine).

    Must run before the first JAX computation of the session; otherwise restart the kernel.
    Returns `compute_info()`."""
    if device not in ("cpu", "gpu", "auto"):
        raise ValueError("device must be 'cpu', 'gpu' or 'auto'")
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be 'float32' or 'float64'")
    jax.config.update("jax_enable_x64", precision == "float64")
    if _backends_initialized():
        warnings.warn("JAX is already initialised in this session: device / parallel_chains "
                      "only take effect after restarting the Python kernel.")
    if parallel_chains > 1:
        numpyro.set_host_device_count(int(parallel_chains))
    if device != "cpu":
        try:
            has_gpu = len(jax.devices("gpu")) > 0
        except RuntimeError:
            has_gpu = False
        if device == "gpu" and not has_gpu:
            raise RuntimeError("no GPU visible to JAX. Install a CUDA build of JAX "
                               "(pip install 'palm-bulk[gpu]') or use device='cpu'.")
        device = "gpu" if has_gpu else "cpu"
    numpyro.set_platform(device)
    return compute_info()


def compute_info():
    """Platform and devices JAX will use."""
    d = jax.devices()
    return dict(platform=d[0].platform, n_devices=len(d), devices=[str(x) for x in d],
                precision="float64" if jax.config.jax_enable_x64 else "float32")


def resolve_chain_method(chains, chain_method="auto"):
    """'auto': one device per chain available -> 'parallel'; GPU -> 'vectorized'
    (all chains batched on one GPU); otherwise 'sequential'."""
    if chain_method != "auto":
        return chain_method
    if chains <= 1:
        return "sequential"
    if jax.local_device_count() >= chains:
        return "parallel"
    if jax.default_backend() == "gpu":
        return "vectorized"
    return "sequential"
