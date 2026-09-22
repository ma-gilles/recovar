import numpy as np
import jax.numpy as jnp
import pytest
from test_sparse_pass2_bucketed_perf import MockDataset, IMAGE_SHAPE, ForwardModelConfig
from recovar.em.sparse_pass2.sparse_pass2_bucket_io import prepare_unshifted_bucket_operands

@pytest.mark.parametrize("diagnostic", [False, True])
def test_generic_noise_operands_follow_selected_precision(monkeypatch, diagnostic):
    monkeypatch.setenv("RECOVAR_SPARSE_PASS2_F64_NOISE_OPERANDS", str(int(diagnostic)))
    ds=MockDataset(n_images=3,seed=612)
    ids=np.arange(3)
    config=ForwardModelConfig.from_dataset(ds,disc_type="linear_interp",process_fn=ds.process_images)
    n_half=IMAGE_SHAPE[0]*(IMAGE_SHAPE[1]//2+1)
    noise=jnp.linspace(.8,1.4,n_half,dtype=jnp.float64)
    out=prepare_unshifted_bucket_operands(ds,jnp.asarray(ds._images),jnp.asarray(ds.CTF_params),ids,
        noise_variance_half=noise,config=config,score_with_masked_images=False,
        image_corrections=None,scale_corrections=None,image_pre_shifts=None,use_float64_scoring=False)
    assert out.score_weighted_half.dtype == (jnp.complex128 if diagnostic else jnp.complex64)
    assert out.ctf2_over_nv_half.dtype == (jnp.float64 if diagnostic else jnp.float32)
    if not diagnostic:
        expected_inverse=np.reciprocal(np.asarray(noise)).astype(np.float32)
        np.testing.assert_array_equal(out.inverse_noise_half,expected_inverse)
        weighted=np.asarray(out.ctf_half)*expected_inverse[None,:]
        np.testing.assert_array_equal(out.ctf2_over_nv_half,weighted*np.asarray(out.ctf_half))
