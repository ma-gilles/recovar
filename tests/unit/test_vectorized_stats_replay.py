import numpy as np
import pytest
from test_compact_capacity_integration import _fused_kclass_capacity_fixture, _fused_kclass_result_arrays
from recovar.em.sparse_pass2 import sparse_pass2_bucketed as engine
from recovar.em.classification.k_class_results import SparseKClassHostStatistics

@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("device_scalars", [False, True])
def test_vectorized_replay_matches_ordered_multibucket_results(monkeypatch, noise, device_scalars):
    flags={"RECOVAR_DISABLE_CUDA":"1", "RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS":"1",
           "RECOVAR_SPARSE_KCLASS_COMPACT_PAIRS_MIN_BUCKET_SIZE":"1",
           "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_MAX_IMAGES_PER_MICROBATCH":"1",
           "RECOVAR_SPARSE_PASS2_IMAGE_CAPACITY":"1",
           "RECOVAR_SPARSE_KCLASS_COMPACT_PAIR_DEVICE_INDEX":"1",
           "RECOVAR_SPARSE_KCLASS_DEVICE_CHUNK_SCALARS":str(int(device_scalars))}
    for name,value in flags.items():monkeypatch.setenv(name,value)
    monkeypatch.setattr(engine,"quantized_image_capacity",lambda n,**kw:4)
    calls=[]
    original=SparseKClassHostStatistics.update_bucket
    def update(self,**kw):
        calls.append(kw.get("vectorized_stats_replay",False))
        return original(self,**kw)
    monkeypatch.setattr(SparseKClassHostStatistics,"update_bucket",update)
    kwargs=_fused_kclass_capacity_fixture();kwargs["accumulate_noise"]=noise
    outputs=[]
    for mode in ["0","1"]:
        monkeypatch.setenv("RECOVAR_SPARSE_KCLASS_VECTORIZED_STATS_REPLAY",mode)
        outputs.append(_fused_kclass_result_arrays(engine.compute_k_class_pass2_stats_sparse_fused(**kwargs)))
    assert calls.count(True)>1 and calls.count(False)==calls.count(True)
    assert outputs[0].keys()==outputs[1].keys()
    for name in outputs[0]:np.testing.assert_array_equal(outputs[0][name],outputs[1][name],err_msg=name)
