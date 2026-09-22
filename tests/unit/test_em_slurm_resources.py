import pytest
from scripts import em_slurm_resources as runner

pytestmark = pytest.mark.unit

def test_slurm_field_parser_exposes_exact_nonexclusive_one_gpu_contract():
    text = (
        "JobId=42 JobState=RUNNING NumNodes=1 NodeList=della-h1 "
        "ReqTRES=cpu=8,gres/gpu=1,mem=192G,node=1 "
        "AllocTRES=cpu=8,gres/gpu=1,mem=192G,node=1 "
        "Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=* "
        "TresPerNode=gres/gpu:h100:1 OverSubscribe=OK"
    )

    fields = runner._parse_scontrol_fields(text)

    assert fields["ReqTRES"] == fields["AllocTRES"]
    assert fields["AllocTRES"] == "cpu=8,gres/gpu=1,mem=192G,node=1"
    assert fields["Socks/Node"] == "*"
    assert fields["NtasksPerN:B:S:C"] == "0:0:*:*"
    assert runner._gpu_count_from_tres(fields["ReqTRES"]) == 1
    assert fields["OverSubscribe"] == "OK"
    assert runner._gpu_count_from_tres("gres/gpu:h100=1") == 1
    assert runner._gpu_count_from_tres("gres/gpu=1,gres/gpu:h100=1") == 1
