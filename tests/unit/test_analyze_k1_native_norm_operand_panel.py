import numpy as np

from scripts.analyze_k1_native_norm_operand_panel import analyze


def test_analyze_native_norm_operand_panel_localizes_split():
    native_panel = {
        "records": [
            {
                "image_name": "10@particles.mrcs",
                "native_part_id": 4,
                "half": 1,
                "native": {
                    "current_size": 0.1,
                    "high_shell": 0.2,
                    "total": 0.3,
                    "new_norm": np.sqrt(0.6),
                },
            },
            {
                "image_name": "20@particles.mrcs",
                "native_part_id": 7,
                "half": 2,
                "native": {
                    "current_size": 0.4,
                    "high_shell": 0.5,
                    "total": 0.9,
                    "new_norm": np.sqrt(1.8),
                },
            },
        ]
    }
    recovar = {
        9: {
            "half": 1,
            "source_index": 9,
            "current_size": 0.11,
            "high_shell": 0.2,
            "total": 0.31,
            "new_norm": np.sqrt(0.62),
        },
        19: {
            "half": 2,
            "source_index": 19,
            "current_size": 0.4,
            "high_shell": 0.52,
            "total": 0.92,
            "new_norm": np.sqrt(1.84),
        },
    }

    report = analyze(native_panel, recovar)

    assert report["count"] == 2
    assert report["records"][0]["dominant_absolute_split_delta"] == "current_size"
    assert report["records"][1]["dominant_absolute_split_delta"] == "high_shell"
    assert report["summary"]["dominant_absolute_split_count"] == {
        "current_size": 1,
        "high_shell": 1,
    }
