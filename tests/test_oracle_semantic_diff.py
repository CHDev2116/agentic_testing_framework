from eval.oracle_semantic_diff import diff_snapshots


def test_diff_snapshots_detects_release_change():
    baseline = {
        "label": "v1",
        "cases": {
            "hist-001": {
                "release": "GO",
                "conflict": "Consistent Pass",
                "semantic_errors": [],
                "override_applied": False,
                "description": "test",
            }
        },
    }
    current = {
        "label": "v2",
        "cases": {
            "hist-001": {
                "release": "NO_GO",
                "conflict": "Consistent Fail",
                "semantic_errors": ["semantic: x"],
                "override_applied": False,
                "description": "test",
            }
        },
    }
    report = diff_snapshots(baseline, current)
    assert report.has_changes
    assert len(report.changed) == 1
    assert "release" in report.changed[0].fields
    assert "NO_GO" in report.changed[0].summary_line()


def test_diff_snapshots_no_change():
    doc = {
        "label": "v1",
        "cases": {
            "hist-001": {
                "release": "NO_GO",
                "conflict": "Consistent Fail",
                "semantic_errors": [],
                "override_applied": False,
            }
        },
    }
    report = diff_snapshots(doc, doc)
    assert not report.has_changes
    assert report.unchanged_count == 1
