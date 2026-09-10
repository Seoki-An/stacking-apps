import numpy as np
import pytest

from stacking_common.pointcloud import (
    IncrementalRegistration, PointCloudData, RegistrationQualityConfig, RegistrationResult,
)


class Backend:
    def __init__(self):
        self.targets = []
        self.transform = np.eye(4)
        self.error = False

    def register(self, source, target, initial_transform):
        self.targets.append(target.points.copy())
        if self.error:
            raise RuntimeError("registration failed")
        return RegistrationResult(self.transform, 1, 1, 0, len(source), ())


def _aligner():
    backend = Backend()
    quality = RegistrationQualityConfig(0.5, 0.5, 0.01, 0.1, 15)
    return IncrementalRegistration(backend, quality, 0.01, downsample=lambda cloud, _: cloud), backend


def _cloud(x):
    return PointCloudData([[x, 0, 0], [x, 1, 0], [x, 0, 1]])


def test_rejected_view_does_not_pollute_reference_and_undo_rebuilds_it():
    aligner, backend = _aligner()
    first = aligner.propose(_cloud(0))
    aligner.commit(first, selected=True)
    second = aligner.propose(_cloud(1))
    aligner.commit(second, selected=False)
    third = aligner.propose(_cloud(2))
    np.testing.assert_array_equal(backend.targets[-1], _cloud(0).points)
    aligner.commit(third, selected=True)
    assert len(aligner.reference) == 6
    aligner.undo()
    aligner.undo()
    np.testing.assert_array_equal(aligner.reference.points, _cloud(0).points)
    aligner.undo()
    assert aligner.reference is None


def test_correction_is_applied_once_and_bad_registration_cannot_be_forced():
    aligner, backend = _aligner()
    aligner.commit(aligner.propose(_cloud(0)), selected=True)
    backend.transform[0, 3] = -0.02
    good = aligner.propose(_cloud(0.02))
    np.testing.assert_allclose(good.cloud.points, _cloud(0).points)
    aligner.commit(good, selected=True)
    backend.transform[0, 3] = 2
    bad = aligner.propose(_cloud(0))
    assert "large_translation_correction" in bad.decision.reasons
    np.testing.assert_array_equal(bad.correction, np.eye(4))
    with pytest.raises(ValueError, match="failed registration"):
        aligner.commit(bad, selected=True)
    aligner.commit(bad, selected=False)
    backend.error = True
    failed = aligner.propose(_cloud(0))
    assert failed.decision.reasons == ("registration_error",)


def test_empty_seed_and_stale_proposals_cannot_enter_reference():
    aligner, _ = _aligner()
    empty = aligner.propose(PointCloudData(np.zeros((0, 3))))
    assert not empty.decision.accepted
    with pytest.raises(ValueError):
        aligner.commit(empty, selected=True)
    first = aligner.propose(_cloud(0))
    aligner.propose(_cloud(1))
    with pytest.raises(ValueError, match="stale"):
        aligner.commit(first, selected=True)
