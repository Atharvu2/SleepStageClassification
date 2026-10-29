import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from build_dataset import feat_batch

def test_feature_shape_and_finite_values():
    x=np.random.default_rng(7).normal(size=(3,3000))
    z=feat_batch(x)
    assert z.shape==(3,22)
    assert np.isfinite(z).all()
