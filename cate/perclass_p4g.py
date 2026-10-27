"""True per-class F1 on P4G for ensemble / RPI / HRPI (gold history), mean over seeds."""
import json
import numpy as np
from sklearn.metrics import f1_score
from cate.calib_baselines import softmax
from cate.config import RESULT
from cate.hrpi_analysis import shift_of
from cate.official import load_official
from cate.rla import adapt, dialogue_order, select
from cate.spire import load_crossfit

dialogs, samples, k = load_official("p4g", check=False)
yv = np.array([s["label"] for s in samples["valid"]]); yt = np.array([s["label"] for s in samples["test"]])
ov, ot = dialogue_order(samples["valid"]), dialogue_order(samples["test"])
lv, lt = shift_of(dialogs, samples, k, "valid"), shift_of(dialogs, samples, k, "test")
acc = {"ensemble": [], "RPI": [], "HRPI": []}
for seed in (42, 43, 44):
    P = {a: [] for a in acc}
    for r in load_crossfit("p4g", "roberta-base", seed):
        zv, zt = np.array(r["splits"]["valid"]["logits"]), np.array(r["splits"]["test"]["logits"])
        s2, h, _ = select(zv + lv, yv, ov); s2r, hr, _ = select(zv, yv, ov)
        P["ensemble"].append(softmax(zt))
        P["RPI"].append(softmax(zt + adapt(zt, yt, ot, s2r, hr)))
        P["HRPI"].append(softmax(zt + lt + adapt(zt + lt, yt, ot, s2, h)))
    for a, v in P.items():
        acc[a].append(100 * f1_score(yt, np.mean(v, 0).argmax(1), labels=list(range(k)), average=None, zero_division=0))
out = {a: np.mean(v, 0).tolist() for a, v in acc.items()}
(RESULT / "perclass_p4g.json").write_text(json.dumps(out), encoding="utf-8")
print({a: np.round(v, 1).tolist() for a, v in out.items()})
