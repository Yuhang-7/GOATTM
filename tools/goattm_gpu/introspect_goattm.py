import inspect

import quadrode_gpu_goattm as q

keywords = ("rank", "low", "dense", "quadratic", "linear", "energy")
names = sorted(name for name in dir(q) if any(k in name.lower() for k in keywords))
for name in names:
    obj = getattr(q, name)
    try:
        sig = str(inspect.signature(obj))
    except Exception:
        sig = ""
    print(f"{name}{sig}")
