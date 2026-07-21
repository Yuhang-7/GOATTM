import json
import sys

p = sys.argv[1]
by_step = {}
validations = []

with open(p) as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("event") == "closure":
            by_step.setdefault(obj["optimizer_step"], []).append(obj)
        elif obj.get("event") == "validation":
            validations.append(obj)

print("validations:")
for v in validations:
    print(
        v["optimizer_step"],
        "train_loss",
        v["train_data_loss"],
        "test_loss",
        v["test_data_loss"],
        "train_raw",
        v["train_qoi_relative_error_raw"],
        "test_raw",
        v["test_qoi_relative_error_raw"],
        "normal_res",
        v["normal_relative_residual"],
    )

print("\nsteps 20-30 closure summary:")
for step in range(20, 31):
    rows = by_step.get(step, [])
    if not rows:
        continue
    first = rows[0]
    last = rows[-1]
    best_loss = min(rows, key=lambda x: x["loss"])
    best_data = min(rows, key=lambda x: x["data_loss"])
    print(
        "step {step:2d}: n={n:2d} first_loss={first_loss:.8g} "
        "last_loss={last_loss:.8g} best_loss={best_loss:.8g} "
        "best_data={best_data:.8g} last_grad={last_grad:.8g} "
        "best_grad={best_grad:.8g}".format(
            step=step,
            n=len(rows),
            first_loss=first["loss"],
            last_loss=last["loss"],
            best_loss=best_loss["loss"],
            best_data=best_data["data_loss"],
            last_grad=last["grad_norm"],
            best_grad=best_loss["grad_norm"],
        )
    )

last_step = max(by_step)
print("\nlast closure:")
print(json.dumps(by_step[last_step][-1], sort_keys=True))
