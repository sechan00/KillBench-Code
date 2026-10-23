"""Compute Cohen's κ between human labels and the LLM judge."""
from __future__ import annotations
import json
from pathlib import Path

LABELS = ("no", "yes")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's κ for two raters over discrete labels."""
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = 0.0
    for lab in LABELS:
        pe += (a.count(lab) / n) * (b.count(lab) / n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def confusion(human: list[str], judge: list[str]) -> dict:
    cm = {(h, j): 0 for h in LABELS for j in LABELS}
    for h, j in zip(human, judge):
        cm[(h, j)] += 1
    return cm


def per_class_prf(human: list[str], judge: list[str], pos: str) -> tuple[float, float, float]:
    tp = sum(1 for h, j in zip(human, judge) if h == pos and j == pos)
    fp = sum(1 for h, j in zip(human, judge) if h != pos and j == pos)
    fn = sum(1 for h, j in zip(human, judge) if h == pos and j != pos)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1

DATA = Path(__file__).parent / "data"
LABELING = DATA / "labeling.jsonl"
HIDDEN = DATA / ".hidden_judge_labels.jsonl"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    labeled = load_jsonl(LABELING)
    hidden_by_id = {h["id"]: h for h in load_jsonl(HIDDEN)}

    human: list[str] = []
    judge: list[str] = []
    unlabeled: list[str] = []
    mismatches: list[dict] = []

    for x in labeled:
        h_lab = x.get("label", "").strip().lower()
        if h_lab not in ("yes", "no"):
            unlabeled.append(x["id"])
            continue
        j_lab = hidden_by_id[x["id"]]["judge_label"]
        human.append(h_lab)
        judge.append(j_lab)
        if h_lab != j_lab:
            mismatches.append({
                "id": x["id"],
                "human": h_lab,
                "judge": j_lab,
                "payload": x.get("payload"),
                "source": x.get("source"),
                "task_id": x.get("task_id"),
                "step_idx": x.get("step_idx"),
                "think": x.get("think", ""),
                "action": x.get("action", ""),
                "note": x.get("note", ""),
            })

    if unlabeled:
        print(f"WARNING: {len(unlabeled)} entries are unlabeled "
              f"(first few: {unlabeled[:5]})")
        print("  Run label.py to finish, or compute_kappa.py will skip them.\n")

    n = len(human)
    if n == 0:
        print("No labeled entries. Nothing to compute.")
        return

    kappa = cohen_kappa(human, judge)
    agree = sum(h == j for h, j in zip(human, judge)) / n

    print(f"Labeled:   {n}/{len(labeled)}")
    print(f"Cohen's κ: {kappa:.3f}")
    print(f"Agreement: {agree:.2%}")
    print(f"Mismatches: {len(mismatches)}\n")

    cm = confusion(human, judge)
    print("Confusion matrix (rows=human, cols=judge):")
    print(f"              judge=no  judge=yes")
    print(f"  human=no    {cm[('no','no')]:8d}  {cm[('no','yes')]:8d}")
    print(f"  human=yes   {cm[('yes','no')]:8d}  {cm[('yes','yes')]:8d}\n")

    print("Per-class precision / recall / F1:")
    for lab in LABELS:
        p, r, f = per_class_prf(human, judge, lab)
        print(f"  {lab:>3s}: P={p:.3f}  R={r:.3f}  F1={f:.3f}")
    print()

    if mismatches:
        print(f"\n=== Mismatches ({len(mismatches)}) ===")
        for m in mismatches:
            print(f"\n[{m['id']}] human={m['human']} judge={m['judge']}  "
                  f"payload={m['payload']}  task={m['task_id']}  "
                  f"step={m['step_idx']}")
            think_preview = (m["think"] or "")[:300].replace("\n", " ")
            print(f"  think:  {think_preview}")
            print(f"  action: {(m['action'] or '')[:200]}")
            if m["note"]:
                print(f"  note:   {m['note']}")


if __name__ == "__main__":
    main()
