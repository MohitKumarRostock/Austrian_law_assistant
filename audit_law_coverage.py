import re
import pandas as pd

SOFT_HYPHEN = "\u00ad"
HYPH = r"\-\u2010\u2011\u2012\u2013\u2014"

def norm(s: str) -> str:
    if not s:
        return ""
    t = s.replace(SOFT_HYPHEN, "").lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(rf"([a-zäöüß])([{HYPH}])\s+([a-zäöüß])", r"\1\3", t)  # irr- thum -> irrthum
    return t.strip()

def count_matches(series: pd.Series, pattern: str) -> int:
    pat = re.compile(pattern, re.I)
    return int(series.map(lambda x: bool(pat.search(norm(str(x))))).sum())

df = pd.read_parquet("ris_sentences.parquet")
abgb = df[df["law_type"] == "ABGB"].copy()

print("ABGB rows:", len(abgb))
print("ABGB pages min/max:", int(abgb["page"].min()), int(abgb["page"].max()))

s = abgb["sentence"].astype(str)

tests = {
    "irrtum": r"\birrtum\b",
    "irrthum": r"\birrthum\b",
    "anfecht*": r"\banfecht\w*\b",
    "§871": r"§\s*871",
    "871": r"\b871\b",
    "§872": r"§\s*872",
    "§873": r"§\s*873",
}

for name, pat in tests.items():
    print(f"{name:10s}:", count_matches(s, pat))

print("\nSamples for §871 / 871 / irrtum / irrthum (first 5 each):")
for name, pat in [("§871", tests["§871"]), ("871", tests["871"]), ("irrtum", tests["irrtum"]), ("irrthum", tests["irrthum"])]:
    r = re.compile(pat, re.I)
    hits = abgb[s.map(lambda x: bool(r.search(norm(str(x)))))].head(5)
    print(f"\n-- {name} ({len(hits)}) --")
    for t in hits["sentence"].tolist():
        print(t[:220])
