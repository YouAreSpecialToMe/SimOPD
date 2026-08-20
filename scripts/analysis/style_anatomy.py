"""四方风格解剖:教师 / 初始学生 / c2@250 / c4@250(+vanilla@250 病态对照)。
逐响应测:长度、结构标记、验算口癖、答案位置、答案后余量、重复度;并抽真实样本。"""
import glob, re
import pandas as pd
D = "/mgfs/shared/Group_GY/changhao/simopd_data"
CELLS = [("教师 4B-Instruct", f"{D}/evals_style/teacher_greedy_text__math500__step-1__*.parquet"),
         ("初始学生 1.7B-Base", f"{D}/evals_style/base_greedy_text__math500__step-1__*.parquet"),
         ("c2@250(健康)", f"{D}/evals_style/c2_greedy_text__math500__step250__*.parquet"),
         ("c4@250(健康)", f"{D}/evals_style/c4_greedy_text__math500__step250__*.parquet"),
         ("vanilla@250(病态)", f"{D}/evals_vanilla_sweep/vanilla_s0_16k__math500__step250__*.parquet")]
def distinct_ngram(t, n=20):
    w = t.split()
    if len(w) < n + 5: return 1.0
    G = [" ".join(w[i:i+n]) for i in range(len(w)-n)]
    return len(set(G)) / len(G)
MARK = {"### 标题": r"^#{2,3} ", "**加粗小节": r"\*\*[A-Z][^*]{2,40}\*\*", "$$display": r"\$\$",
        "--- 分隔": r"^---\s*$", "python代码块": r"```python", "✅": r"✅",
        "Final Answer 模板": r"\*\*Final Answer", "Wait 口癖": r"\bWait\b",
        "verify/check 口癖": r"\b([Vv]erify|[Dd]ouble-check|[Ll]et me check|[Ll]et us check)\b",
        "We are done": r"We are done"}
rows, samples = [], {}
for name, pat in CELLS:
    fs = sorted(glob.glob(pat))
    if not fs: print(f"{name}: MISSING"); continue
    df = pd.read_parquet(fs[-1])
    T = df.response.astype(str)
    fin = df.truncated == 0
    r = {"模型": name, "n": len(df), "score": df.correct.mean(), "trunc%": 100*(~fin).mean(),
         "len中位": df.resp_len.median(), "len p90": df.resp_len.quantile(.9)}
    fb = T.str.find("\\boxed")
    tot = T.str.len().clip(lower=1)
    r["有boxed%"] = 100*(fb >= 0).mean()
    r["首boxed位置/全长"] = (fb[fb >= 0]/tot[fb >= 0]).median()
    r["boxed次数中位"] = T.str.count(r"\\boxed").median()
    tail_frac = 1 - (fb[(fb >= 0) & fin]/tot[(fb >= 0) & fin])
    r["答后余量中位(完成的)"] = tail_frac.median() if len(tail_frac) else float("nan")
    r["尾部distinct20gram"] = T.map(lambda t: distinct_ngram(t[-4000:])).median()
    for k, rx in MARK.items():
        flags = re.M if rx.startswith("^") else 0
        if "口癖" in k or k == "We are done":
            r[k+"/篇"] = T.map(lambda t: len(re.findall(rx, t, flags))).mean()
        else:
            r[k] = 100*T.map(lambda t: bool(re.search(rx, t, flags))).mean()
    rows.append(r)
    med = df[fin & (df.correct == 1)] if (fin & (df.correct==1)).any() else df
    med = med.iloc[(med.resp_len - med.resp_len.median()).abs().argsort()[:1]]
    if len(med):
        t = str(med.response.iloc[0])
        samples[name] = (med.problem_id.iloc[0], int(med.resp_len.iloc[0]), t[:300], t[-300:])
out = pd.DataFrame(rows).set_index("模型")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(out.round(3).to_string())
print("\n================ 真实样本(各取一条中位长度的答对响应:开头 300 / 结尾 300 字符)")
for name, (pid, ln, head, tail) in samples.items():
    print(f"\n----- {name} | {pid} | {ln} tokens")
    print("[开头]", repr(head))
    print("[结尾]", repr(tail))
print("STYLE_ANATOMY_DONE")
