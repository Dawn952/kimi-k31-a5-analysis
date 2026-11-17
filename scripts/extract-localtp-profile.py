"""Extract small, de-identified dependency evidence from an existing A5 capture.

Only the first module windows and aggregate intervals are published. Timestamps
are relative; no raw trace, process metadata, host, paths or tensor values leave
the capture directory. Source semantics are checked against its pinned revision.
"""
import argparse
import csv
import json
import statistics
from decimal import Decimal
from pathlib import Path


def extract(csv_path):
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    rows = []
    for line, row in enumerate(raw, 2):
        rows.append(dict(row=line, type=row["Type"],
                         stream=row["Stream ID"], core=row["Accelerator Core"],
                         start=Decimal(row["Start Time(us)"].strip()),
                         duration=float(row["Duration(us)"]), output=row["Output Shapes"].strip('"')))
    rows.sort(key=lambda row: row["start"])
    blocks = []
    starts = [i for i, row in enumerate(rows) if row["type"] == "AttnResFwd"]
    for begin, end in zip(starts, starts[1:]):
        block = rows[begin:end]
        kinds = {row["type"] for row in block}
        family = ("kda" if "RecurrentKda" in kinds else "dsa" if "QuantLightningIndexer" in kinds
                  else "shared" if "MoeGatingTopK" in kinds
                  else "dense" if sum(row["type"] == "QuantBatchMatmulV3" for row in block) == 4
                  and "MlaPrologV3" not in kinds else None)
        if family:
            blocks.append((family, block))
    for i, row in enumerate(rows):
        if row["type"] == "RmsNorm":
            block = rows[i:i + 18]
            if sum(q["type"] == "MatMulV3" and q["output"] == "64,20480" for q in block) == 2:
                end = next(j for j, q in enumerate(block) if q["type"] == "_gather_draft_sampled_kernel")
                blocks.append(("head", block[:end]))
        if row["type"] == "GatherV2" and row["output"] == "512,7168":
            # The first lookup in each pair has two preceding IDs AGs.
            prior = rows[max(0, i - 18):i]
            ags = [q for q in prior if q["type"] == "hcom_allGather_"]
            if len(ags) != 2:
                continue
            start = rows.index(ags[0])
            stop = next(j for j in range(i + 1, min(i + 40, len(rows)))
                        if rows[j]["type"] == "hcom_reduceScatter_AicpuKernel"
                        and any(q["type"] == "hcom_reduceScatter_AicpuKernel" for q in rows[i:j]))
            blocks.append(("embedding", rows[start:stop + 1]))

    def annotate(block, family):
        types = lambda name: [i for i, row in enumerate(block) if row["type"] == name]
        groups = {"in": [] if family == "embedding" else [0]}
        if family == "kda":
            mm, q = types("MatMulV3"), types("QuantBatchMatmulV3")
            groups.update(f=mm[:2], ag=types("hcom_allGather_"), g=[mm[2]], quant=types("DynamicMxQuant"),
                          beta=[mm[3]], gr=[types("hcom_alltoall_")[0]], q=q,
                          cat=types("ConcatD"), zero=types("ZerosLike"), ret=[types("hcom_alltoall_")[1]],
                          restore=[i for i in range(types("AsStrided")[0], types("CausalConv1d")[0])],
                          conv=types("CausalConv1d"), core=types("RecurrentKda"),
                          grestore=[types("Transpose")[-2]], norm=types("layer_norm_gated_fwd_kernel"),
                          olayout=[types("Transpose")[-1]], oa=[types("hcom_alltoall_")[-1]],
                          o=[mm[-1]], rs=types("hcom_reduceScatter_"))
        elif family == "dsa":
            mm = types("MatMulV3")
            prolog = types("MlaPrologV3")[0]
            g = next(i for i in mm if block[i]["output"] == "128,1536")
            qprep = [i for i, row in enumerate(block) if row["stream"] == "417"]
            keyproj = next(i for i in mm if block[i]["output"] == "16,160")
            idx = types("QuantLightningIndexer")[0]
            groups.update(ag=types("hcom_allGather_"), quant=types("DynamicMxQuant"), prolog=[prolog],
                          g=[g], keyproj=[keyproj], gr=[types("hcom_alltoall_")[0]], qprep=qprep,
                          kprep=[i for i in range(keyproj + 1, idx) if block[i]["core"] != "COMMUNICATION"
                                 and i not in qprep], idx=[idx], concat=types("ConcatD"),
                          sfa=types("KvQuantSparseFlashAttentionVllm"), uv=types("TransposeBatchMatMul"),
                          grestore=[types("Transpose")[0]], gate=types("_flash_attention_gate_kernel"),
                          olayout=[types("Transpose")[-1]], oa=[types("hcom_alltoall_")[-1]],
                          o=[mm[-1]], rs=types("hcom_reduceScatter_"))
        elif family == "shared":
            q = types("QuantBatchMatmulV3")
            groups.update(router=types("MatMulV3"), ag=types("hcom_allGather_"),
                          down_quant=[types("DynamicMxQuant")[0]], latent_down=[q[0]],
                          quant=[types("DynamicMxQuant")[1]], up=[q[1]], topk=types("MoeGatingTopK"),
                          dispatch=types("MoeDistributeDispatchV3"), gmm1=types("GroupedMatmulSituQuant"),
                          gmm2=types("GroupedMatmul"), combine=types("MoeDistributeCombineV3"),
                          act=types("SituMxQuant"), down=[q[2]], rs=types("hcom_reduceScatter_"),
                          up_quant=types("RmsNormDynamicMxQuant"), latent_up=[q[3]], add=types("ForeachAddList"))
        elif family == "dense":
            q, ag, rs = types("QuantBatchMatmulV3"), types("hcom_allGather_"), types("hcom_reduceScatter_")
            dq = types("DynamicMxQuant")
            groups.update(ag0=[ag[0]], ag1=[ag[1]], quant0=[dq[0]], up0=[q[0]],
                          act0=list(range(q[0] + 1, q[1])), down0=[q[1]],
                          quant1=[dq[2]], up1=[q[2]], act1=list(range(q[2] + 1, q[3])),
                          down1=[q[3]], rs0=[rs[0]], rs1=[rs[1]])
        elif family == "head":
            mm, ag, a2a, tr = types("MatMulV3"), types("hcom_allGather_"), types("hcom_alltoall_"), types("Transpose")
            groups.update(ag0=[ag[0]], ag1=[ag[1]], mm0=[mm[0]], mm1=[mm[1]], a2a0=[a2a[0]],
                          a2a1=[a2a[1]], restore0=[tr[0]], restore1=[tr[1]], join=types("ConcatD"))
        else:
            ag, lookup, mask, rs = (types(name) for name in ("hcom_allGather_", "GatherV2", "MaskedFill", "hcom_reduceScatter_AicpuKernel"))
            compute = lambda indices: [i for i in indices if block[i]["core"] in ("AI_VECTOR_CORE", "MIX_AIV")]
            groups.update(ag0=[ag[0]], ag1=[ag[1]], lookup0=compute(range(ag[1] + 1, mask[0] + 1)),
                          lookup1=compute(range(rs[0] + 1, mask[1] + 1)), rs0=[rs[0]], rs1=[rs[1]])
        return groups

    pairs = {"kda": [("ag", "f"), ("quant", "beta"), ("gr", "quant"), ("ret", "zero"), ("gr", "core"), ("ret", "g")],
             "dsa": [("ag", "prolog"), ("gr", "idx"), ("gr", "keyproj"), ("qprep", "kprep")],
             "shared": [("ag", "latent_down"), ("rs", "latent_up"), ("up", "gmm1"), ("down", "combine")],
             "dense": [("ag1", "up0"), ("rs0", "up1")],
             "head": [("ag1", "mm0"), ("a2a0", "mm1")], "embedding": [("rs0", "lookup1")]}
    result = {}
    for family in pairs:
        selected = [block for kind, block in blocks if kind == family]
        samples = []
        for block in selected:
            groups = annotate(block, family)
            origin = block[0]["start"]
            events = [{**row, "start_us": round(float(row["start"] - origin), 3)} for row in block]
            for row in events:
                del row["start"]
            overlaps = {}
            for left, right in pairs[family]:
                overlaps[left + ":" + right] = round(sum(max(0, min(events[a]["start_us"] + events[a]["duration"],
                    events[b]["start_us"] + events[b]["duration"]) - max(events[a]["start_us"], events[b]["start_us"]))
                    for a in groups[left] for b in groups[right]), 3)
            samples.append(dict(events=events, groups=groups, overlaps=overlaps))
        first = samples[0]
        metrics = {key: {"median_us": round(statistics.median(sample["overlaps"][key] for sample in samples), 3),
                         "min_us": min(sample["overlaps"][key] for sample in samples),
                         "max_us": max(sample["overlaps"][key] for sample in samples)} for key in first["overlaps"]}
        durations = {key: round(statistics.median(sum(sample["events"][i]["duration"] for i in sample["groups"][key])
                                                for sample in samples), 3) for key in first["groups"]}
        result[family] = dict(sample_count=len(samples), first=first, overlap_stats=metrics, kernel_sum_median_us=durations)
    return dict(schema=1, date="2026-11-13", hardware="A5", phase="warmup-c128", worker_steps=10,
                ascend_revision="3634a7eae94897d1a7b14d00807323e5af6b4a04",
                vllm_revision="4ba787ec9959e566d3b587a796dff07673324da5",
                configuration=dict(dp=32, tp=1, localtp=8, concurrency=128, input_tokens=131072,
                                   output_tokens=1024, speculative_steps=3, fixed_acceptance=2.7,
                                   multistream_overlap_shared_expert=True),
                source_csv="kernel_details.csv", kernel_records=len(rows),
                scope="One rank; first module windows and aggregate device interval intersections. Intersections are not end-to-end gains.",
                domains=result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/localtp-profile-20261113.json")
    args = parser.parse_args()
    evidence = extract(args.kernel_csv)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: {"samples": value["sample_count"], "overlaps": value["overlap_stats"]}
                      for key, value in evidence["domains"].items()}, ensure_ascii=False, indent=2))
