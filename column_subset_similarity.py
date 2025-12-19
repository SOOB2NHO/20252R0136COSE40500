#!/usr/bin/env python3
"""
Latex 포맷을 사용하여 컬럼 부분집합의 쿼리-테이블 유사도 분석

- 모든 가능한 컬럼 부분집합(2^n)을 생성
- 각 부분집합에 대해 쿼리-테이블 유사도 계산
- 최고 유사도 부분집합(best_subset) 식별
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

# Unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def latex_table_str(table_array: Sequence[Sequence]) -> str:
    """테이블 배열을 LaTeX tabular 형식으로 변환."""
    if not table_array:
        return ""
    
    headers = table_array[0]
    data_rows = table_array[1:]
    
    if not headers:
        return ""
    
    num_cols = len(headers)
    
    # LaTeX 특수 문자 이스케이프
    def escape_latex(text: str) -> str:
        if text is None:
            return ""
        text = str(text)
        # LaTeX 특수 문자 이스케이프
        text = text.replace("\\", "\\textbackslash{}")
        text = text.replace("{", "\\{")
        text = text.replace("}", "\\}")
        text = text.replace("$", "\\$")
        text = text.replace("&", "\\&")
        text = text.replace("%", "\\%")
        text = text.replace("#", "\\#")
        text = text.replace("^", "\\textasciicircum{}")
        text = text.replace("_", "\\_")
        text = text.replace("~", "\\textasciitilde{}")
        return text
    
    # 헤더 변환
    header_strs = [escape_latex(h) if h is not None else "" for h in headers]
    
    # LaTeX tabular 시작
    latex = "\\begin{tabular}{|" + "c|" * num_cols + "}\n"
    latex += "\\hline\n"
    
    # 헤더 행
    latex += " & ".join(header_strs) + " \\\\\n"
    latex += "\\hline\n"
    
    # 데이터 행 변환
    for row in data_rows:
        row_strs = [escape_latex(item) if item is not None else "" for item in row]
        # 행 길이가 컬럼 수보다 적으면 빈 문자열로 채움
        while len(row_strs) < num_cols:
            row_strs.append("")
        latex += " & ".join(row_strs[:num_cols]) + " \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}"
    
    return latex


def compute_subset_similarities(
    question: str,
    table_array: Sequence[Sequence],
    model: SentenceTransformer,
    relevant_columns: Sequence[str] | None = None,
) -> List[Dict]:
    """테이블의 모든 컬럼 부분집합에 대해 쿼리-테이블 유사도를 계산 (LaTeX 버전)."""
    if not table_array or len(table_array) < 2:
        raise ValueError("table_array는 최소 헤더와 한 개 이상의 데이터 행을 포함해야 합니다.")

    headers = table_array[0]
    rows = table_array[1:]
    num_cols = len(headers)
    
    # 컬럼 수가 20개 초과면 에러 발생 (호출부에서 스킵하도록)
    if num_cols > 20:
        raise ValueError(f"컬럼 수가 너무 많습니다 ({num_cols}개). 최대 20개까지 지원합니다.")

    query_emb = model.encode(question, normalize_embeddings=True)

    rel_columns_set = set(relevant_columns or [])

    subset_infos: List[Dict] = []
    for bits in range(1 << num_cols):
        selected_cols = [
            headers[idx]
            for idx in range(num_cols)
            if (bits >> idx) & 1
        ]

        selected_rows = [
            [
                row[idx] if idx < len(row) else ""
                for idx in range(num_cols)
                if (bits >> idx) & 1
            ]
            for row in rows
        ]

        table_subarray = [selected_cols] + selected_rows if selected_cols else [[]]
        latex_str = latex_table_str(table_subarray)
        table_emb = model.encode(
            latex_str,
            normalize_embeddings=True,
        )

        similarity = float(np.dot(query_emb, table_emb))
        is_relevant = (
            bool(rel_columns_set)
            and len(selected_cols) == len(rel_columns_set)
            and set(selected_cols) == rel_columns_set
        )

        subset_infos.append(
            {
                "bitmask": bits,
                "columns": selected_cols,
                "size": len(selected_cols),
                "similarity": similarity,
                "is_relevant": is_relevant,
            }
        )

    subset_infos.sort(key=lambda item: item["similarity"], reverse=True)
    return subset_infos


def plot_distributions(
    subset_infos: Sequence[Dict],
    output_dir: Path,
    histogram_name: str = "dev_first_table_similarity_latex.png",
    scatter_name: str = "dev_first_table_similarity_by_size_latex.png",
) -> None:
    """유사도 분포 및 컬럼 개수 대비 유사도 산점도를 저장."""
    output_dir.mkdir(parents=True, exist_ok=True)

    similarities = [entry["similarity"] for entry in subset_infos]
    sizes = [entry["size"] for entry in subset_infos]
    relevant_entries = [entry for entry in subset_infos if entry.get("is_relevant")]

    histogram_path = output_dir / histogram_name
    plt.figure(figsize=(8, 5))
    plt.hist(similarities, bins=10, color="#4e79a7", edgecolor="black", alpha=0.75, label="All subsets")
    if relevant_entries:
        rel_sims = [entry["similarity"] for entry in relevant_entries]
        plt.hist(
            rel_sims,
            bins=10,
            color="#f28e2b",
            edgecolor="black",
            alpha=0.85,
            label="Relevant-only subset",
        )
    plt.title("Similarity Distribution (All vs. Relevant Subset) - LaTeX")
    plt.xlabel("Cosine Similarity (normalized embeddings)")
    plt.ylabel("Count")
    if relevant_entries:
        plt.legend()
    plt.tight_layout()
    plt.savefig(histogram_path)
    plt.close()

    scatter_path = output_dir / scatter_name
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(sizes, similarities, c=similarities, cmap="viridis", s=60, label="All subsets")
    plt.colorbar(scatter, label="Similarity")
    if relevant_entries:
        plt.scatter(
            [entry["size"] for entry in relevant_entries],
            [entry["similarity"] for entry in relevant_entries],
            color="#d62728",
            s=120,
            edgecolors="black",
            label="Relevant-only subset",
        )
    if sizes:
        plt.xticks(range(0, max(sizes) + 1))
    plt.title("Similarity vs. Number of Columns Included (LaTeX)")
    plt.xlabel("Subset Size (# of columns)")
    plt.ylabel("Cosine Similarity")
    plt.grid(alpha=0.3)
    if relevant_entries:
        plt.legend()
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()

    print(f"📊 저장 완료: {histogram_path}")
    print(f"📊 저장 완료: {scatter_path}")


def parse_log_file(log_path: Path) -> List[Dict]:
    """로그 파일에서 기존 계산 결과를 파싱.
    
    Returns:
        기존 table_results 형식의 리스트 (subsets는 빈 리스트, best/relevant subset만 포함)
    """
    if not log_path.exists():
        return []
    
    results: List[Dict] = []
    current_table: Optional[Dict] = None
    
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            # Table 시작
            table_match = re.match(r"📄 Table (\d+) \(feta_id: (.+)\)", line)
            if table_match:
                if current_table:
                    results.append(current_table)
                
                table_num = int(table_match.group(1))
                feta_id = table_match.group(2)
                current_table = {
                    "table_num": table_num,
                    "feta_id": feta_id,
                    "record": {"feta_id": int(feta_id) if feta_id.isdigit() else None},
                    "subsets": [],  # 로그에는 모든 부분집합 정보가 없음
                    "best_subset": None,
                    "relevant_subset": None,
                }
                continue
            
            if not current_table:
                continue
            
            # 최고 유사도 파티션
            best_match = re.search(
                r"🔝 최고 유사도 파티션: sim=([\d.]+) \(size=(\d+)\) \| cols=\[(.+?)\]",
                line
            )
            if best_match:
                sim = float(best_match.group(1))
                size = int(best_match.group(2))
                cols_str = best_match.group(3)
                # 컬럼 리스트 파싱
                if cols_str == "<empty>":
                    cols = []
                else:
                    # 쉼표로 구분하고 따옴표 제거
                    cols = [c.strip().strip("'\"") for c in cols_str.split(",") if c.strip()]
                current_table["best_subset"] = {
                    "similarity": sim,
                    "size": size,
                    "columns": cols,
                }
                continue
            
            # relevant columns 파티션
            rel_match = re.search(
                r"🎯 relevant columns 파티션: sim=([\d.]+) \(size=(\d+)\) \| cols=\[(.+?)\]",
                line
            )
            if rel_match:
                sim = float(rel_match.group(1))
                size = int(rel_match.group(2))
                cols_str = rel_match.group(3)
                if cols_str == "<empty>":
                    cols = []
                else:
                    cols = [c.strip().strip("'\"") for c in cols_str.split(",") if c.strip()]
                current_table["relevant_subset"] = {
                    "similarity": sim,
                    "size": size,
                    "columns": cols,
                }
                continue
    
    # 마지막 테이블 추가
    if current_table:
        results.append(current_table)
    
    return results


def load_records(jsonl_path: Path, limit: int, start_from: int = 1) -> List[Dict]:
    """JSONL 파일에서 레코드를 로드.
    
    Args:
        jsonl_path: JSONL 파일 경로
        limit: 로드할 최대 레코드 수 (0이면 전체)
        start_from: 시작할 레코드 번호 (1부터 시작, 이전 레코드는 스킵)
    """
    records: List[Dict] = []
    skipped = 0
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"⚠️ JSON decode error (line {line_num}): {exc}")
                continue

            input_block = data.get("input") or {}
            question = input_block.get("question")
            table_array = input_block.get("table_array")
            if not question or not table_array:
                continue

            # start_from 이전 레코드는 스킵
            skipped += 1
            if skipped < start_from:
                continue

            # limit이 설정되어 있고 이미 충분한 레코드를 로드했으면 중단
            if limit and len(records) >= limit:
                break

            relevant_columns = (
                data.get("output", {}).get("relevant_columns")
                or data.get("output", {}).get("relevant_columns_flat")
            )

            records.append(
                {
                    "question": question,
                    "table_array": table_array,
                    "relevant_columns": relevant_columns,
                    "feta_id": data.get("feta_id"),
                    "instance_id": data.get("instance_id"),
                    "raw": data,
                }
            )
    
    if start_from > 1:
        print(f"⏭️  처음 {start_from - 1}개 레코드를 스킵했습니다.")
        sys.stdout.flush()
    
    return records


def summarize(subset_infos: Sequence[Dict], top_k: int = 5) -> None:
    """상·하위 k개의 부분집합 요약 출력."""
    print("\n🔝 Top subsets by similarity:")
    sys.stdout.flush()
    for entry in list(subset_infos)[:top_k]:
        cols = entry["columns"] if entry["columns"] else ["<empty>"]
        rel_flag = " *relevant*" if entry.get("is_relevant") else ""
        print(f"  size={entry['size']:>2} | sim={entry['similarity']:.4f} | cols={cols}{rel_flag}")
        sys.stdout.flush()

    print("\n🔻 Bottom subsets by similarity:")
    sys.stdout.flush()
    for entry in list(subset_infos)[-top_k:]:
        cols = entry["columns"] if entry["columns"] else ["<empty>"]
        rel_flag = " *relevant*" if entry.get("is_relevant") else ""
        print(f"  size={entry['size']:>2} | sim={entry['similarity']:.4f} | cols={cols}{rel_flag}")
        sys.stdout.flush()

    rel_entries = [entry for entry in subset_infos if entry.get("is_relevant")]
    if rel_entries:
        print("\n🎯 Relevant-only subset statistics:")
        sys.stdout.flush()
        for entry in rel_entries:
            print(f"  similarity={entry['similarity']:.4f} | columns={entry['columns']}")
            sys.stdout.flush()
    else:
        print("\n⚠️ Relevant-only subset을 찾지 못했습니다.")
        sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute query similarity distribution across all column subsets (LaTeX version)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/subeen/DaisLab/SACU/data/SACU/SACU_dev.jsonl"),
        help="JSONL 파일 경로 (기본: data/SACU/SACU_dev.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/subeen/DaisLab/SACU/data/SACU"),
        help="시각화 결과 저장 디렉터리 (기본: 데이터 디렉터리)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="NovaSearch/stella_en_400M_v5",
        help="SentenceTransformer 모델 (기본: NovaSearch/stella_en_400M_v5)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="상/하위 출력 개수 (기본: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="분석할 상위 테이블 수 (0이면 전체 테이블 사용)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="시작할 테이블 번호 (1부터 시작, 이미 처리된 테이블을 스킵)",
    )
    parser.add_argument(
        "--violin-name",
        type=str,
        default="dev_top_tables_similarity_violin_latex.png",
        help="상위 테이블 분포 그래프 파일명",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input, args.limit, args.start_from)
    if not records:
        raise ValueError("분석할 레코드를 찾지 못했습니다.")

    model_kwargs: Dict[str, Any] = {}
    if "stella" in args.model.lower():
        model_kwargs["trust_remote_code"] = True
    model = SentenceTransformer(args.model, **model_kwargs)

    table_results: List[Dict] = []
    
    # start_from을 고려하여 인덱스 조정
    start_idx = args.start_from
    for idx, record in enumerate(records, start=start_idx):
        print(f"\n{'='*80}")
        sys.stdout.flush()
        feta_id = record.get("feta_id")
        label = feta_id if feta_id is not None else f"index-{idx}"
        print(f"📄 Table {idx} (feta_id: {label}) [LaTeX]")
        sys.stdout.flush()
        print(f"📝 Question: {record['question']}")
        sys.stdout.flush()

        table_array = record["table_array"]
        relevant_columns = record.get("relevant_columns") or []
        headers = table_array[0] if table_array else []
        num_cols = len(headers)
        print(f"📋 Headers ({num_cols}): {headers}")
        sys.stdout.flush()
        
        # 컬럼 수가 20개 초과면 스킵
        if num_cols > 20:
            print(f"⚠️  컬럼 수가 너무 많습니다 ({num_cols}개 > 20개). 이 테이블을 스킵합니다.")
            sys.stdout.flush()
            continue
        
        if relevant_columns:
            print(f"🎯 Relevant columns: {relevant_columns}")
            sys.stdout.flush()
        else:
            print("⚠️ Relevant columns 정보가 없습니다.")
            sys.stdout.flush()

        try:
            subset_infos = compute_subset_similarities(
                record["question"],
                table_array,
                model=model,
                relevant_columns=relevant_columns,
            )
        except ValueError as e:
            print(f"⚠️  에러 발생: {e}. 이 테이블을 스킵합니다.")
            sys.stdout.flush()
            continue

        best_subset = subset_infos[0] if subset_infos else None
        relevant_subset = next((entry for entry in subset_infos if entry.get("is_relevant")), None)

        if best_subset:
            print(
                f"   🔝 최고 유사도 파티션: sim={best_subset['similarity']:.4f} "
                f"(size={best_subset['size']}) | cols={best_subset['columns'] or ['<empty>']}"
            )
            sys.stdout.flush()
        if relevant_subset:
            print(
                f"   🎯 relevant columns 파티션: sim={relevant_subset['similarity']:.4f} "
                f"(size={relevant_subset['size']}) | cols={relevant_subset['columns']}"
            )
            sys.stdout.flush()
        else:
            print("   ⚠️ Relevant columns 조합과 일치하는 파티션이 없습니다.")
            sys.stdout.flush()

        summarize(subset_infos, top_k=args.top_k)

        table_results.append(
            {
                "record": record,
                "subsets": subset_infos,
                "best_subset": best_subset,
                "relevant_subset": relevant_subset,
            }
        )

    # 상위 테이블 전체 분포 그래프
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 분포 데이터 생성 (subsets가 있는 경우만, 로그에서 파싱한 경우는 빈 리스트)
    distributions = []
    for result in table_results:
        if result.get("subsets"):
            # 실제 계산된 subsets가 있는 경우
            dist = [entry["similarity"] for entry in result["subsets"]]
        else:
            # 로그에서 파싱한 경우: best와 relevant subset만으로 근사 분포 생성
            # (실제 분포는 아니지만 점 표시는 가능)
            dist = []
            if result.get("best_subset"):
                dist.append(result["best_subset"]["similarity"])
            if result.get("relevant_subset"):
                dist.append(result["relevant_subset"]["similarity"])
            if not dist:
                dist = [0.0]
        distributions.append(dist)
    positions = np.arange(1, len(distributions) + 1)

    plt.figure(figsize=(max(12, len(distributions) * 1.3), 6))
    violin_parts = plt.violinplot(
        distributions,
        positions=positions,
        showmeans=True,
        showextrema=False,
    )
    for body in violin_parts["bodies"]:
        body.set_facecolor("#4e79a7")
        body.set_alpha(0.45)
    if "cmeans" in violin_parts:
        violin_parts["cmeans"].set_edgecolor("#2f4b7c")
        violin_parts["cmeans"].set_linewidth(1.5)

    # 최고 파티션과 relevant 파티션 점 표시
    best_x: List[float] = []
    best_y: List[float] = []
    relevant_x: List[float] = []
    relevant_y: List[float] = []

    for pos, result in zip(positions, table_results):
        best = result["best_subset"]
        if best:
            best_x.append(pos)
            best_y.append(best["similarity"])
        rel = result["relevant_subset"]
        if rel:
            relevant_x.append(pos)
            relevant_y.append(rel["similarity"])

    legend_handles = []
    legend_labels = []
    if best_x:
        best_scatter = plt.scatter(
            best_x,
            best_y,
            color="#1f77b4",
            marker="D",
            s=70,
            label="Best subset",
        )
        legend_handles.append(best_scatter)
        legend_labels.append("Best subset")
    if relevant_x:
        rel_scatter = plt.scatter(
            relevant_x,
            relevant_y,
            color="#d62728",
            s=90,
            edgecolors="black",
            label="Relevant subset",
        )
        legend_handles.append(rel_scatter)
        legend_labels.append("Relevant subset")

    table_labels: List[str] = []
    for result in table_results:
        record = result.get("record", {})
        # 로그에서 파싱한 경우 table_num 사용
        if "table_num" in result:
            label = str(result.get("feta_id", result.get("table_num", "")))
        else:
            label = record.get("feta_id") or record.get("instance_id") or ""
        table_labels.append(str(label))

    plt.xticks(positions, table_labels, rotation=45, ha="right")
    plt.xlabel("Table (feta_id or index)")
    plt.ylabel("Cosine Similarity")
    plt.title("Top Dev Tables: Similarity Distribution Across Column Subsets (LaTeX)")
    plt.grid(axis="y", alpha=0.3)
    if legend_handles:
        plt.legend(legend_handles, legend_labels, loc="best")

    violin_path = output_dir / args.violin_name
    plt.tight_layout()
    plt.savefig(violin_path)
    plt.close()
    print(f"\n📊 상위 테이블 분포 그래프 저장 완료: {violin_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

