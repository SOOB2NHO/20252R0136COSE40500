#!/usr/bin/env python3
# analyze_query_query_similarity.py
# 
# Original query와 LLM query들의 유사도 비교 분석
# - 각 테이블(original query)에 대해:
#   1. 같은 테이블의 LLM query들(정답)과의 유사도 계산
#   2. 정답 LLM query들 중 최대 유사도(max)를 기준으로 사용
#   3. 다른 테이블의 LLM query들(비정답)과의 유사도 계산
#   4. 비정답 LLM query들 중 최대 유사도(max)를 통계 비교에 사용
#   5. 정답의 최대 유사도보다 높은 비정답 쿼리 개수 계산
# - 목적: LLM이 생성한 query가 original query와 얼마나 유사한지, 
#         그리고 retrieval에서 혼동될 가능성이 있는지 분석

import os
import json
import numpy as np
from typing import List, Dict, Any
import torch
import random
from pathlib import Path
import sys

# 환경 변수 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()

# SACU model 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "model"))
from encoder import SacuTableEncoder


def load_qa_subset(file_path: str) -> List[Dict]:
    """QA subset 파일 로딩 (LLM이 생성한 query들)
    
    Args:
        file_path: dev_QA_subset.jsonl 파일 경로
        
    Returns:
        각 항목은 {"feta_id", "query"}를 포함하는 딕셔너리 리스트
        각 테이블당 약 5개의 LLM 생성 query가 있음
    """
    items = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            items.append({
                "feta_id": data["feta_id"],
                "query": data["input"]["question"],
            })
    
    return items


def load_original_dev(file_path: str) -> List[Dict]:
    """Original dev 파일 로딩 (원본 query)
    
    Args:
        file_path: SACU_dev.jsonl 파일 경로
        
    Returns:
        각 항목은 {"feta_id", "query"}를 포함하는 딕셔너리 리스트
        이 query들이 평가 기준이 됨
    """
    items = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            items.append({
                "feta_id": data["feta_id"],
                "query": data["input"]["question"],
            })
    
    return items


def analyze_query_query_similarity(
    eval_queries: List[str],
    eval_feta_ids: List[int],
    qa_items: List[Dict],
    encoder: SacuTableEncoder,
    num_samples: int = None
):
    """Original query와 LLM query들의 유사도 비교 분석
    
    각 original query에 대해:
    - 정답: 같은 feta_id를 가진 LLM query들 (각 테이블당 약 5개)
    - 비정답: 다른 feta_id를 가진 LLM query들
    
    Args:
        eval_queries: 평가 기준이 되는 original query 리스트
        eval_feta_ids: 각 query에 대응하는 테이블 ID 리스트
        qa_items: LLM이 생성한 query 리스트 (각 항목은 feta_id 포함)
        encoder: 임베딩을 생성할 encoder
        num_samples: 분석할 샘플 수 (None이면 전체)
        
    Returns:
        results 딕셔너리:
            - correct_query_similarities: 정답 LLM query들과의 유사도 리스트
            - incorrect_query_similarities: 비정답 LLM query들과의 최대 유사도 리스트
            - differences: 정답-비정답 유사도 차이 리스트
            - higher_incorrect_counts: 각 테이블당 정답보다 높은 비정답 쿼리 개수
    """
    
    results = {
        # 각 테이블당 정답 LLM query들(약 5개)의 최대/평균 유사도
        "correct_max_similarities": [],  # 각 테이블의 정답 LLM query 최대 유사도
        "correct_avg_similarities": [],  # 각 테이블의 정답 LLM query 평균 유사도
        # 각 테이블당 비정답 LLM query들(다른 테이블의 모든 query)의 최대/평균 유사도
        "incorrect_max_similarities": [],  # 각 테이블의 비정답 LLM query 최대 유사도
        "incorrect_avg_similarities": [],  # 각 테이블의 비정답 LLM query 평균 유사도
        # 각 테이블의 정답-비정답 차이
        "max_differences": [],  # 각 테이블의 최대 유사도 차이 (correct_max - incorrect_max)
        "avg_differences": [],  # 각 테이블의 평균 유사도 차이 (correct_avg - incorrect_avg)
        # 각 테이블에서 정답보다 높은 비정답 쿼리 개수
        "higher_incorrect_counts_max": [],  # 정답 max보다 높은 비정답 쿼리 개수 (테이블당)
        "higher_incorrect_counts_avg": [],  # 정답 avg보다 높은 비정답 쿼리 개수 (테이블당)
        "num_tables": 0,  # 분석한 테이블 수 (마지막에 이 테이블들의 평균을 계산)
    }
    
    if num_samples is None:
        num_samples = len(eval_queries)
    else:
        num_samples = min(num_samples, len(eval_queries))
    
    print(f"\n{'='*60}")
    print(f"📊 Original query vs LLM query 유사도 비교 ({num_samples}개 샘플)")
    print(f"   - 정답: 같은 테이블의 LLM query들")
    print(f"   - 비정답: 다른 테이블의 LLM query들")
    print(f"{'='*60}\n", flush=True)
    
    # 평가 쿼리 임베딩 생성 (original dev의 query들)
    print(f"🔄 Original query 임베딩 생성 중... ({num_samples}개)")
    eval_query_embeddings = encoder.model.encode(
        eval_queries[:num_samples],
        convert_to_tensor=False,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )
    
    # QA subset의 모든 LLM query 임베딩 생성
    print(f"\n🔄 LLM query 임베딩 생성 중... ({len(qa_items)}개)")
    qa_queries = [item["query"] for item in qa_items]
    qa_feta_ids = [item["feta_id"] for item in qa_items]
    qa_query_embeddings = encoder.model.encode(
        qa_queries,
        convert_to_tensor=False,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )
    
    print(f"\n🔄 유사도 계산 중...")
    
    # 각 original query에 대해 분석
    for idx in range(num_samples):
        eval_query_emb = eval_query_embeddings[idx]  # 현재 original query의 임베딩
        eval_feta_id = eval_feta_ids[idx]  # 현재 테이블 ID
        
        # Step 1: 정답 LLM query 찾기
        # 같은 feta_id를 가진 LLM query들 (각 테이블당 약 5개)
        correct_query_embs = []
        for qa_idx, qa_feta_id in enumerate(qa_feta_ids):
            if qa_feta_id == eval_feta_id:
                correct_query_embs.append(qa_query_embeddings[qa_idx])
        
        if not correct_query_embs:
            # 정답 LLM query가 없으면 스킵
            continue
        
        # Step 2: 정답 LLM query들과의 유사도 계산
        correct_sims = [np.dot(eval_query_emb, correct_emb) for correct_emb in correct_query_embs]
        max_correct_sim = max(correct_sims)  # 정답 중 최대 유사도
        avg_correct_sim = np.mean(correct_sims)  # 정답 평균 유사도
        
        # Step 3: 비정답 LLM query 찾기
        # 다른 feta_id를 가진 LLM query들 (다른 테이블의 query들)
        incorrect_query_embs = []
        for qa_idx, qa_feta_id in enumerate(qa_feta_ids):
            if qa_feta_id != eval_feta_id:
                incorrect_query_embs.append(qa_query_embeddings[qa_idx])
        
        if not incorrect_query_embs:
            continue
        
        # Step 4: 비정답 LLM query들과의 유사도 계산
        incorrect_sims = [np.dot(eval_query_emb, incorrect_emb) for incorrect_emb in incorrect_query_embs]
        max_incorrect_sim = max(incorrect_sims)  # 비정답 중 최대 유사도
        avg_incorrect_sim = np.mean(incorrect_sims)  # 비정답 평균 유사도
        
        # Step 5: 정답보다 유사도가 높은 비정답 쿼리 개수 계산
        # 정답의 최대/평균 유사도보다 높은 비정답 쿼리가 몇 개나 있는지 계산
        higher_incorrect_count_max = sum(1 for sim in incorrect_sims if sim > max_correct_sim)
        higher_incorrect_count_avg = sum(1 for sim in incorrect_sims if sim > avg_correct_sim)
        
        # Step 6: 결과 저장
        results["correct_max_similarities"].append(max_correct_sim)
        results["correct_avg_similarities"].append(avg_correct_sim)
        results["incorrect_max_similarities"].append(max_incorrect_sim)
        results["incorrect_avg_similarities"].append(avg_incorrect_sim)
        results["max_differences"].append(max_correct_sim - max_incorrect_sim)
        results["avg_differences"].append(avg_correct_sim - avg_incorrect_sim)
        results["higher_incorrect_counts_max"].append(higher_incorrect_count_max)
        results["higher_incorrect_counts_avg"].append(higher_incorrect_count_avg)
        results["num_tables"] += 1
        
        # 진행 상황 출력
        if (idx + 1) % 100 == 0:
            print(f"   진행: {idx + 1}/{num_samples} 완료")
    
    return results


def print_statistics(results: Dict):
    """분석 결과 통계 출력
    
    출력 내용:
    - 정답/비정답 LLM query의 최대/평균 유사도 및 분포
    - 정답이 비정답보다 높은 비율 (max, avg 각각)
    - 각 테이블당 정답보다 유사도가 높은 비정답 쿼리 개수 (max/avg 기준)
    """
    print(f"\n{'='*60}")
    print(f"📈 Original query vs LLM query 유사도 비교 결과")
    print(f"{'='*60}\n")
    
    if results["num_tables"] == 0:
        print("❌ 비교할 데이터가 없습니다.")
        return
    
    num_tables = results["num_tables"]
    correct_max = np.array(results["correct_max_similarities"])
    correct_avg = np.array(results["correct_avg_similarities"])
    incorrect_max = np.array(results["incorrect_max_similarities"])
    incorrect_avg = np.array(results["incorrect_avg_similarities"])
    max_diffs = np.array(results["max_differences"])
    avg_diffs = np.array(results["avg_differences"])
    
    # 최대 유사도 통계
    # 각 테이블의 정답 LLM query 최대 유사도들의 평균
    print(f"🔹 정답 LLM query 최대 유사도:")
    print(f"   테이블당 평균: {np.mean(correct_max):.4f} (모든 테이블의 정답 max 평균)")
    print(f"   (표준편차: {np.std(correct_max):.4f})")
    print(f"   (최소값: {np.min(correct_max):.4f}, 최대값: {np.max(correct_max):.4f})")
    print()
    
    # 각 테이블의 비정답 LLM query 최대 유사도들의 평균
    print(f"🔹 비정답 LLM query 최대 유사도:")
    print(f"   테이블당 평균: {np.mean(incorrect_max):.4f} (모든 테이블의 비정답 max 평균)")
    print(f"   (표준편차: {np.std(incorrect_max):.4f})")
    print(f"   (최소값: {np.min(incorrect_max):.4f}, 최대값: {np.max(incorrect_max):.4f})")
    print()
    
    avg_max_diff = np.mean(max_diffs)
    print(f"📊 최대 유사도 차이:")
    print(f"   정답_max - 비정답_max: {avg_max_diff:+.4f}")
    if avg_max_diff > 0:
        print(f"   → 정답 최대 유사도가 {avg_max_diff:.4f} 더 높음")
    else:
        print(f"   → 비정답 최대 유사도가 {abs(avg_max_diff):.4f} 더 높음")
    
    correct_higher_max = (max_diffs > 0).sum()
    correct_higher_max_pct = correct_higher_max / num_tables * 100
    print(f"   정답이 더 높은 테이블: {correct_higher_max}/{num_tables} ({correct_higher_max_pct:.1f}%)")
    print()
    
    # 평균 유사도 통계
    # 각 테이블의 정답 LLM query 평균 유사도들의 평균
    print(f"🔹 정답 LLM query 평균 유사도:")
    print(f"   테이블당 평균: {np.mean(correct_avg):.4f} (모든 테이블의 정답 avg 평균)")
    print(f"   (표준편차: {np.std(correct_avg):.4f})")
    print(f"   (최소값: {np.min(correct_avg):.4f}, 최대값: {np.max(correct_avg):.4f})")
    print()
    
    # 각 테이블의 비정답 LLM query 평균 유사도들의 평균
    print(f"🔹 비정답 LLM query 평균 유사도:")
    print(f"   테이블당 평균: {np.mean(incorrect_avg):.4f} (모든 테이블의 비정답 avg 평균)")
    print(f"   (표준편차: {np.std(incorrect_avg):.4f})")
    print(f"   (최소값: {np.min(incorrect_avg):.4f}, 최대값: {np.max(incorrect_avg):.4f})")
    print()
    
    avg_avg_diff = np.mean(avg_diffs)
    print(f"📊 평균 유사도 차이:")
    print(f"   정답_avg - 비정답_avg: {avg_avg_diff:+.4f}")
    if avg_avg_diff > 0:
        print(f"   → 정답 평균 유사도가 {avg_avg_diff:.4f} 더 높음")
    else:
        print(f"   → 비정답 평균 유사도가 {abs(avg_avg_diff):.4f} 더 높음")
    
    correct_higher_avg = (avg_diffs > 0).sum()
    correct_higher_avg_pct = correct_higher_avg / num_tables * 100
    print(f"   정답이 더 높은 테이블: {correct_higher_avg}/{num_tables} ({correct_higher_avg_pct:.1f}%)")
    print()
    
    # 정답보다 유사도가 높은 비정답 쿼리 개수 (테이블당 평균)
    if results["higher_incorrect_counts_max"]:
        higher_counts_max = np.array(results["higher_incorrect_counts_max"])
        higher_counts_avg = np.array(results["higher_incorrect_counts_avg"])
        
        print(f"📊 정답보다 유사도가 높은 비정답 쿼리 개수:")
        print(f"   (정답 max 기준) 테이블당 평균: {np.mean(higher_counts_max):.2f}개")
        print(f"   (표준편차: {np.std(higher_counts_max):.2f})")
        print(f"   (최소값: {np.min(higher_counts_max):.0f}개, 최대값: {np.max(higher_counts_max):.0f}개)")
        print()
        print(f"   (정답 avg 기준) 테이블당 평균: {np.mean(higher_counts_avg):.2f}개")
        print(f"   (표준편차: {np.std(higher_counts_avg):.2f})")
        print(f"   (최소값: {np.min(higher_counts_avg):.0f}개, 최대값: {np.max(higher_counts_avg):.0f}개)")
        print(f"   (총 {num_tables}개 테이블 분석)")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Original query vs LLM query 유사도 분석")
    parser.add_argument("--qa-file", type=str, 
                       default="/home/subeen/DaisLab/SACU/data/SACU/QA_tables/dev_QA_subset.jsonl",
                       help="QA subset 파일 경로 (기본값: dev_QA_subset.jsonl)")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="분석할 샘플 수 (기본값: 전체)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Original query vs LLM query 유사도 분석")
    print("정답 LLM query (같은 테이블) vs 비정답 LLM query (다른 테이블)")
    print("="*60)
    
    # Encoder 초기화
    print("\n🔧 Encoder 초기화 중...", flush=True)
    encoder = SacuTableEncoder(
        use_stella=False,  # E5 모델 사용
        use_e5=True,
        verbose=False,
        enable_query_generation=False,
        enable_column_relevance=False,
        num_rows=150,
        device="cuda"
    )
    
    # 데이터 로딩
    data_dir = Path("/home/subeen/DaisLab/SACU/data/SACU")
    qa_subset_file = Path(args.qa_file)
    original_dev_file = data_dir / "original_tables" / "SACU_dev.jsonl"
    
    print(f"\n📝 QA subset 데이터 로딩 중: {qa_subset_file}", flush=True)
    qa_items = load_qa_subset(str(qa_subset_file))
    print(f"   ✓ {len(qa_items)}개 샘플 로딩 완료", flush=True)
    
    print(f"\n📝 Original dev 데이터 로딩 중...", flush=True)
    orig_items = load_original_dev(str(original_dev_file))
    print(f"   ✓ {len(orig_items)}개 샘플 로딩 완료", flush=True)
    
    # 평가 쿼리는 original dev의 query 사용
    eval_queries = [item["query"] for item in orig_items]
    eval_feta_ids = [item["feta_id"] for item in orig_items]
    
    # 분석 실행
    results = analyze_query_query_similarity(
        eval_queries,
        eval_feta_ids,
        qa_items,
        encoder,
        num_samples=args.num_samples
    )
    
    # 통계 출력
    print_statistics(results)
    
    print("✅ 분석 완료!", flush=True)

