#!/usr/bin/env python3
"""
다양한 서브셋 버전의 retrieval 성능 평가 및 비교 (specific_encoder 사용)

- Baseline, Relevant subset, Best subset, LLM subset, QA subset, QA query subset, LaTeX best subset 평가
- Query-table 유사도 기반 retrieval (일반 subset)
- Query-query 유사도 기반 retrieval (qa_query_subset)
- Recall@K, MRR 등 메트릭 계산
- 결과 비교 및 분석
- specific_encoder.py의 type-specific embedding 및 vertical partitioning 지원

Author: Subset Retrieval Evaluation Team
Version: 1.0
"""
import os
# GPU 지정 (환경변수가 없으면 GPU 1 사용 - 가장 여유로운 GPU)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# SACU model 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "model"))
from retriever import SacuEmbeddingRetriever, SacuTableRetriever
from specific_encoder import SacuTableEncoder as SpecificSacuTableEncoder


def load_sacu_data(jsonl_path: Path, limit: int = None, subset_type: str = None, allowed_feta_ids: set = None) -> Tuple[List[str], List[Dict]]:
    """Load SACU JSONL data.
    
    Args:
        jsonl_path: Path to JSONL file
        limit: Maximum number of entries to load (None = all)
        subset_type: Type of subset ('best_subset', 'relevant_subset', 'llm_subset') 
                     to mark tables with partition_id > 0
        allowed_feta_ids: Set of feta_ids to filter (None = all)
    
    Returns:
        queries: List of questions
        tables: List of table dicts with feta_id, table_id, db_id, table, subset_type, question (if available)
    """
    queries = []
    tables = []
    
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            if limit and len(queries) >= limit:
                break
            
            data = json.loads(line)
            feta_id = data['feta_id']
            
            # Filter by allowed_feta_ids if provided
            if allowed_feta_ids is not None and feta_id not in allowed_feta_ids:
                continue
            
            # Handle different data structures
            # Some files have 'input' key, some don't
            if 'input' in data:
                input_data = data['input']
                table_array = input_data.get('table_array')
                question = input_data.get('question')  # May be None for no_question files
            else:
                # Fallback: try to get table_array directly from data
                table_array = data.get('table_array')
                question = data.get('question')
            
            # Skip if no table_array found
            if table_array is None:
                continue
            
            # Only add question to queries if it exists
            if question:
                queries.append(question)
            
            table_dict = {
                'feta_id': feta_id,
                'table_id': str(feta_id),
                'db_id': 'sacu_db',
                'table': table_array,
            }
            
            # Add question if available
            if question:
                table_dict['question'] = question
            
            # Mark subset tables for partition identification
            if subset_type:
                table_dict['subset_type'] = subset_type
            
            tables.append(table_dict)
    
    return queries, tables


class SubsetAwareTableRetriever(SacuTableRetriever):
    """Custom table retriever that recognizes subset tables from table_id markers."""
    
    def index_tables(
        self,
        dataset_name: str,
        encoder,
        corpus,
        num_rows=None,
    ):
        """Override to detect subset tables from table_id markers."""
        if self.verbose:
            print(f"🔄 Embedding all tables for {dataset_name}…")
            print(f"embedding dim: {encoder.embedding_dim}")
            print("model: stella_en_400M_v5")
            is_partitioning_on = getattr(encoder, 'enable_type_specific_embedding', False)
            print(f"🔗 Type-specific embedding: {'ON' if is_partitioning_on else 'OFF'}")

        embeds, meta = [], []
        import psutil
        mem_before = psutil.virtual_memory().percent if self.verbose else 0

        for b_idx, batch in enumerate(corpus):
            tables = batch["table"]
            db_ids = batch["database_id"]
            tab_ids = batch["table_id"]

            if self.verbose and (b_idx < 5 or b_idx % 10 == 0):
                cur_mem = psutil.virtual_memory().percent
                print(f"📦 Batch {b_idx+1}: {len(tables)} tables (mem {cur_mem:.1f}%)")

            for tbl, db_id, tab_id in zip(tables, db_ids, tab_ids):
                try:
                    # Check if table_id has subset marker
                    original_tab_id = str(tab_id)
                    partition_id = 0  # 기본값: 원본 테이블
                    
                    if isinstance(tab_id, str):
                        # Check for subset markers and extract original table_id
                        # relevant_subset -> PART1, best_subset -> PART2, llm_subset -> PART3, qa_subset -> PART4, qa_query_subset -> PART6, latex_best_subset -> PART5
                        if tab_id.endswith('_latex_best_subset') or '_latex_best_subset' in tab_id:
                            partition_id = 5
                            original_tab_id = tab_id.replace('_latex_best_subset', '')
                        elif tab_id.endswith('_qa_query_subset') or '_qa_query_subset' in tab_id:
                            partition_id = 6
                            original_tab_id = tab_id.replace('_qa_query_subset', '')
                        elif tab_id.endswith('_relevant_subset') or '_relevant_subset' in tab_id:
                            partition_id = 1
                            original_tab_id = tab_id.replace('_relevant_subset', '')
                        elif tab_id.endswith('_best_subset') or '_best_subset' in tab_id:
                            partition_id = 2
                            original_tab_id = tab_id.replace('_best_subset', '')
                        elif tab_id.endswith('_llm_subset') or '_llm_subset' in tab_id:
                            partition_id = 3
                            original_tab_id = tab_id.replace('_llm_subset', '')
                        elif tab_id.endswith('_qa_subset') or '_qa_subset' in tab_id:
                            partition_id = 4
                            original_tab_id = tab_id.replace('_qa_subset', '')
                    
                    emb_result = encoder.encode_table(tbl, table_id=original_tab_id)
                    
                    # specific_encoder는 항상 List[Dict]를 반환 (table embedding만 포함)
                    # query embedding은 encode_query()를 별도로 호출하여 처리
                    if isinstance(emb_result, list):
                        for subtable_info in emb_result:
                            embeds.append(subtable_info['embedding'])
                            
                            # If this is a subset table, use the partition_id we determined
                            # (don't override if encode_table already set a partition_id > 0)
                            subtable_partition_id = subtable_info.get('partition_id', 0)
                            if partition_id > 0 and subtable_partition_id == 0:
                                subtable_partition_id = partition_id
                            
                            meta.append(
                                {
                                    "database_id": str(db_id),
                                    "table_id": original_tab_id,
                                    "partition_id": subtable_partition_id,
                                    "has_query_embedding": False,  # encode_table() 반환값에는 query_embedding 없음 (encode_query()로 별도 처리)
                                }
                            )
                    else:
                        # 단일 임베딩인 경우 (이론상 발생하지 않지만 방어적 코딩)
                        if isinstance(emb_result, np.ndarray):
                            embeds.append(emb_result)
                        elif isinstance(emb_result, dict):
                            embeds.append(emb_result['embedding'])
                        else:
                            # 첫 번째 요소 사용
                            embeds.append(emb_result[0]['embedding'] if isinstance(emb_result, list) else emb_result)
                        meta.append(
                            {
                                "database_id": str(db_id),
                                "table_id": original_tab_id,
                                "partition_id": partition_id,
                                "has_query_embedding": False,  # encode_table() 반환값에는 query_embedding 없음 (encode_query()로 별도 처리)
                            }
                        )
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  embedding fail ({tab_id}): {e}")
                    continue

        if self.verbose:
            mem_after = psutil.virtual_memory().percent
            print(f"💾 Memory: {mem_before:.1f}% → {mem_after:.1f}%")

        self.embeddings_cache[dataset_name] = {
            "embeddings": np.array(embeds),
            "metadata": meta,
        }


class SubsetRetrievalEvaluator:
    """Evaluator for subset table retrieval."""
    
    def __init__(self, retriever: SacuEmbeddingRetriever, top_k: int = 10):
        self.retriever = retriever
        self.top_k = top_k
        # Replace the internal table_retriever with our custom one
        # Get verbose from the internal retriever
        verbose = retriever.retriever.verbose if hasattr(retriever.retriever, 'verbose') else False
        self.verbose = verbose  # Store verbose for use in _retrieve_by_query_similarity
        self.retriever.retriever = SubsetAwareTableRetriever(verbose=verbose)
        
        # Metrics tracking
        self.num_overlap = 0
        self.total_tables = 0
        self.total_tables_capped = 0
        self.total_queries_processed = 0
    
    def evaluate(self, queries: List[str], gold_tables: List[Dict],
                 corpus_tables: List[Dict], dataset_name: str, subset_type: str = None) -> Dict:
        """Evaluate retrieval performance.
        
        Args:
            queries: List of questions
            gold_tables: Gold standard tables with table_id
            corpus_tables: All tables in the corpus (should have 'question' field for qa_query_subset)
            dataset_name: Name for logging
            subset_type: Type of subset being evaluated (e.g., 'qa_query_subset')
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*70}")
        print(f"📊 {dataset_name} 평가 시작")
        print(f"{'='*70}")
        print(f"총 쿼리 수: {len(queries)}")
        print(f"코퍼스 크기: {len(corpus_tables)}개 테이블")
        print(f"Top-K: {self.top_k}")
        
        # Check if this is qa_query_subset mode
        use_query_similarity = (subset_type == 'qa_query_subset')
        
        if use_query_similarity:
            print(f"🔍 검색 방식: Query-Query 유사도 기반 (qa_query_subset mode)")
        else:
            print(f"🔍 검색 방식: Query-Table 유사도 기반")
        
        # Retrieve and evaluate
        print(f"\n🔍 검색 및 평가 중...")
        
        # Reset metrics
        self.num_overlap = 0
        self.total_tables = 0
        self.total_tables_capped = 0
        self.total_queries_processed = 0
        
        start_time = time.time()
        
        # Process queries in batches (for compatibility with update_retrieval_metrics format)
        query_batch = {
            'queries': queries,
            'gold_tables': gold_tables,
        }
        
        # Retrieve results for all queries
        retrieval_results = []
        hits = []
        
        if use_query_similarity:
            # Query-query similarity based retrieval
            retrieval_results = self._retrieve_by_query_similarity(queries, corpus_tables, dataset_name)
        else:
            # Query-table similarity based retrieval (original method)
            # Index corpus
            print(f"\n🔄 테이블 인덱싱 중...")
            corpus = self._prepare_corpus(corpus_tables)
            self.retriever.embed_corpus(dataset_name, corpus)
            
            # Retrieve results for all queries
            for i, query in enumerate(queries):
                retrieved = self.retriever.retrieve(query, dataset_name, self.top_k)
                retrieval_results.append(retrieved)
            
            # Track hits for MRR calculation
        # Helper function to remove subset markers (same as in _update_retrieval_metrics)
        def remove_subset_marker(table_id_str):
            """Remove subset_type markers from table_id for matching with gold tables."""
            markers = ['_qa_query_subset', '_qa_subset', '_best_subset', '_relevant_subset', 
                      '_llm_subset', '_latex_best_subset']
            for marker in markers:
                if table_id_str.endswith(marker) or marker in table_id_str:
                    return table_id_str.replace(marker, '')
            return table_id_str
        
        for i, (query, gold_info) in enumerate(zip(queries, gold_tables)):
            gold_table_id = gold_info['table_id']
            gold_db_id = gold_info['db_id']
            retrieved = retrieval_results[i]
            hit = False
            for rank, (db_id, table_id) in enumerate(retrieved):
                # Remove subset markers for comparison
                cleaned_table_id = remove_subset_marker(str(table_id))
                # Handle both single table_id and list of table_ids
                if not isinstance(gold_table_id, list):
                    gold_table_ids = [str(gold_table_id)]
                else:
                    gold_table_ids = [str(t) for t in gold_table_id]
                
                if str(db_id) == str(gold_db_id) and cleaned_table_id in gold_table_ids:
                    hit = True
                    hits.append(rank + 1)
                    break
            if not hit:
                hits.append(None)
            
            # Progress
            if (i + 1) % 50 == 0 or (i + 1) == len(queries):
                # Calculate current recall using new method
                current_recall = self.num_overlap / self.total_tables if self.total_tables > 0 else 0.0
                print(f"  진행률: {i+1}/{len(queries)} (Recall@{self.top_k}: {current_recall*100:.2f}%)")
        
        # Update metrics using the new format
        self._update_retrieval_metrics(query_batch, retrieval_results)
        
        elapsed = time.time() - start_time
        
        # Compute statistics using new method
        performance = self._calculate_table_retrieval_performance(
            self.top_k,
            total_retrieval_duration_process=elapsed,
            total_retrieval_duration_wall_clock=elapsed,
            num_queries_retrieved=len(queries)
        )
        mrr = np.mean([1.0/rank if rank else 0.0 for rank in hits])
        
        # Get semantic space preservation statistics
        encoder = self.retriever.encoder
        semantic_stats = None
        if hasattr(encoder, 'get_semantic_space_stats'):
            semantic_stats = encoder.get_semantic_space_stats()
        
        stats = {
            'dataset': dataset_name,
            'total_queries': len(queries),
            'top_k': self.top_k,
            f'recall@{self.top_k}': performance['recall'],
            'accuracy': performance['accuracy'],
            'capped_recall': performance['capped_recall'],
            'mrr': mrr,
            'total_time': elapsed,
            'avg_time_per_query': elapsed / len(queries),
            'semantic_stats': semantic_stats,
        }
        
        self._print_results(stats)
        
        return stats
    
    def _retrieve_by_query_similarity(self, queries: List[str], corpus_tables: List[Dict], dataset_name: str) -> List[List[Tuple]]:
        """Retrieve tables: qa_query_subset uses query-query similarity, others use query-table similarity.
        
        Args:
            queries: List of evaluation queries
            corpus_tables: List of corpus tables (qa_query_subset has 'question' field, others use table)
            dataset_name: Name for logging
        
        Returns:
            List of retrieval results, each is List[Tuple[db_id, table_id]]
        """
        # Separate qa_query_subset tables from others
        qa_query_tables = []
        other_tables = []
        
        for table in corpus_tables:
            if table.get('subset_type') == 'qa_query_subset' and 'question' in table:
                qa_query_tables.append(table)
            else:
                other_tables.append(table)
        
        print(f"\n🔄 코퍼스 분리:")
        print(f"   - qa_query_subset 테이블: {len(qa_query_tables)}개 (query-query similarity)")
        print(f"   - 기타 테이블: {len(other_tables)}개 (query-table similarity)")
        
        encoder = self.retriever.encoder
        
        # 1. Embed evaluation queries (두 가지 버전 생성)
        # - 하이브리드 쿼리: qa_query_subset의 query-query similarity용 (augmentation 적용)
        # - 하이브리드 쿼리: other_tables의 query-table similarity용
        print(f"\n🔄 평가 쿼리 임베딩 생성 중...")
        print(f"   - 쿼리 (qa_query_subset용): {len(queries)}개")
        query_embeddings_for_qa = []  # 쿼리 (qa_query_subset용)
        for query in queries:
            query_emb_qa = encoder.encode_query(query, use_hybrid=False)  # query-query matching용 (text mode)
            query_embeddings_for_qa.append(query_emb_qa)
        
        print(f"   - 쿼리 (other_tables용): {len(queries)}개")
        query_embeddings_for_table = []  # 쿼리 (other_tables용)
        for query in queries:
            query_emb_table = encoder.encode_query(query, use_hybrid=False)  # query-table matching용 (text mode)
            query_embeddings_for_table.append(query_emb_table)
        query_embeddings_for_qa = np.array(query_embeddings_for_qa)
        query_embeddings_for_table = np.array(query_embeddings_for_table)
        
        all_table_info = []
        all_similarities_list = []  # List of similarity arrays for each query
        qa_table_info = []  # Initialize outside if block to use in statistics
        
        # 2. Process qa_query_subset tables with query-query similarity
        if qa_query_tables:
            print(f"\n🔄 qa_query_subset 테이블: Query 임베딩 생성 중...")
            qa_questions = []
            for table in qa_query_tables:
                qa_questions.append(table['question'])
                table_id = table['table_id']
                subset_type = table.get('subset_type', None)
                if subset_type:
                    table_id = f"{table_id}_{subset_type}"
                qa_table_info.append((table['db_id'], table_id))
            
            # qa_query_subset의 경우 query-query similarity 사용
            qa_query_embeddings = []
            for question in qa_questions:
                qa_emb = encoder.encode_query(question, use_hybrid=False)  # query-query matching용 (text mode)
                qa_query_embeddings.append(qa_emb)
            qa_query_embeddings = np.array(qa_query_embeddings)
            
            # Compute query-query similarities for all queries at once
            # Shape: (queries, qa_tables)
            qa_similarities = np.dot(query_embeddings_for_qa, qa_query_embeddings.T)
            
            all_table_info.extend(qa_table_info)
        else:
            qa_similarities = np.zeros((len(queries), 0))
        
        # 3. Process other tables with query-table similarity
        if other_tables:
            print(f"\n🔄 기타 테이블: Table 임베딩 생성 중...")
            other_table_info = []
            other_embeddings = []
            
            for table in other_tables:
                table_id = table['table_id']
                subset_type = table.get('subset_type', None)
                if subset_type:
                    table_id = f"{table_id}_{subset_type}"
                other_table_info.append((table['db_id'], table_id))
                
                # Get table embedding using encoder
                emb_result = encoder.encode_table(table['table'], table_id=str(table_id))
                
                # specific_encoder는 항상 List[Dict]를 반환
                if isinstance(emb_result, list):
                    # 첫 번째 요소의 embedding 사용
                    table_emb = emb_result[0]['embedding']
                elif isinstance(emb_result, np.ndarray):
                    # Direct numpy array (이론상 발생하지 않지만 방어적 코딩)
                    table_emb = emb_result
                elif isinstance(emb_result, dict):
                    # Dict format (이론상 발생하지 않지만 방어적 코딩)
                    table_emb = emb_result['embedding']
                else:
                    # Fallback
                    table_emb = emb_result[0]['embedding'] if isinstance(emb_result, list) else emb_result
                
                other_embeddings.append(table_emb)
            
            if other_embeddings:
                other_embeddings = np.array(other_embeddings)
                # Embeddings are already normalized from encode_table, so no need to normalize again
                # (Double normalization is safe but unnecessary)
                
                # Compute query-table similarities for all queries at once
                # Shape: (queries, other_tables)
                # other_tables는 type-specific embedding을 사용하므로 하이브리드 쿼리 사용
                other_similarities = np.dot(query_embeddings_for_table, other_embeddings.T)
            else:
                other_similarities = np.zeros((len(queries), 0))
            
            all_table_info.extend(other_table_info)
        else:
            other_similarities = np.zeros((len(queries), 0))
        
        # 4. Combine similarities and select top-K for each query
        print(f"\n🔍 유사도 통합 및 Top-{self.top_k} 선택 중...")
        
        # Concatenate similarities: (queries, qa_tables + other_tables)
        all_similarities = np.concatenate([qa_similarities, other_similarities], axis=1)
        
        # Track how many qa_query_subset tables are in top-K
        num_qa_in_topk = 0
        num_other_in_topk = 0
        
        retrieval_results = []
        verbose = self.verbose  # Use stored verbose from __init__
        
        for i in range(len(queries)):
            # Get top-k indices for this query
            top_k_indices = np.argsort(all_similarities[i])[::-1][:self.top_k]
            
            # Get corresponding table info
            retrieved = [all_table_info[idx] for idx in top_k_indices]
            retrieval_results.append(retrieved)
            
            # Track partition stats for this query
            partition_stats = {"original": 0, "partitioned": 0}
            
            # Log results in the same format as SacuTableRetriever.search() for extract_partition_stats.py
            if verbose:
                print(f"🔍 Search {dataset_name} | '{queries[i][:50]}…'")
            
            # Count qa_query_subset vs other tables in top-K and log partition info
            for rank, idx in enumerate(top_k_indices):
                similarity = all_similarities[i][idx]
                db_id, table_id = all_table_info[idx]
                
                # Determine partition type
                if idx < len(qa_table_info):
                    # This is a qa_query_subset table (PART6)
                    partition_type = "PART6"
                    partition_stats["partitioned"] += 1
                    num_qa_in_topk += 1
                else:
                    # This is an other table - determine partition_id from table_id
                    partition_type = "ORIG"  # Default
                    if isinstance(table_id, str):
                        if '_latex_best_subset' in table_id:
                            partition_type = "PART5"
                            partition_stats["partitioned"] += 1
                        elif '_qa_subset' in table_id:
                            partition_type = "PART4"
                            partition_stats["partitioned"] += 1
                        elif '_llm_subset' in table_id:
                            partition_type = "PART3"
                            partition_stats["partitioned"] += 1
                        elif '_best_subset' in table_id:
                            partition_type = "PART2"
                            partition_stats["partitioned"] += 1
                        elif '_relevant_subset' in table_id:
                            partition_type = "PART1"
                            partition_stats["partitioned"] += 1
                        else:
                            partition_stats["original"] += 1
                    else:
                        partition_stats["original"] += 1
                    num_other_in_topk += 1
                
                # Log in the same format as SacuTableRetriever.search()
                if verbose:
                    print(f"   #{rank+1:>2}  sim={similarity:.4f}  {table_id} ({partition_type})")
            
            # Log partition stats for this query (same format as SacuTableRetriever.search())
            if verbose:
                total = partition_stats["original"] + partition_stats["partitioned"]
                print(f"   📊 Partition stats: {partition_stats['original']}/{total} original, {partition_stats['partitioned']}/{total} partitioned")
        
        # Print statistics
        total_retrieved = len(queries) * self.top_k
        print(f"   📊 검색 결과 통계:")
        print(f"      - qa_query_subset 테이블이 Top-{self.top_k}에 포함된 횟수: {num_qa_in_topk}/{total_retrieved} ({num_qa_in_topk/total_retrieved*100:.2f}%)")
        print(f"      - 기타 테이블이 Top-{self.top_k}에 포함된 횟수: {num_other_in_topk}/{total_retrieved} ({num_other_in_topk/total_retrieved*100:.2f}%)")
        
        return retrieval_results
    
    def _update_retrieval_metrics(self, query_batch, new_retrieval_results):
        """Update retrieval metrics based on query batch and retrieval results.
        
        Args:
            query_batch: Dict with 'queries' and 'gold_tables' lists
            new_retrieval_results: List of retrieval results (each is List[Tuple[db_id, table_id]])
        """
        queries = query_batch['queries']
        gold_tables = query_batch['gold_tables']
        
        for idx in range(len(new_retrieval_results)):
            gold_info = gold_tables[idx]
            gold_db_id = str(gold_info['db_id'])
            gold_table_id = gold_info['table_id']
            retrieval_result = new_retrieval_results[idx]
            
            # Normalize gold_table_id to list (treat all datasets as a list)
            if not isinstance(gold_table_id, list):
                gold_table_id = [gold_table_id]
            
            # Create normalized gold tables set
            # E.g. {('soccer_3', 'club'), ('soccer_3', 'players')}
            normalized_gold_tables = set((gold_db_id, str(t)) for t in gold_table_id)
            
            # Convert retrieval results to set of tuples, removing subset_type markers from table_id
            # Remove subset_type markers (e.g., "_qa_query_subset", "_best_subset") for matching
            def remove_subset_marker(table_id_str):
                """Remove subset_type markers from table_id for matching with gold tables."""
                markers = ['_qa_query_subset', '_qa_subset', '_best_subset', '_relevant_subset', 
                          '_llm_subset', '_latex_best_subset']
                for marker in markers:
                    if table_id_str.endswith(marker) or marker in table_id_str:
                        return table_id_str.replace(marker, '')
                return table_id_str
            
            normalized_retrieval_set = set((db_id, remove_subset_marker(str(table_id))) 
                                         for db_id, table_id in retrieval_result)
            
            # Calculate intersection
            self.num_overlap += len(normalized_gold_tables.intersection(normalized_retrieval_set))
            self.total_tables += len(normalized_gold_tables)
            
            # Cap denominator at len(retrieval_result) (aka `k`)
            self.total_tables_capped += min(len(normalized_gold_tables), len(retrieval_result))
            
            self.total_queries_processed += 1
    
    def _calculate_table_retrieval_performance(self, top_k, total_retrieval_duration_process=0.0, 
                                               total_retrieval_duration_wall_clock=0.0, 
                                               num_queries_retrieved=0):
        """Calculate table retrieval performance metrics.
        
        Args:
            top_k: Top-K value for retrieval
            total_retrieval_duration_process: Total process time for retrieval
            total_retrieval_duration_wall_clock: Total wall clock time for retrieval
            num_queries_retrieved: Number of queries retrieved
            
        Returns:
            Dict with accuracy, recall, capped_recall, and timing metrics
        """
        if self.total_queries_processed == 0:
            raise ValueError("haven't processed any queries!")
        
        # Calculate retrieval performance
        accuracy = self.num_overlap / self.total_tables if self.total_tables > 0 else 0.0
        recall = self.num_overlap / self.total_tables if self.total_tables > 0 else 0.0
        capped_recall = self.num_overlap / self.total_tables_capped if self.total_tables_capped > 0 else 0.0
        
        # Calculate average durations
        avg_dur_process = 0.0
        avg_dur_wall_clock = 0.0
        if num_queries_retrieved > 0:
            avg_dur_process = round(total_retrieval_duration_process / num_queries_retrieved, 5)
            avg_dur_wall_clock = round(total_retrieval_duration_wall_clock / num_queries_retrieved, 5)
        
        result = {
            'k': top_k,
            'accuracy': accuracy,
            'recall': recall,
            'capped_recall': capped_recall,
            'retrieval_duration_process': round(total_retrieval_duration_process, 5),
            'avg_retrieval_duration_process': avg_dur_process,
            'retrieval_duration_wall_clock': round(total_retrieval_duration_wall_clock, 5),
            'avg_retrieval_duration_wall_clock': avg_dur_wall_clock,
        }
        
        # Reset metrics for next dataset
        self.total_queries_processed = 0
        self.num_overlap = 0
        self.total_tables = 0
        self.total_tables_capped = 0
        
        return result
    
    def _prepare_corpus(self, tables: List[Dict]):
        """Prepare corpus in batch format."""
        def corpus_generator():
            # subset_type이 있으면 table_id에 마커 추가하여 retriever에서 인식 가능하도록
            table_ids = []
            for t in tables:
                table_id = t['table_id']
                subset_type = t.get('subset_type', None)
                if subset_type:
                    # subset 테이블은 table_id에 subset_type 마커 추가
                    # retriever에서 이를 인식하여 partition_id 설정
                    table_id = f"{table_id}_{subset_type}"
                table_ids.append(table_id)
            
            batch = {
                'table': [t['table'] for t in tables],
                'database_id': [t['db_id'] for t in tables],
                'table_id': table_ids,
            }
            yield batch
        
        return corpus_generator()
    
    def _print_results(self, stats: Dict):
        """Print evaluation results."""
        print(f"\n{'='*70}")
        print(f"🎯 {stats['dataset']} 평가 결과")
        print(f"{'='*70}")
        top_k = stats['top_k']
        recall_key = f'recall@{top_k}'
        print(f"✅ Recall@{top_k}: {stats[recall_key]*100:.2f}%")
        if 'accuracy' in stats:
            print(f"✅ Accuracy: {stats['accuracy']*100:.2f}%")
        if 'capped_recall' in stats:
            print(f"✅ Capped Recall: {stats['capped_recall']*100:.2f}%")
        print(f"✅ MRR: {stats['mrr']:.4f}")
        print(f"⏱️  Total Time: {stats['total_time']:.2f}s")
        print(f"⏱️  Avg Time/Query: {stats['avg_time_per_query']*1000:.2f}ms")
        
        # Print semantic space preservation statistics
        if stats.get('semantic_stats'):
            sem_stats = stats['semantic_stats']
            print(f"\n📊 Semantic Space Preservation Statistics:")
            print(f"   - Total Hybrid Embeddings: {sem_stats['total_hybrid_embeddings']}")
            print(f"   - Preservation Failures: {sem_stats['preservation_failures']}")
            print(f"   - Preservation Rate: {sem_stats['preservation_rate']:.2f}%")
        
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval with different dev subset versions in corpus (using specific_encoder)"
    )
    parser.add_argument(
        '--subset-dir',
        type=Path,
        required=True,
        help='Directory containing dev subset JSONL files'
    )
    parser.add_argument(
        '--sacu-dir',
        type=Path,
        default=Path('/home/subeen/DaisLab/SACU/data/SACU'),
        help='Directory containing original SACU JSONL files (train/dev/test)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['table', 'query'],
        default='table',
        help='Embedding mode for retrieval (table: table-based, query: query-based)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Top-K for evaluation'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output JSON file for results comparison'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of entries to evaluate (for testing)'
    )
    parser.add_argument(
        '--eval-dataset',
        type=str,
        choices=['dev', 'test', 'train'],
        default='dev',
        help='Dataset to evaluate (dev, test, or train)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"🚀 서브셋 코퍼스 Retrieval 성능 비교")
    print(f"{'='*70}")
    print(f"Subset Directory: {args.subset_dir}")
    print(f"SACU Directory: {args.sacu_dir}")
    print(f"Embedding Mode: {args.mode}")
    print(f"Top-K: {args.top_k}")
    print(f"Eval Dataset: {args.eval_dataset}")
    print(f"\n💡 실험 설계:")
    if args.eval_dataset == 'dev':
        print(f"   - Query: dev 데이터셋의 질문 (동일)")
        print(f"   - Corpus 1 (baseline): train + dev(original) + dev(original) + test")
        print(f"   - Corpus 2: train + dev(original) + dev(relevant_subset) + test")
        print(f"   - Corpus 3: train + dev(original) + dev(best_subset) + test")
        print(f"   - Corpus 4: train + dev(original) + dev(llm_subset) + test")
        print(f"   - Corpus 5: train + dev(original) + dev(qa_subset) + test")
        print(f"   - Corpus 6: train + dev(original) + dev(qa_query_subset) + test (query-query similarity)")
        print(f"   - Corpus 7: train + dev(original) + dev(latex_best_subset) + test")
    elif args.eval_dataset == 'test':
        print(f"   - Query: test 데이터셋의 질문 (동일)")
        print(f"   - Corpus 1 (baseline): train + dev(original) + test(original) + test(original)")
        print(f"   - Corpus 2: train + dev(original) + test(original) + test(relevant_subset)")
        print(f"   - Corpus 3: train + dev(original) + test(original) + test(best_subset)")
        print(f"   - Corpus 4: train + dev(original) + test(original) + test(llm_subset)")
        print(f"   - Corpus 5: train + dev(original) + test(original) + test(qa_subset)")
        print(f"   - Corpus 6: train + dev(original) + test(original) + test(qa_query_subset) (query-query similarity)")
        print(f"   - Corpus 7: train + dev(original) + test(original) + test(latex_best_subset)")
    else:  # train
        print(f"   - Query: train 데이터셋의 질문 (동일)")
        print(f"   - Corpus 1 (baseline): train(original) + dev(original) + test(original)")
        print(f"   - Corpus 2: train(relevant_subset) + dev(original) + dev(relevant_subset) + test(relevant_subset)")
        print(f"   - Corpus 3: train(best_subset) + dev(original) + dev(best_subset) + test(best_subset)")
        print(f"   - Corpus 4: train(llm_subset) + dev(original) + dev(llm_subset) + test(llm_subset)")
        print(f"   - Corpus 5: train(qa_subset) + dev(original) + dev(qa_subset) + test(qa_subset)")
        print(f"   - Corpus 6: train(qa_subset) + dev(original) + dev(qa_subset) + test(qa_subset) (query-query similarity)")
        print(f"   - 목표: subset 테이블을 추가했을 때 retrieval 성능 변화 측정")
    
    # GPU info
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        print(f"\n🖥️  GPU 정보:")
        print(f"   디바이스: {torch.cuda.get_device_name(device_id)}")
        print(f"   현재 사용: {torch.cuda.memory_allocated(device_id) / 1024**3:.2f} GB")
    
    # Initialize retriever
    print(f"\n🔧 Retriever 초기화 중...")
    retriever = SacuEmbeddingRetriever(
        verbose=True,
        enable_query_generation=False,
        enable_column_relevance=False,
        num_rows=None,  # 제한 없음: 모든 행 사용
        num_questions=5,
        embedding_mode=args.mode,
        use_stella=True,
    )
    
    # Replace encoder with specific_encoder
    # Table: type-specific embedding 사용 (enable_type_specific_embedding=True)
    # Query: 기본 임베딩만 사용 (use_hybrid=False)
    retriever.encoder = SpecificSacuTableEncoder(
        use_stella=True,
        verbose=True,
        enable_type_specific_embedding=True,  # Table은 type-specific embedding 사용
        num_rows=None,  # 제한 없음: 모든 행 사용
    )
    # Note: specific_encoder는 encoder.py와 달리 embedding_mode 속성이 없음
    # encode_table()은 type-specific embedding 사용
    # encode_query()는 use_hybrid=False로 호출하여 기본 임베딩만 사용
    
    evaluator = SubsetRetrievalEvaluator(retriever, top_k=args.top_k)
    
    # Load train and always-included original data
    print(f"\n{'='*70}")
    print(f"📂 Loading train and always-included original data...")
    print(f"{'='*70}")
    
    # Baseline uses original data, other subsets use QA_subset
    train_path_original = args.sacu_dir / "original_tables" / "SACU_train.jsonl"
    train_path_qa = args.sacu_dir / "subset_tables" / "QA_tables" / "train_QA_subset.jsonl"
    dev_path = args.sacu_dir / "original_tables" / "SACU_dev.jsonl"
    test_path_original = args.sacu_dir / "original_tables" / "SACU_test.jsonl"
    test_path_qa = args.sacu_dir / "subset_tables" / "QA_tables" / "test_QA_subset.jsonl"
    
    # No feta_id filtering - use all data
    allowed_feta_ids = None
    
    # QA_tables directory for qa_subset loading (but no filtering)
    qa_dir = args.sacu_dir / "subset_tables" / "QA_tables"
    
    # Load original data for baseline (train, dev, test)
    _, train_tables_original = load_sacu_data(train_path_original, allowed_feta_ids=None)
    print(f"   Loaded {len(train_tables_original)} tables from original train")
    
    # Dev is not filtered by feta_ids - include all data
    _, orig_dev_tables = load_sacu_data(dev_path, limit=None, subset_type=None, allowed_feta_ids=None)
    print(f"   Loaded {len(orig_dev_tables)} original dev tables (always included)")
    
    # Test original data (for baseline)
    _, orig_test_tables = load_sacu_data(test_path_original, limit=None, subset_type=None, allowed_feta_ids=None)
    print(f"   Loaded {len(orig_test_tables)} original test tables (always included)")
    
    # Load eval queries (from eval_dataset)
    if args.eval_dataset == 'dev':
        eval_path = dev_path
        eval_name = "dev"
    elif args.eval_dataset == 'test':
        eval_path = test_path_original  # Use original test for eval queries
        eval_name = "test"
    else:  # train
        eval_path = train_path_original  # Use original train for eval queries
        eval_name = "train"
    
    if not eval_path.exists():
        print(f"❌ Error: {eval_path} not found")
        sys.exit(1)
    
    eval_queries, eval_gold_tables = load_sacu_data(eval_path, limit=args.limit, allowed_feta_ids=None)
    print(f"\n   Loaded {len(eval_queries)} eval queries from {eval_name}" + (f" (limited to {args.limit})" if args.limit else ""))
    
    # Run evaluation for each subset type
    subset_types = ['baseline', 'relevant_subset', 'best_subset', 'llm_subset', 'qa_subset', 'qa_query_subset']
    all_results = {}
    eval_dataset = args.eval_dataset
    
    for subset_type in subset_types:
        print(f"\n{'='*70}")
        if subset_type == 'baseline':
            if eval_dataset == 'dev':
                print(f"📂 Building corpus (baseline: original dev only)...")
            elif eval_dataset == 'test':
                print(f"📂 Building corpus (baseline: original test only)...")
            else:  # train
                print(f"📂 Building corpus (baseline: original train only)...")
        else:
            if eval_dataset == 'dev':
                print(f"📂 Building corpus (original dev + dev_{subset_type})...")
            elif eval_dataset == 'test':
                print(f"📂 Building corpus (original test + test_{subset_type})...")
            else:  # train
                print(f"📂 Building corpus (original train + train_{subset_type})...")
        print(f"{'='*70}")
        
        # Build corpus
        if subset_type == 'baseline':
            if eval_dataset == 'dev':
                # Baseline: train(original) + dev(original) + test(original)
                corpus_tables = train_tables_original + orig_dev_tables + orig_test_tables
                print(f"   Total corpus: {len(corpus_tables)} tables")
                print(f"   - train (original): {len(train_tables_original)}")
                print(f"   - dev (original): {len(orig_dev_tables)}")
                print(f"   - test (original): {len(orig_test_tables)}")
            elif eval_dataset == 'test':
                # Baseline: train(original) + dev(original) + test(original)
                corpus_tables = train_tables_original + orig_dev_tables + orig_test_tables
                print(f"   Total corpus: {len(corpus_tables)} tables")
                print(f"   - train (original): {len(train_tables_original)}")
                print(f"   - dev (original): {len(orig_dev_tables)}")
                print(f"   - test (original): {len(orig_test_tables)}")
            else:  # train
                # Baseline: train(original) + dev(original) + test(original) (train 포함)
                corpus_tables = train_tables_original + orig_dev_tables + orig_test_tables
                print(f"   Total corpus: {len(corpus_tables)} tables")
                print(f"   - train (original): {len(train_tables_original)}")
                print(f"   - dev (original): {len(orig_dev_tables)}")
                print(f"   - test (original): {len(orig_test_tables)}")
        else:
            # Load subset versions for all splits (train, dev, test)
            # Determine subset paths based on subset_type
            if subset_type == 'qa_subset':
                # QA subset: train, dev, test all use QA_subset
                train_subset_path = args.sacu_dir / "subset_tables" / "QA_tables" / "train_QA_subset.jsonl"
                dev_subset_path = args.sacu_dir / "subset_tables" / "QA_tables" / "dev_QA_subset.jsonl"
                test_subset_path = args.sacu_dir / "subset_tables" / "QA_tables" / "test_QA_subset.jsonl"
            elif subset_type == 'qa_query_subset':
                # QA query subset: train, dev, test all use QA_subset
                train_subset_path = args.sacu_dir / "subset_tables" / "QA_tables" / "train_QA_subset.jsonl"
                dev_subset_path = args.sacu_dir / "subset_tables" / "QA_tables" / "dev_QA_subset.jsonl"
                test_subset_path = args.sacu_dir / "subset_tables" / "QA_tables" / "test_QA_subset.jsonl"
            elif subset_type == 'best_subset':
                # Best subset: train and test use best_subset, dev uses best_subset
                train_subset_path = args.sacu_dir / "subset_tables" / "best_tables" / "train_best_subset.jsonl"
                dev_subset_path = args.sacu_dir / "subset_tables" / "best_tables" / "dev_best_subset.jsonl"
                test_subset_path = args.sacu_dir / "subset_tables" / "best_tables" / "test_best_subset.jsonl"
            elif subset_type == 'relevant_subset':
                # Relevant subset: train and test use relevant_subset, dev uses relevant_subset
                train_subset_path = args.sacu_dir / "subset_tables" / "relevant_tables" / "train_relevant_subset.jsonl"
                dev_subset_path = args.sacu_dir / "subset_tables" / "relevant_tables" / "dev_relevant_subset.jsonl"
                test_subset_path = args.sacu_dir / "subset_tables" / "relevant_tables" / "test_relevant_subset.jsonl"
            elif subset_type == 'llm_subset':
                # LLM subset: train and test use subset
                train_subset_path = args.sacu_dir / "subset_tables" / "llm_tables" / "train_llm_subset.jsonl"
                dev_subset_path = args.sacu_dir / "subset_tables" / "llm_tables" / "dev_llm_subset.jsonl"
                test_subset_path = args.sacu_dir / "subset_tables" / "llm_tables" / "test_llm_subset.jsonl"
            else:
                print(f"⚠️  Warning: Unknown subset_type {subset_type}, skipping")
                continue
            
            # Load subset tables
            train_subset_tables = []
            dev_subset_tables = []
            test_subset_tables = []
            
            # Train uses subset if provided, otherwise original
            if train_subset_path and train_subset_path.exists():
                _, train_subset_tables = load_sacu_data(train_subset_path, limit=args.limit, subset_type=subset_type, allowed_feta_ids=None)
                print(f"   Loaded {len(train_subset_tables)} tables from {train_subset_path.name}" + (f" (limited to {args.limit})" if args.limit else ""))
            else:
                print(f"   Using original train tables: {len(train_tables)}")
            
            # Dev uses subset
            if dev_subset_path and dev_subset_path.exists():
                _, dev_subset_tables = load_sacu_data(dev_subset_path, limit=args.limit, subset_type=subset_type, allowed_feta_ids=None)
                print(f"   Loaded {len(dev_subset_tables)} tables from {dev_subset_path.name}" + (f" (limited to {args.limit})" if args.limit else ""))
            else:
                print(f"   ⚠️  Warning: dev subset not found, using original dev only")
                dev_subset_tables = []
            
            # Test uses subset if provided, otherwise original
            if test_subset_path and test_subset_path.exists():
                _, test_subset_tables = load_sacu_data(test_subset_path, limit=args.limit, subset_type=subset_type, allowed_feta_ids=None)
                print(f"   Loaded {len(test_subset_tables)} tables from {test_subset_path.name}" + (f" (limited to {args.limit})" if args.limit else ""))
            else:
                print(f"   Using original test tables: {len(orig_test_tables)}")
            
            # Build corpus: use subset if available, otherwise original
            # Train: include both original and subset (like dev)
            final_train_tables = train_tables_original  # Always include original
            if train_subset_tables:
                final_train_tables = train_tables_original + train_subset_tables  # Add subset if available
            
            # Test: include both original and subset (like dev and train)
            final_test_tables = orig_test_tables  # Always include original
            if test_subset_tables:
                final_test_tables = orig_test_tables + test_subset_tables  # Add subset if available
            
            corpus_tables = final_train_tables + orig_dev_tables + dev_subset_tables + final_test_tables
            print(f"   Total corpus: {len(corpus_tables)} tables")
            if train_subset_tables:
                print(f"   - train (original): {len(train_tables_original)}")
                print(f"   - train ({subset_type}): {len(train_subset_tables)}")
            else:
                print(f"   - train (original): {len(final_train_tables)}")
            print(f"   - dev (original): {len(orig_dev_tables)}")
            print(f"   - dev ({subset_type}): {len(dev_subset_tables)}")
            if test_subset_tables:
                print(f"   - test (original): {len(orig_test_tables)}")
                print(f"   - test ({subset_type}): {len(test_subset_tables)}")
            else:
                print(f"   - test (original): {len(final_test_tables)}")
        
        # Evaluate
        dataset_name = f"SACU_{eval_dataset}_{subset_type}_{args.mode}"
        results = evaluator.evaluate(eval_queries, eval_gold_tables, corpus_tables, dataset_name, subset_type=subset_type)
        all_results[subset_type] = results
    
    # Print comparison
    eval_dataset_label = eval_dataset.upper()
    print(f"\n{'='*70}")
    print(f"📊 종합 비교 결과 ({eval_dataset_label} 평가)")
    print(f"{'='*70}")
    print(f"{eval_dataset_label + ' Version':<20} {'Recall@10':<15} {'MRR':<10} {'Avg Time (ms)':<15}")
    print(f"{'-'*70}")
    
    for subset_type in subset_types:
        if subset_type in all_results:
            res = all_results[subset_type]
            recall_key = f'recall@{args.top_k}'
            recall = res[recall_key] * 100
            mrr = res['mrr']
            avg_time = res['avg_time_per_query'] * 1000
            print(f"{subset_type:<20} {recall:<15.2f} {mrr:<10.4f} {avg_time:<15.2f}")
    
    print(f"{'='*70}\n")
    
    # Save results
    if args.output:
        with args.output.open('w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"✅ Results saved to {args.output}")
    
    print("\n✅ 평가 완료!")


if __name__ == '__main__':
    main()

