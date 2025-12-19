#!/usr/bin/env python3
"""
컬럼 부분집합 유사도 분석 로그나 LLM 결과로부터 서브셋 테이블 JSONL 파일을 생성
- LLM 컬럼 추출 결과 JSON 파싱
- 원본 테이블에서 지정된 컬럼만 추출하여 서브셋 테이블 생성
- SACU JSONL 포맷으로 출력
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Regex patterns for parsing log
TABLE_PATTERN = re.compile(r"^📄 Table (\d+) \(feta_id: (.+)\)(?: \[(?:LaTeX|Markdown)\])?")
QUESTION_PATTERN = re.compile(r"^📝 Question: (.+)$")
HEADERS_PATTERN = re.compile(r"^📋 Headers \((\d+)\): (.+)$")
RELEVANT_PATTERN = re.compile(r"^🎯 Relevant columns: (.+)$")
BEST_PATTERN = re.compile(r"^   🔝 최고 유사도 파티션: sim=([0-9.]+) \(size=(\d+)\) \| cols=(.+)$")


def parse_column_list(text: str) -> List[str]:
    """Parse column list from string like ['col1', 'col2'] or <empty>."""
    text = text.strip()
    if text == "['<empty>']" or text == "<empty>":
        return []
    
    # Remove outer brackets and split by comma
    if text.startswith('[') and text.endswith(']'):
        text = text[1:-1]
    
    # Parse quoted column names
    columns = []
    for part in text.split(','):
        part = part.strip()
        if part.startswith("'") and part.endswith("'"):
            columns.append(part[1:-1])
        elif part.startswith('"') and part.endswith('"'):
            columns.append(part[1:-1])
        elif part and part != '<empty>':
            columns.append(part)
    
    return columns


def load_original_data(jsonl_path: Path) -> Dict[int, Dict[str, Any]]:
    """Load original SACU data indexed by feta_id."""
    data = {}
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if 'feta_id' not in record:
                    print(f"⚠️  Warning: Line {line_num} missing 'feta_id', skipping")
                    continue
                feta_id = record['feta_id']
                data[feta_id] = record
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Line {line_num} has invalid JSON: {e}, skipping")
                continue
    return data


def parse_log(log_path: Path) -> List[Dict[str, Any]]:
    """Parse column_subset_similarity log file."""
    results = []
    
    current_entry = None
    
    with log_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            
            # New table entry
            table_match = TABLE_PATTERN.match(line)
            if table_match:
                if current_entry:
                    results.append(current_entry)
                
                current_entry = {
                    'table_num': int(table_match.group(1)),
                    'feta_id': int(table_match.group(2)),
                    'question': None,
                    'headers': [],
                    'relevant_columns': [],
                    'best_columns': [],
                    'best_similarity': None,
                }
                continue
            
            if not current_entry:
                continue
            
            # Question
            question_match = QUESTION_PATTERN.match(line)
            if question_match:
                current_entry['question'] = question_match.group(1)
                continue
            
            # Headers
            headers_match = HEADERS_PATTERN.match(line)
            if headers_match:
                headers_str = headers_match.group(2)
                current_entry['headers'] = parse_column_list(headers_str)
                continue
            
            # Relevant columns
            relevant_match = RELEVANT_PATTERN.match(line)
            if relevant_match:
                relevant_str = relevant_match.group(1)
                current_entry['relevant_columns'] = parse_column_list(relevant_str)
                continue
            
            # Best subset
            best_match = BEST_PATTERN.match(line)
            if best_match:
                current_entry['best_similarity'] = float(best_match.group(1))
                current_entry['best_columns'] = parse_column_list(best_match.group(3))
                continue
    
    # Don't forget last entry
    if current_entry:
        results.append(current_entry)
    
    return results


def load_llm_results(json_path: Path) -> Dict[int, Dict[str, Any]]:
    """Load LLM column extraction results indexed by feta_id.
    
    Only supports format: {"feta_id": int, "output": {"relevant_columns": [...]}}
    Supports both JSON array and JSONL formats.
    """
    data = {}
    
    with json_path.open('r', encoding='utf-8') as f:
        content = f.read().strip()
    
        # Try JSON array format first
        try:
            llm_results = json.loads(content)
            if isinstance(llm_results, list):
                for entry in llm_results:
                    feta_id = int(entry['feta_id'])
                    # Only support {"feta_id": , "output": {}} format
                    if 'output' not in entry:
                        print(f"⚠️  Warning: Entry for feta_id {feta_id} missing 'output' field, skipping")
                        continue
                    if 'relevant_columns' not in entry['output']:
                        print(f"⚠️  Warning: Entry for feta_id {feta_id} missing 'output.relevant_columns', skipping")
                        continue
                    
                    columns = entry['output']['relevant_columns']
                    data[feta_id] = {
                        'llm_predicted_columns': columns,
                    }
                return data
        except json.JSONDecodeError:
            pass
        
        # Try JSONL format (one JSON object per line)
        f.seek(0)
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                feta_id = int(entry['feta_id'])
                
                # Only support {"feta_id": , "output": {}} format
                if 'output' not in entry:
                    print(f"⚠️  Warning: Line {line_num} (feta_id {feta_id}) missing 'output' field, skipping")
                    continue
                if 'relevant_columns' not in entry['output']:
                    print(f"⚠️  Warning: Line {line_num} (feta_id {feta_id}) missing 'output.relevant_columns', skipping")
                    continue
                
                columns = entry['output']['relevant_columns']
                data[feta_id] = {
                    'llm_predicted_columns': columns,
                }
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Failed to parse line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"⚠️  Warning: Failed to process line {line_num}: {e}")
                continue
    
    return data


def load_qa_subset_with_questions(json_path: Path) -> List[Dict[str, Any]]:
    """Load QA subset file with questions (each line is a separate record).
    
    Supports format: {"feta_id": int, "question": str, "relevant_columns": [...]}
    Each line becomes a separate record (multiple questions per feta_id are preserved).
    """
    items = []
    
    with json_path.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                feta_id = int(entry['feta_id'])
                
                # Support {"feta_id": , "question": , "relevant_columns": } format
                if 'question' not in entry:
                    print(f"⚠️  Warning: Line {line_num} (feta_id {feta_id}) missing 'question' field, skipping")
                    continue
                if 'relevant_columns' not in entry:
                    print(f"⚠️  Warning: Line {line_num} (feta_id {feta_id}) missing 'relevant_columns' field, skipping")
                    continue
                
                items.append({
                    'feta_id': feta_id,
                    'question': entry['question'],
                    'relevant_columns': entry['relevant_columns'],
                })
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Failed to parse line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"⚠️  Warning: Failed to process line {line_num}: {e}")
                continue
    
    return items


def create_subset_table(original_table: List[List[str]], columns_to_keep: List[str], 
                       all_headers: List[str]) -> List[List[str]]:
    """Extract subset of table with only specified columns."""
    if not columns_to_keep:
        # Empty subset - return just headers
        return [all_headers]
    
    # Find column indices
    indices = []
    for col in columns_to_keep:
        if col in all_headers:
            indices.append(all_headers.index(col))
    
    if not indices:
        return [all_headers]
    
    # Extract columns
    subset_table = []
    for row in original_table:
        subset_row = [row[i] if i < len(row) else "" for i in indices]
        subset_table.append(subset_row)
    
    return subset_table


def create_datasets_from_log(original_data: Dict[int, Dict[str, Any]], 
                             parsed_log: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Create two versions of datasets from log: best_subset, relevant_subset."""
    
    datasets = {
        'best_subset': [],
        'relevant_subset': [],
    }
    
    for entry in parsed_log:
        feta_id = entry['feta_id']
        
        if feta_id not in original_data:
            print(f"⚠️  Warning: feta_id {feta_id} not found in original data, skipping")
            continue
        
        original = original_data[feta_id]
        table_array = original['input']['table_array']
        question = original['input']['question']
        original_headers = table_array[0]
        
        # Relevant subset
        relevant_subset = create_subset_table(
            table_array, 
            entry['relevant_columns'], 
            original_headers
        )
        datasets['relevant_subset'].append({
            'feta_id': feta_id,
            'input': {
                'question': question,
                'table_array': relevant_subset,
            },
            'output': {
                'relevant_columns': entry['relevant_columns']
            },
            'metadata': {
                'original_columns': original_headers,
                'kept_columns': entry['relevant_columns'],
            }
        })
        
        # Best subset
        best_subset = create_subset_table(
            table_array,
            entry['best_columns'],
            original_headers
        )
        datasets['best_subset'].append({
            'feta_id': feta_id,
            'input': {
                'question': question,
                'table_array': best_subset,
            },
            'output': {
                'relevant_columns': entry['relevant_columns']  # Keep original for evaluation
            },
            'metadata': {
                'original_columns': original_headers,
                'kept_columns': entry['best_columns'],
                'best_similarity': entry['best_similarity'],
            }
        })
    
    return datasets


def create_llm_subset_dataset(original_data: Dict[int, Dict[str, Any]], 
                              llm_results: Dict[int, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Create LLM subset dataset.
    
    Returns:
        dataset: List of subset records
        missing_feta_ids: List of feta_ids not found in original data
    """
    
    dataset = []
    missing_feta_ids = []
    
    for feta_id, llm_result in llm_results.items():
        if feta_id not in original_data:
            missing_feta_ids.append(feta_id)
            continue
        
        original = original_data[feta_id]
        table_array = original['input']['table_array']
        question = original['input']['question']
        original_headers = table_array[0]
        
        # Get LLM predicted columns
        llm_columns = llm_result.get('llm_predicted_columns', [])
        
        # Create subset table
        llm_subset = create_subset_table(
            table_array,
            llm_columns,
            original_headers
        )
        
        # Create record (same format as best_subset and relevant_subset)
        record = {
            'feta_id': feta_id,
            'input': {
                'question': question,
                'table_array': llm_subset,
            },
            'output': {
                'relevant_columns': original.get('output', {}).get('relevant_columns', [])  # Keep original for evaluation
            },
            'metadata': {
                'original_columns': original_headers,
                'kept_columns': llm_columns,
            }
        }
        
        dataset.append(record)
    
    return dataset, missing_feta_ids


def create_qa_subset_dataset_with_questions(original_data: Dict[int, Dict[str, Any]], 
                                            qa_items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Create QA subset dataset with questions (each question becomes a separate record).
    
    Args:
        original_data: Original SACU data indexed by feta_id
        qa_items: List of QA items, each with feta_id, question, and relevant_columns
        
    Returns:
        dataset: List of subset records (one per question)
        missing_feta_ids: List of feta_ids not found in original data
    """
    
    dataset = []
    missing_feta_ids = []
    
    for qa_item in qa_items:
        feta_id = qa_item['feta_id']
        question = qa_item['question']
        relevant_columns = qa_item['relevant_columns']
        
        if feta_id not in original_data:
            if feta_id not in missing_feta_ids:
                missing_feta_ids.append(feta_id)
            continue
        
        original = original_data[feta_id]
        table_array = original['input']['table_array']
        original_headers = table_array[0]
        
        # Create subset table
        subset_table = create_subset_table(
            table_array,
            relevant_columns,
            original_headers
        )
        
        # Create record (same format as dev_QA_subset.jsonl)
        record = {
            'feta_id': feta_id,
            'input': {
                'question': question,
                'table_array': subset_table,
            }
        }
        
        dataset.append(record)
    
    return dataset, missing_feta_ids


def main():
    parser = argparse.ArgumentParser(
        description="Create subset table datasets from logs and/or LLM results"
    )
    
    # Input sources
    parser.add_argument(
        '--log_file',
        type=Path,
        default=None,
        help='Path to column_subset_similarity log file (for best_subset, relevant_subset)'
    )
    parser.add_argument(
        '--llm_results',
        type=Path,
        default=None,
        help='Path to LLM results file (JSON array or JSONL format). Must be in format: {"feta_id": int, "output": {"relevant_columns": [...]}} (for llm_subset)'
    )
    parser.add_argument(
        '--original_jsonl',
        type=Path,
        required=True,
        help='Path to original SACU JSONL file (e.g., SACU_dev.jsonl)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.'),
        help='Output directory for generated datasets (default: current directory)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='',
        help='Prefix for output filenames (e.g., "dev" -> dev_best_subset.jsonl)'
    )
    
    args = parser.parse_args()
    
    # Check that at least one input source is provided
    if not args.log_file and not args.llm_results:
        parser.error("At least one of --log_file or --llm_results must be provided")
    
    print("="*70)
    print("📊 Create Subset Table Datasets")
    print("="*70)
    
    # Load original data
    print(f"\n📖 Loading original data from {args.original_jsonl}...")
    original_data = load_original_data(args.original_jsonl)
    print(f"   ✅ Loaded {len(original_data)} tables")
    
    # Process log file if provided
    if args.log_file:
        print(f"\n📋 Parsing log file {args.log_file}...")
        parsed_log = parse_log(args.log_file)
        print(f"   ✅ Parsed {len(parsed_log)} table entries")
        
        print(f"\n🔧 Creating datasets from log...")
        datasets = create_datasets_from_log(original_data, parsed_log)
        
        # Save log-based datasets
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        for dataset_name, records in datasets.items():
            prefix = f"{args.prefix}_" if args.prefix else ""
            output_path = args.output_dir / f"{prefix}{dataset_name}.jsonl"
            
            with output_path.open('w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            print(f"   💾 Saved {len(records)} tables to {output_path}")
        
        print("\n📊 Log-based datasets summary:")
        print(f"   Relevant subset: {len(datasets['relevant_subset'])} tables")
        print(f"   Best subset: {len(datasets['best_subset'])} tables")
    
    # Process LLM results if provided
    if args.llm_results:
        # Try to load as QA subset with questions first (format: {"feta_id": ..., "question": ..., "relevant_columns": ...})
        print(f"\n📋 Loading QA subset from {args.llm_results}...")
        qa_items = load_qa_subset_with_questions(args.llm_results)
        
        if qa_items:
            # Format with questions - each question becomes a separate record
            print(f"   ✅ Loaded {len(qa_items)} QA items (with questions)")
            
            print(f"\n🔧 Creating QA subset dataset with questions...")
            qa_dataset, missing_feta_ids = create_qa_subset_dataset_with_questions(original_data, qa_items)
            print(f"   ✅ Created {len(qa_dataset)} subset records")
            
            # Warnings
            if missing_feta_ids:
                print(f"\n⚠️  Warning: {len(missing_feta_ids)} feta_ids not found in original data")
                print(f"   Examples: {missing_feta_ids[:5]}")
            
            # Save QA subset
            args.output_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"{args.prefix}_" if args.prefix else ""
            qa_output_path = args.output_dir / f"{prefix}qa_subset.jsonl"
            
            print(f"\n💾 Saving to {qa_output_path}...")
            with qa_output_path.open('w', encoding='utf-8') as f:
                for record in qa_dataset:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            print(f"   ✅ Saved {len(qa_dataset)} entries")
            
            # Statistics
            total = len(qa_dataset)
            unique_feta_ids = len(set(r['feta_id'] for r in qa_dataset))
            avg_questions_per_table = total / unique_feta_ids if unique_feta_ids > 0 else 0
            
            print("\n📊 Summary:")
            print(f"   Total QA items: {len(qa_items)}")
            print(f"   Successfully created: {len(qa_dataset)}")
            print(f"   Unique feta_ids: {unique_feta_ids}")
            print(f"   Average questions per table: {avg_questions_per_table:.2f}")
            print(f"   Missing feta_ids: {len(missing_feta_ids)}")
        else:
            # Try to load as LLM results (format: {"feta_id": ..., "output": {"relevant_columns": ...}})
            print(f"\n📋 Loading LLM results from {args.llm_results}...")
            llm_results = load_llm_results(args.llm_results)
            print(f"   ✅ Loaded {len(llm_results)} LLM predictions")
            
            print(f"\n🔧 Creating LLM subset dataset...")
            llm_dataset, missing_feta_ids = create_llm_subset_dataset(original_data, llm_results)
            print(f"   ✅ Created {len(llm_dataset)} subset tables")
            
            # Warnings
            if missing_feta_ids:
                print(f"\n⚠️  Warning: {len(missing_feta_ids)} feta_ids not found in original data")
                print(f"   Examples: {missing_feta_ids[:5]}")
            
            # Save LLM subset
            args.output_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"{args.prefix}_" if args.prefix else ""
            llm_output_path = args.output_dir / f"{prefix}llm_subset.jsonl"
            
            print(f"\n💾 Saving to {llm_output_path}...")
            with llm_output_path.open('w', encoding='utf-8') as f:
                for record in llm_dataset:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            print(f"   ✅ Saved {len(llm_dataset)} entries")
            
            # Statistics
            total = len(llm_dataset)
            empty_subsets = sum(1 for r in llm_dataset if not r['metadata']['kept_columns'])
            avg_columns = sum(len(r['metadata']['kept_columns']) for r in llm_dataset) / total if total > 0 else 0
            
            print("\n📊 Summary:")
            print(f"   Total LLM results: {len(llm_results)}")
            print(f"   Successfully created: {len(llm_dataset)}")
            print(f"   Missing feta_ids: {len(missing_feta_ids)}")
            print(f"   Empty subsets: {empty_subsets}")
            print(f"   Average columns per subset: {avg_columns:.2f}")
    
    print("\n" + "="*70)
    print("✅ Done!")
    print("="*70)
    print("\n💡 Usage examples:")
    print("   # Create all subsets from log:")
    print("   python create_subset_tables.py --log_file log.txt --original_jsonl SACU_dev.jsonl --prefix dev")
    print("   # Create LLM subset:")
    print("   python create_subset_tables.py --llm_results results.json --original_jsonl SACU_dev.jsonl --prefix dev")
    print("   # Create all subsets (log + LLM):")
    print("   python create_subset_tables.py --log_file log.txt --llm_results results.json --original_jsonl SACU_dev.jsonl --prefix dev")


if __name__ == '__main__':
    main()
