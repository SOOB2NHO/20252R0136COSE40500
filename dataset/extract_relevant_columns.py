#!/usr/bin/env python3
"""
각 서브셋 파일에서 relevant_columns 추출 및 비교

- QA, Best, LLM, Relevant subset 파일에서 컬럼 정보 추출
- 각 feta_id별 컬럼 비교
- 통계 및 분석 결과 출력
"""
import json
from collections import defaultdict

# 파일 경로
files = {
    'QA': 'data/SACU/QA_tables/dev_QA_subset.jsonl',
    'Best': 'data/SACU/subset_tables/dev_best_subset.jsonl',
    'LLM': 'data/SACU/subset_tables/dev_llm_subset.jsonl',
    'Relevant': 'data/SACU/subset_tables/dev_relevant_subset.jsonl'
}

# 각 파일에서 전체 데이터 추출 (question, table_array 포함)
results = {}
full_data = {}  # 전체 데이터 저장 (question, table_array 포함)
for name, filepath in files.items():
    print(f"\n{'='*100}")
    print(f"{name} Subset - Relevant Columns")
    print(f"{'='*100}")
    
    feta_columns = defaultdict(list)
    feta_full_data = defaultdict(list)  # 전체 데이터 저장
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                feta_id = data['feta_id']
                table_array = data.get('input', {}).get('table_array', [])
                
                # 모든 파일에서 table_array의 헤더(첫 번째 행)를 relevant_columns로 사용
                relevant_cols = table_array[0] if table_array else []
                
                # 전체 데이터 저장
                entry_data = {
                    'question': data.get('input', {}).get('question', ''),
                    'table_array': table_array,
                    'relevant_columns': relevant_cols
                }
                feta_full_data[feta_id].append(entry_data)
                
                # 중복 제거 (같은 feta_id에 대해 같은 relevant_columns가 여러 번 나올 수 있음)
                if relevant_cols not in feta_columns[feta_id]:
                    feta_columns[feta_id].append(relevant_cols)
    
    # 결과 저장
    results[name] = dict(feta_columns)
    full_data[name] = dict(feta_full_data)
    
    # 출력 (상위 50개만)
    count = 0
    for feta_id in sorted(feta_columns.keys()):
        if count >= 50:
            break
        print(f"\nFeta ID: {feta_id}")
        for idx, cols in enumerate(feta_columns[feta_id], 1):
            if len(feta_columns[feta_id]) > 1:
                print(f"  Entry {idx}: {cols}")
            else:
                print(f"  {cols}")
        count += 1

# 요약 통계
print(f"\n{'='*100}")
print("Summary Statistics")
print(f"{'='*100}")

all_feta_ids = set()
for name, data in results.items():
    all_feta_ids.update(data.keys())

print(f"\nTotal unique feta_ids across all files: {len(all_feta_ids)}")
print(f"\nRelevant columns per file:")
for name, data in results.items():
    total_entries = sum(len(cols_list) for cols_list in data.values())
    unique_patterns = set()
    for cols_list in data.values():
        for cols in cols_list:
            unique_patterns.add(tuple(sorted(cols)))
    
    print(f"  {name:10s}: {len(data)} feta_ids, {total_entries} entries, {len(unique_patterns)} unique column patterns")

# QA와 Best/LLM/Relevant 비교
print(f"\n{'='*100}")
print("QA vs Best/LLM/Relevant - Relevant Columns Match Check")
print(f"{'='*100}")

qa_data = results['QA']
best_data = results['Best']
llm_data = results['LLM']
relevant_data = results['Relevant']

# 상위 50개의 feta_id만 처리
top_50_feta_ids = sorted(qa_data.keys())[:50]
print(f"\nProcessing top 50 feta_ids: {top_50_feta_ids}")

# feta_id별 match 통계
feta_id_match_stats = {
    'Best': {},
    'LLM': {},
    'Relevant': {}
}

# Match되는 데이터 수집
matched_data = {
    'Best': [],
    'LLM': [],
    'Relevant': []
}

# 모든 파일의 relevant_columns를 하나의 JSONL 파일로 통합 저장
output_file = 'data/SACU/relevant_columns_all.jsonl'
with open(output_file, 'w') as f:
    # 상위 50개의 feta_id에 대해 처리
    for feta_id in top_50_feta_ids:
        qa_cols_list = qa_data[feta_id]
        best_cols = set(tuple(sorted(cols)) for cols in best_data.get(feta_id, [[]])[0:1]) if feta_id in best_data else set()
        llm_cols = set(tuple(sorted(cols)) for cols in llm_data.get(feta_id, [[]])[0:1]) if feta_id in llm_data else set()
        rel_cols = set(tuple(sorted(cols)) for cols in relevant_data.get(feta_id, [[]])[0:1]) if feta_id in relevant_data else set()
        
        # feta_id별 match 초기화
        feta_id_match_stats['Best'][feta_id] = {'match': False, 'total_qa': 0}
        feta_id_match_stats['LLM'][feta_id] = {'match': False, 'total_qa': 0}
        feta_id_match_stats['Relevant'][feta_id] = {'match': False, 'total_qa': 0}
        
        # QA의 각 항목을 별도 줄로 저장
        qa_full_list = full_data['QA'].get(feta_id, [])
        for qa_entry in qa_full_list:
            qa_cols = qa_entry['relevant_columns']
            qa_cols_set = tuple(sorted(qa_cols))
            
            # 일치 여부 체크
            matches_best = qa_cols_set in best_cols if best_cols else False
            matches_llm = qa_cols_set in llm_cols if llm_cols else False
            matches_relevant = qa_cols_set in rel_cols if rel_cols else False
            
            # feta_id별 통계 업데이트
            feta_id_match_stats['Best'][feta_id]['total_qa'] += 1
            if matches_best:
                feta_id_match_stats['Best'][feta_id]['match'] = True
            feta_id_match_stats['LLM'][feta_id]['total_qa'] += 1
            if matches_llm:
                feta_id_match_stats['LLM'][feta_id]['match'] = True
            feta_id_match_stats['Relevant'][feta_id]['total_qa'] += 1
            if matches_relevant:
                feta_id_match_stats['Relevant'][feta_id]['match'] = True
            
            # Match되는 데이터 수집
            if matches_best:
                matched_data['Best'].append({
                    'feta_id': feta_id,
                    'QA': qa_cols,
                    'Best': sorted(best_data.get(feta_id, [[]])[0]) if feta_id in best_data else []
                })
            if matches_llm:
                matched_data['LLM'].append({
                    'feta_id': feta_id,
                    'QA': qa_cols,
                    'LLM': sorted(llm_data.get(feta_id, [[]])[0]) if feta_id in llm_data else []
                })
            if matches_relevant:
                matched_data['Relevant'].append({
                    'feta_id': feta_id,
                    'QA': qa_cols,
                    'Relevant': sorted(relevant_data.get(feta_id, [[]])[0]) if feta_id in relevant_data else []
                })
            
            # QA 항목을 JSONL에 저장 (헤더만 저장)
            table_array_header = qa_entry['table_array'][:1] if qa_entry['table_array'] else []
            entry = {
                'from': 'QA',
                'feta_id': feta_id,
                'input': {
                    'question': qa_entry['question'],
                    'table_array': table_array_header
                },
                'output': {
                    'relevant_columns': qa_cols
                }
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Best 항목 저장 (첫 번째 항목만, 헤더만 저장)
        if feta_id in full_data['Best'] and len(full_data['Best'][feta_id]) > 0:
            best_entry = full_data['Best'][feta_id][0]
            table_array_header = best_entry.get('table_array', [])[:1] if best_entry.get('table_array', []) else []
            entry = {
                'from': 'Best',
                'feta_id': feta_id,
                'input': {
                    'question': best_entry.get('question', ''),
                    'table_array': table_array_header
                },
                'output': {
                    'relevant_columns': best_entry['relevant_columns']
                }
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # LLM 항목 저장 (첫 번째 항목만, 헤더만 저장)
        if feta_id in full_data['LLM'] and len(full_data['LLM'][feta_id]) > 0:
            llm_entry = full_data['LLM'][feta_id][0]
            table_array_header = llm_entry.get('table_array', [])[:1] if llm_entry.get('table_array', []) else []
            entry = {
                'from': 'LLM',
                'feta_id': feta_id,
                'input': {
                    'question': llm_entry.get('question', ''),
                    'table_array': table_array_header
                },
                'output': {
                    'relevant_columns': llm_entry['relevant_columns']
                }
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Relevant 항목 저장 (첫 번째 항목만, 헤더만 저장)
        if feta_id in full_data['Relevant'] and len(full_data['Relevant'][feta_id]) > 0:
            rel_entry = full_data['Relevant'][feta_id][0]
            table_array_header = rel_entry.get('table_array', [])[:1] if rel_entry.get('table_array', []) else []
            entry = {
                'from': 'Relevant',
                'feta_id': feta_id,
                'input': {
                    'question': rel_entry.get('question', ''),
                    'table_array': table_array_header
                },
                'output': {
                    'relevant_columns': rel_entry['relevant_columns']
                }
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Match 통계를 표 형식으로 출력 (feta_id 기준)
print(f"\n{'='*70}")
print("📊 Match Statistics (by feta_id)")
print(f"{'='*70}")
print(f"{'Source':<15} {'Matched feta_ids':<20} {'Total feta_ids':<20} {'Match Rate':<15}")
print(f"{'-'*70}")
for source in ['Best', 'LLM', 'Relevant']:
    matched_count = sum(1 for stats in feta_id_match_stats[source].values() if stats['match'])
    total_count = len(feta_id_match_stats[source])
    match_rate = (matched_count / total_count * 100) if total_count > 0 else 0
    print(f"{source:<15} {matched_count:<20} {total_count:<20} {match_rate:.1f}%")
print(f"{'='*70}")

# feta_id별 상세 통계
print(f"\n{'='*70}")
print("📋 Detailed Statistics by feta_id")
print(f"{'='*70}")
print(f"{'feta_id':<10} {'Best':<10} {'LLM':<10} {'Relevant':<10} {'QA entries':<10}")
print(f"{'-'*70}")
for feta_id in top_50_feta_ids:
    best_match = '✓' if feta_id_match_stats['Best'][feta_id]['match'] else '✗'
    llm_match = '✓' if feta_id_match_stats['LLM'][feta_id]['match'] else '✗'
    rel_match = '✓' if feta_id_match_stats['Relevant'][feta_id]['match'] else '✗'
    qa_count = feta_id_match_stats['Best'][feta_id]['total_qa']
    print(f"{feta_id:<10} {best_match:<10} {llm_match:<10} {rel_match:<10} {qa_count:<10}")
print(f"{'='*70}")

# Match되는 데이터 출력
print(f"\n{'='*70}")
print("📋 Matched Cases")
print(f"{'='*70}")

for source in ['Best', 'LLM', 'Relevant']:
    if matched_data[source]:
        print(f"\n{source} - {len(matched_data[source])} matched cases:")
        print(f"{'-'*70}")
        for idx, match_case in enumerate(matched_data[source][:50], 1):  # 상위 50개만 출력
            print(f"  {idx}. Feta ID: {match_case['feta_id']}")
            print(f"     QA:        {match_case['QA']}")
            print(f"     {source}:     {match_case[source]}")
        if len(matched_data[source]) > 50:
            print(f"     ... and {len(matched_data[source]) - 50} more cases")

print(f"\nSaved all relevant_columns to {output_file} (JSONL format)")

print(f"\n{'='*100}")
print("Done!")
print(f"{'='*100}")

