# 20252R0136COSE40500

SACU 데이터셋 서브셋 생성 및 평가 프로젝트

## 프로젝트 개요

이 저장소는 SACU (Structured Answer Complex Understanding) 데이터셋에 대한 서브셋 테이블 생성 및 검색(retrieval) 성능 평가를 위한 도구들을 포함합니다.

## 주요 기능

- **데이터셋 생성**: 컬럼 부분집합 분석 결과로부터 서브셋 테이블 JSONL 파일 생성
- **컬럼 추출**: 다양한 서브셋(QA, Best, LLM, Relevant)에서 관련 컬럼 정보 추출 및 비교
- **검색 평가**: 서브셋 기반 테이블 검색 성능 평가
- **쿼리 분석**: Original query와 LLM query들의 유사도 비교 분석
- **컬럼 유사도 분석**: 컬럼 부분집합 간 유사도 계산

## 프로젝트 구조

```
.
├── dataset/                      # 데이터셋 생성 및 처리 스크립트
│   ├── extract_relevant_columns.py    # 서브셋에서 관련 컬럼 추출
│   ├── create_subset_tables.py        # 서브셋 테이블 JSONL 파일 생성
│   └── *.jsonl                        # 데이터 파일
├── evaluation/                   # 평가 및 분석 스크립트
│   ├── evaluate.py                   # 검색 성능 평가
│   ├── analyze_query.py              # 쿼리 유사도 분석
│   └── column_subset_similarity.py   # 컬럼 부분집합 유사도 분석
├── requirements.txt              # Python 의존성
└── README.md                     # 이 파일
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 서브셋 테이블 생성

```bash
# 로그 파일이나 LLM 결과로부터 서브셋 테이블 생성
python dataset/create_subset_tables.py --input <log_file> --output <output.jsonl>
```

### 2. 관련 컬럼 추출 및 비교

```bash
# 다양한 서브셋에서 컬럼 정보 추출 및 비교
python dataset/extract_relevant_columns.py
```

### 3. 검색 성능 평가

```bash
# 서브셋 기반 테이블 검색 성능 평가
python evaluation/evaluate.py --subset-dir <path> --sacu-dir <path>
```

### 4. 쿼리 유사도 분석

```bash
# Original query와 LLM query들의 유사도 비교
python evaluation/analyze_query.py --input <data.jsonl>
```

## 지원하는 서브셋 타입

- **Baseline**: 원본 테이블
- **Relevant Subset**: 관련 컬럼만 포함
- **Best Subset**: 최고 유사도 파티션
- **LLM Subset**: LLM이 추출한 컬럼
- **QA Subset**: QA 관련 컬럼
- **QA Query Subset**: QA query 기반 서브셋
- **LaTeX Best Subset**: LaTeX 최적 서브셋

## 평가 메트릭

- Recall@K
- MRR (Mean Reciprocal Rank)
- Accuracy
- Capped Recall

## 라이선스

이 프로젝트는 COSE40500 강의를 위한 것입니다.
