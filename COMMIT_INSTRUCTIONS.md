# Git Setup 및 Commit 가이드

터미널에서 다음 명령어를 순서대로 실행하세요:

## 1. Git 초기화 및 브랜치 설정

```bash
cd "/Users/hosubin/Desktop/KU/4학년/Computer Science Colloquium/20252R0136COSE40500"

# Git 초기화 (이미 되어있으면 스킵)
git init

# 브랜치를 main으로 설정
git branch -M main
```

## 2. 원격 저장소 추가

```bash
# 원격 저장소가 없으면 추가
git remote add origin https://github.com/SOOB2NHO/20252R0136COSE40500.git

# 이미 추가되어 있으면 확인
git remote -v
```

## 3. 모든 파일 추가 및 커밋

```bash
# 모든 변경사항 추가
git add .

# 커밋
git commit -m "first commit"

# 또는 여러 개의 커밋을 만들고 싶다면:
# git add README.md
# git commit -m "Add README.md"
# git add evaluate.py
# git commit -m "Add evaluation script"
# git add analyze_query.py
# git commit -m "Add query analysis script"
# ... (반복)
```

## 4. GitHub에 푸시 (선택사항)

```bash
git push -u origin main
```

## 현재 파일 목록
- evaluate.py
- analyze_query.py
- README.md
- requirements.txt
- .gitignore
- setup.sh
- setup_commits.sh

