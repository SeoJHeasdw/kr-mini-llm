# 한국어 데이터 수집 가이드

Korean LLM 학습을 위한 한국어 텍스트 데이터 수집 가이드입니다.

## 📊 데이터 요구사항

### 최소 요구사항
- **Small 모델 (150M)**: 최소 1GB, 권장 5GB
- **Medium 모델 (350M)**: 최소 5GB, 권장 10GB
- **Large 모델 (800M)**: 최소 10GB, 권장 20GB+

### 품질 기준
- ✅ 깨끗한 한국어 텍스트
- ✅ 다양한 도메인 (뉴스, 위키, 블로그, 책 등)
- ✅ 문법적으로 올바른 문장
- ❌ 중복 제거
- ❌ 광고/스팸 제거
- ❌ 개인정보 제거

## 🎯 추천 데이터 소스

### 1. 공개 데이터셋 (가장 쉬움)

#### AI Hub (한국어 AI 데이터)
- **URL**: https://aihub.or.kr/
- **데이터**: 뉴스, 대화, 문서 등
- **크기**: 수십 GB
- **장점**: 고품질, 정제됨, 무료
- **단점**: 회원가입 필요

```bash
# 다운로드 후
mkdir -p data/raw/aihub
# 압축 해제 및 텍스트 추출
```

#### 모두의 말뭉치
- **URL**: https://corpus.korean.go.kr/
- **데이터**: 신문, 문어, 구어, 메신저 등
- **크기**: 수 GB
- **장점**: 국립국어원 제공, 고품질
- **단점**: 신청 및 승인 필요

#### Hugging Face 한국어 데이터셋
```python
from datasets import load_dataset

# 한국어 위키피디아
wiki = load_dataset("wikipedia", "20220301.ko")

# 한국어 뉴스
news = load_dataset("klue", "ynat")

# 한국어 대화
dialogue = load_dataset("smilegate-ai/kor_nlu")
```

### 2. 위키피디아 (추천)

#### 한국어 위키피디아 덤프
- **URL**: https://dumps.wikimedia.org/kowiki/
- **크기**: 약 1-2GB (압축), 5-10GB (텍스트)
- **장점**: 고품질, 다양한 주제, 무료
- **단점**: 전처리 필요

```bash
# 다운로드
wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2

# WikiExtractor로 텍스트 추출
pip install wikiextractor
python -m wikiextractor.WikiExtractor \
    kowiki-latest-pages-articles.xml.bz2 \
    --output data/raw/wiki \
    --bytes 100M \
    --json
```

### 3. 뉴스 크롤링

#### 네이버 뉴스
```python
import requests
from bs4 import BeautifulSoup
import time

def crawl_naver_news(keyword, pages=100):
    """네이버 뉴스 크롤링"""
    articles = []
    
    for page in range(1, pages + 1):
        url = f"https://search.naver.com/search.naver?where=news&query={keyword}&start={page*10}"
        
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 뉴스 링크 추출
            news_links = soup.select('.news_tit')
            
            for link in news_links:
                article_url = link['href']
                # 각 기사 크롤링
                article_text = crawl_article(article_url)
                articles.append(article_text)
                
                time.sleep(0.5)  # 서버 부하 방지
                
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    return articles

# 사용 예시
keywords = ['기술', '경제', '사회', '문화', '과학']
for keyword in keywords:
    articles = crawl_naver_news(keyword, pages=50)
    # 저장
```

**주의사항**:
- ⚠️ robots.txt 확인
- ⚠️ 크롤링 속도 제한 (0.5초 이상 간격)
- ⚠️ 저작권 확인 (학습 목적 명시)

### 4. 공공 데이터

#### 국회 회의록
- **URL**: https://open.assembly.go.kr/
- **데이터**: 국회 본회의, 위원회 회의록
- **크기**: 수 GB
- **장점**: 공식 문서, 무료

#### 법령 데이터
- **URL**: https://www.law.go.kr/
- **데이터**: 법률, 시행령, 시행규칙
- **크기**: 수백 MB
- **장점**: 정제된 텍스트

#### 판례 데이터
- **URL**: https://www.law.go.kr/LSW/precInfoP.do
- **데이터**: 대법원 판례
- **크기**: 수 GB

### 5. 책/문학 (저작권 주의)

#### 한국 고전 문학
- **URL**: https://www.nl.go.kr/ (국립중앙도서관)
- **데이터**: 저작권 만료 고전 문학
- **크기**: 수백 MB
- **장점**: 고품질 문학 텍스트

#### 프로젝트 구텐베르크 (한국어)
- **URL**: https://www.gutenberg.org/
- **데이터**: 저작권 만료 한국어 책
- **크기**: 제한적

### 6. 소셜 미디어 (주의 필요)

#### Reddit (한국 서브레딧)
```python
import praw

reddit = praw.Reddit(
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_SECRET',
    user_agent='Korean LLM Data Collector'
)

# 한국 관련 서브레딧
subreddits = ['korea', 'hanguk', 'korean']

for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    for submission in subreddit.hot(limit=1000):
        text = submission.title + "\n" + submission.selftext
        # 저장
```

**주의사항**:
- ⚠️ 개인정보 제거 필수
- ⚠️ 욕설/혐오 표현 필터링
- ⚠️ API 사용 제한 확인

## 🛠️ 데이터 수집 스크립트

### 통합 크롤러 예제

```python
"""
한국어 데이터 수집 스크립트
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict
import requests
from bs4 import BeautifulSoup

class KoreanDataCollector:
    """한국어 데이터 수집기"""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_wikipedia(self):
        """위키피디아 데이터 수집"""
        print("📚 위키피디아 데이터 다운로드 중...")
        
        # WikiExtractor 사용
        os.system("""
            wget https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2
            python -m wikiextractor.WikiExtractor \
                kowiki-latest-pages-articles.xml.bz2 \
                --output data/raw/wiki \
                --bytes 100M \
                --json
        """)
        
    def collect_news(self, keywords: List[str], pages_per_keyword: int = 100):
        """뉴스 데이터 수집"""
        print(f"📰 뉴스 데이터 수집 중... (키워드: {len(keywords)}개)")
        
        all_articles = []
        
        for keyword in keywords:
            print(f"  - '{keyword}' 검색 중...")
            articles = self._crawl_naver_news(keyword, pages_per_keyword)
            all_articles.extend(articles)
            time.sleep(1)
        
        # 저장
        output_file = self.output_dir / "news.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for article in all_articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        print(f"✅ 뉴스 {len(all_articles)}개 수집 완료")
        
    def _crawl_naver_news(self, keyword: str, pages: int) -> List[Dict]:
        """네이버 뉴스 크롤링"""
        articles = []
        
        for page in range(1, pages + 1):
            url = f"https://search.naver.com/search.naver?where=news&query={keyword}&start={page*10}"
            
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                news_items = soup.select('.news_area')
                
                for item in news_items:
                    try:
                        title = item.select_one('.news_tit').text.strip()
                        content = item.select_one('.news_dsc').text.strip()
                        
                        articles.append({
                            'title': title,
                            'content': content,
                            'keyword': keyword
                        })
                    except:
                        continue
                
                time.sleep(0.5)  # 서버 부하 방지
                
            except Exception as e:
                print(f"    오류: {e}")
                continue
        
        return articles
    
    def collect_public_data(self):
        """공공 데이터 수집"""
        print("🏛️ 공공 데이터 수집 중...")
        
        # 국회 회의록, 법령 등
        # API 키 필요
        pass
    
    def merge_all_data(self):
        """모든 데이터 병합"""
        print("🔄 데이터 병합 중...")
        
        all_texts = []
        
        # 위키피디아
        wiki_dir = self.output_dir / "wiki"
        if wiki_dir.exists():
            for file in wiki_dir.rglob("*.json"):
                with open(file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        all_texts.append(data['text'])
        
        # 뉴스
        news_file = self.output_dir / "news.jsonl"
        if news_file.exists():
            with open(news_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    all_texts.append(data['title'] + '\n' + data['content'])
        
        # 통합 파일 저장
        output_file = self.output_dir / "korean_corpus.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in all_texts:
                f.write(text + '\n\n')
        
        print(f"✅ 총 {len(all_texts)}개 문서 병합 완료")
        print(f"   출력: {output_file}")
        
        # 통계
        total_chars = sum(len(text) for text in all_texts)
        print(f"   총 문자 수: {total_chars:,}")
        print(f"   예상 크기: {total_chars / 1024 / 1024:.1f} MB")

# 사용 예시
if __name__ == "__main__":
    collector = KoreanDataCollector()
    
    # 1. 위키피디아 수집
    # collector.collect_wikipedia()
    
    # 2. 뉴스 수집
    keywords = [
        '인공지능', '기술', '경제', '사회', '문화',
        '과학', '정치', '스포츠', '건강', '교육'
    ]
    collector.collect_news(keywords, pages_per_keyword=50)
    
    # 3. 데이터 병합
    collector.merge_all_data()
```

## 📋 데이터 전처리

### 1. 텍스트 정제

```python
import re

def clean_text(text: str) -> str:
    """텍스트 정제"""
    
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # URL 제거
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # 이메일 제거
    text = re.sub(r'\S+@\S+', '', text)
    
    # 전화번호 제거
    text = re.sub(r'\d{2,3}-\d{3,4}-\d{4}', '', text)
    
    # 특수문자 정리 (한글, 영문, 숫자, 기본 문장부호만 유지)
    text = re.sub(r'[^가-힣a-zA-Z0-9\s.,!?\'\"()[\]{}:;-]', '', text)
    
    # 연속 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 연속 줄바꿈 제거 (최대 2개)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()
```

### 2. 중복 제거

```python
from collections import defaultdict

def remove_duplicates(texts: List[str], threshold: float = 0.9) -> List[str]:
    """중복 문서 제거"""
    
    unique_texts = []
    seen_hashes = set()
    
    for text in texts:
        # 간단한 해시 기반 중복 제거
        text_hash = hash(text[:1000])  # 첫 1000자로 해시
        
        if text_hash not in seen_hashes:
            unique_texts.append(text)
            seen_hashes.add(text_hash)
    
    print(f"중복 제거: {len(texts)} → {len(unique_texts)}")
    return unique_texts
```

### 3. 품질 필터링

```python
def filter_quality(text: str, min_length: int = 100) -> bool:
    """품질 필터링"""
    
    # 너무 짧은 텍스트
    if len(text) < min_length:
        return False
    
    # 한글 비율 확인 (최소 50%)
    korean_chars = len(re.findall(r'[가-힣]', text))
    if korean_chars / len(text) < 0.5:
        return False
    
    # 반복 패턴 확인
    if re.search(r'(.{10,})\1{3,}', text):
        return False
    
    return True
```

## 📦 데이터 구조

### 권장 디렉토리 구조

```
data/
├── raw/                    # 원본 데이터
│   ├── wiki/              # 위키피디아
│   ├── news.jsonl         # 뉴스
│   ├── books/             # 책
│   └── public/            # 공공 데이터
├── processed/             # 전처리된 데이터
│   ├── train.txt          # 학습 데이터
│   ├── valid.txt          # 검증 데이터
│   └── test.txt           # 테스트 데이터
└── korean_corpus.txt      # 통합 말뭉치
```

## 🎯 추천 수집 전략

### 초보자 (빠른 시작)
1. **Hugging Face 데이터셋** 사용
2. **한국어 위키피디아** 다운로드
3. 총 5-10GB 목표

### 중급자 (균형잡힌 품질)
1. 위키피디아 + 뉴스 크롤링
2. AI Hub 공개 데이터셋
3. 총 10-20GB 목표

### 고급자 (최고 품질)
1. 다양한 소스 조합
2. 도메인별 균형 맞추기
3. 철저한 전처리
4. 총 20GB+ 목표

## ⚖️ 법적 고려사항

### 저작권
- ✅ 공개 데이터셋 사용
- ✅ 저작권 만료 자료
- ✅ 크리에이티브 커먼즈 라이선스
- ⚠️ 뉴스/블로그: 학습 목적 명시
- ❌ 유료 콘텐츠 무단 사용

### 개인정보
- ❌ 이름, 주소, 전화번호 제거
- ❌ 이메일, 주민등록번호 제거
- ✅ 익명화 처리

### 크롤링 에티켓
- ✅ robots.txt 준수
- ✅ 적절한 속도 제한 (0.5초 이상)
- ✅ User-Agent 명시
- ❌ 서버 과부하 유발 금지

## 📊 데이터 품질 체크리스트

- [ ] 총 크기: 모델에 맞는 충분한 양
- [ ] 다양성: 여러 도메인 포함
- [ ] 품질: 문법적으로 올바름
- [ ] 정제: HTML, URL 등 제거
- [ ] 중복 제거: 같은 내용 반복 없음
- [ ] 균형: 특정 주제 편중 없음
- [ ] 인코딩: UTF-8
- [ ] 개인정보: 모두 제거됨

---

**다음 단계**: 데이터 수집 후 `scripts/prepare_data.py`로 전처리하세요.

**Made with Bob** 🤖