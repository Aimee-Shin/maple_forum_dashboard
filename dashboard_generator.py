import os
import re
import json
import argparse
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
import time
from bs4 import BeautifulSoup

# --- Paths Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")
TEMPLATE_FILE = os.path.join(BASE_DIR, "dashboard_template.html")
OUTPUT_FILE = os.path.join(BASE_DIR, "maple_forum_dashboard.html")

# --- Environment Variable Loader ---
def load_env():
    env_vars = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    env_vars[key.strip()] = val.strip()
    return env_vars

ENV = load_env()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or ENV.get("GEMINI_API_KEY", "")

# --- Gemini API Call ---
def call_gemini_api(prompt):
    """
    Call Gemini 2.5 Flash API using urllib.request as referenced in test_gemini.py.
    Includes rate limit (HTTP 429) retry logic.
    """
    if not GEMINI_API_KEY:
        print("[오류] Gemini API Key가 .env 파일에 설정되지 않았습니다.")
        return None

    # Base safety delay of 13 seconds to strictly respect the 5 RPM limit
    time.sleep(13)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    data = json.dumps(payload).encode("utf-8")
    
    max_retries = 3
    retry_delay = 60 # wait 60 seconds on rate limit

    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        try:
            response = urllib.request.urlopen(req, timeout=30)
            res_data = json.loads(response.read().decode("utf-8"))
            text_response = res_data['candidates'][0]['content']['parts'][0]['text']
            return text_response.strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"   [Warning] API Rate Limit (429) hit on attempt {attempt}/{max_retries}. Sleeping for {retry_delay}s before retry...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"[API 오류] HTTP {e.code}: {e.read().decode('utf-8')}")
                return None
        except Exception as e:
            print(f"[API 오류] 예외 발생: {e}")
            # wait a bit on other network glitches
            time.sleep(5)
            continue
            
    print("[API 오류] 최대 재시도 횟수를 초과했습니다.")
    return None

# --- AI Analyzer for Forum Post ---
def analyze_post(title, content):
    """
    Analyze sentiment, category, summary and extract Korean keywords.
    """
    prompt = f"""
    Analyze the following MapleStory forum thread (Title and Content).
    Return a JSON object with the following keys:
    - "sentiment": Choose one of: "Positive", "Neutral", "Negative". If it's a bug report, server lag, crash, or complaint, choose "Negative". If it's general praise or thank you, choose "Positive". Otherwise, choose "Neutral".
    - "category": Choose the most appropriate Korean category from: "게임 오류/버그", "서버/접속 장애", "게임 밸런스", "캐시샵/비즈니스 모델", "시스템 개선/건의", "이벤트/프로모션", "기타"
    - "summary": A one-line summary in Korean of the post.
    - "keywords": A JSON array of 3 to 5 core keywords/topics in Korean extracted from the content (e.g., ["큐브", "드롭률", "팅김 현상", "QoL"]). Each keyword should be a concise Korean noun or short phrase (1-3 words).

    Title: {title}
    Content: {content[:4000]}
    """
    
    res_text = call_gemini_api(prompt)
    if not res_text:
        return {
            "sentiment": "Neutral",
            "category": "기타",
            "summary": "AI 분석 호출에 실패했습니다.",
            "keywords": ["오류"]
        }

    try:
        return json.loads(res_text)
    except Exception as e:
        print(f"[파싱 오류] JSON 파싱 실패: {e}. 응답 텍스트: {res_text}")
        return {
            "sentiment": "Neutral",
            "category": "기타",
            "summary": "AI 분석 결과 파싱에 실패했습니다.",
            "keywords": ["오류"]
        }

# --- AI Analyzer for Batch of Forum Posts ---
def analyze_posts_batch(posts):
    """
    Analyze a batch of posts in a single Gemini API call to optimize speed and stay within rate limits.
    """
    if not posts:
        return {}
        
    posts_data_for_ai = []
    for p in posts:
        posts_data_for_ai.append({
            "id": p["id"],
            "title": p["title"],
            "content": p["content"][:2000] # truncate content
        })
        
    prompt = f"""
    Analyze the sentiment, category, summary, and keywords for the following list of MapleStory forum posts.
    Return a JSON object containing a single key "results" which is a JSON array. Each element in the array must correspond to one post and contain the following keys:
    - "id": The string id of the post (must match the input id exactly).
    - "sentiment": Choose one of: "Positive", "Neutral", "Negative". If it's a bug report, server lag, crash, or complaint, choose "Negative". If it's general praise or thank you, choose "Positive". Otherwise, choose "Neutral".
    - "category": Choose the most appropriate Korean category from: "게임 오류/버그", "서버/접속 장애", "게임 밸런스", "캐시샵/비즈니스 모델", "시스템 개선/건의", "이벤트/프로모션", "기타"
    - "summary": A one-line summary in Korean of the post.
    - "keywords": A JSON array of 3 to 5 core keywords/topics in Korean extracted from the content (e.g., ["큐브", "드롭률", "팅김 현상", "QoL"]). Each keyword should be a concise Korean noun or short phrase (1-3 words).

    Posts to analyze:
    {json.dumps(posts_data_for_ai, ensure_ascii=False, indent=2)}
    """
    
    res_text = call_gemini_api(prompt)
    if not res_text:
        return {}
        
    try:
        data = json.loads(res_text)
        results_list = data.get("results", [])
        return {item["id"]: item for item in results_list if "id" in item}
    except Exception as e:
        print(f"[파싱 오류] 일괄 분석 JSON 파싱 실패: {e}. 응답: {res_text[:500] if res_text else ''}")
        return {}

# --- Web Scraper for Suggestions & Feedback ---
def scrape_suggestions_threads(max_pages=1):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    base_url = "https://forums.maplestory.nexon.net"
    threads = []
    
    print(f"[크롤러] 포럼 Suggestions & Feedback 게시글 수집 시작 (총 {max_pages} 페이지)...")
    
    for page in range(1, max_pages + 1):
        if page == 1:
            forum_url = "https://forums.maplestory.nexon.net/categories/suggestions-and-feedback"
        else:
            forum_url = f"https://forums.maplestory.nexon.net/categories/suggestions-and-feedback/p{page}"
            
        print(f" -> {page} 페이지 크롤링 중: {forum_url}")
        
        try:
            req = urllib.request.Request(forum_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read()
        except Exception as e:
            print(f" -> [오류] {page} 페이지 수집 실패: {e}")
            continue
            
        soup = BeautifulSoup(html, "html.parser")
        title_tags = soup.select("a.Title")
        
        page_count = 0
        for tag in title_tags:
            title = tag.get_text().strip()
            href = tag.get("href", "")
            
            if "/discussion/" not in href:
                continue
                
            full_url = urllib.parse.urljoin(base_url, href)
            
            match = re.search(r'/discussion/(\d+)', full_url)
            discussion_id = match.group(1) if match else full_url
            
            # Duplication check
            if any(t["id"] == discussion_id for t in threads):
                continue
                
            threads.append({
                "id": discussion_id,
                "title": title,
                "url": full_url
            })
            page_count += 1
            
        print(f"    * {page} 페이지에서 {page_count}개 스레드 감지 완료.")
        time.sleep(1) # Polite delay between list pages
        
    return threads

def scrape_thread_content(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read()
            
        soup = BeautifulSoup(html, "html.parser")
        target = soup.find(class_=lambda x: x and "MessageList" in x and "Discussion" in x)
        if not target:
            target = soup.select_one(".MessageList.Discussion") or soup.select_one(".MessageList")
            
        if target:
            text = target.get_text(separator="\n").strip()
            # Clean up white spaces
            text = re.sub(r'\n+', '\n', text)
            return text
        else:
            return "본문을 찾을 수 없습니다."
    except Exception as e:
        print(f"    * [오류] 스레드 본문 수집 실패: {e}")
        return f"본문 수집 중 에러 발생: {e}"

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description="MapleStory Forum Dashboard Generator")
    parser.add_argument("--test", action="store_true", help="테스트 모드: 최근 5개 스레드만 분석")
    parser.add_argument("--pages", type=int, default=1, help="크롤링할 페이지 수 (기본: 1)")
    args = parser.parse_args()

    start_time = datetime.now()
    now_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

    print("=========================================================")
    print(" 메이플스토리 포럼 실시간 동향 대시보드 빌더 실행")
    print(f" 시작 시각: {now_str}")
    print("=========================================================")

    if not GEMINI_API_KEY:
        print("[에러] .env 파일에 GEMINI_API_KEY가 존재하지 않거나 빈 값입니다.")
        print("대시보드 생성을 취소합니다.")
        return

    # Step 1: Forum list scraping
    threads = scrape_suggestions_threads(max_pages=args.pages)
    
    if not threads:
        print("[에러] 수집된 스레드가 없습니다.")
        return

    if args.test:
        print(f"[테스트 모드] 최신 5개 스레드만 분석 대상으로 제한합니다 (원래 총 {len(threads)}개).")
        threads = threads[:5]

    total_threads = len(threads)
    
    # Scrape all thread contents first
    print(f"\n[크롤러] {total_threads}개 스레드의 상세 본문 내용 수집 중...")
    scraped_posts = []
    for idx, thread in enumerate(threads, 1):
        print(f" -> [{idx}/{total_threads}] ID: {thread['id']} 본문 수집 중...")
        content = scrape_thread_content(thread['url'])
        scraped_posts.append({
            "id": thread["id"],
            "title": thread["title"],
            "url": thread["url"],
            "content": content
        })
        time.sleep(0.5) # Polite delay

    # Batch analysis in groups of 15 posts (highly robust and safe size for output limits)
    batch_size = 15
    processed_posts = []
    keyword_map = {}
    
    print(f"\n[AI 분석] {total_threads}개 게시글 일괄 분석 가동 (배치 크기: {batch_size})...")
    for i in range(0, len(scraped_posts), batch_size):
        batch = scraped_posts[i:i+batch_size]
        print(f" -> 배치 분석 중: {i+1} ~ {min(i+batch_size, len(scraped_posts))}번째 게시글 (총 {len(batch)}개)...")
        
        batch_results = analyze_posts_batch(batch)
        
        # Merge batch results
        for p in batch:
            analysis = batch_results.get(p["id"], {})
            post_data = {
                "id": p["id"],
                "title": p["title"],
                "url": p["url"],
                "content": p["content"][:1500],
                "category": analysis.get("category", "기타"),
                "sentiment": analysis.get("sentiment", "Neutral"),
                "summary": analysis.get("summary", "요약 정보 없음"),
                "keywords": analysis.get("keywords", [])
            }
            processed_posts.append(post_data)
            
            for kw in post_data["keywords"]:
                kw_clean = kw.strip()
                if kw_clean:
                    keyword_map[kw_clean] = keyword_map.get(kw_clean, 0) + 1

    # Compile word frequencies: [['단어', 빈도], ...]
    word_frequencies = sorted(
        [[k, v] for k, v in keyword_map.items()],
        key=lambda x: x[1],
        reverse=True
    )

    # Step 3: Read dashboard template and replace placeholder
    if not os.path.exists(TEMPLATE_FILE):
        print(f"[에러] 대시보드 템플릿 파일이 없습니다: {TEMPLATE_FILE}")
        return

    print("\n[대시보드 작성] 단일 HTML 파일로 변환 및 컴파일 중...")
    
    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        template_content = f.read()

    dashboard_json = {
        "meta": {
            "run_time": now_str,
            "total_scraped": len(processed_posts)
        },
        "posts": processed_posts,
        "word_frequencies": word_frequencies
    }

    # Replace placeholder with JSON string
    json_str = json.dumps(dashboard_json, ensure_ascii=False, indent=4)
    compiled_html = template_content.replace("/*DATA_PLACEHOLDER*/", json_str)

    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(compiled_html)

    duration = datetime.now() - start_time
    duration_str = str(duration).split('.')[0]
    
    print("=========================================================")
    print(f" [성공] 대시보드가 정상적으로 빌드되었습니다!")
    print(f" - 출력 파일: {OUTPUT_FILE}")
    print(f" - 처리 스레드 수: {len(processed_posts)}")
    print(f" - 소요 시간: {duration_str}")
    print("=========================================================")

if __name__ == "__main__":
    main()
