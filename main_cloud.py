import os
import json
import time
import re
import warnings
import urllib.parse
from datetime import datetime, timezone, timedelta
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
except ImportError:
    HAS_CLOUDSCRAPER = False

load_dotenv()

BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
CHAT_ID = os.getenv("TG_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DATA_DIR = "data"
SEEN_FILE = os.path.join(DATA_DIR, "seen_posts.json")
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize empty JSON array if file does not exist
if not os.path.exists(SEEN_FILE):
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

MAX_POST_AGE_DAYS = 60
PAGES_PER_CHANNEL = 5

# Direct Cloud Session (No Proxy Needed on GitHub Actions)
if HAS_CLOUDSCRAPER:
    session = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'linux', 'mobile': False})
else:
    session = requests.Session()

session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,fa;q=0.8",
})

from google import genai
from google.genai import types

ai_client = genai.Client(api_key=GEMINI_API_KEY)

CHANNELS_TO_SCRAPE = [
    "expertapply", "ApplyIR2UK", "applyforfree", "pargarwiki",
    "applyclub", "ApplyDaily", "computer_phd_apply", "EuropeanPhD"
]

BROAD_PATTERNS = [
    r"uwb", r"ultra[- ]wideband", r"indoor", r"localiz", r"position", r"sens", r"radar",
    r"csi\b", r"cir\b", r"ranging", r"tof\b", r"tdoa\b", r"nlos", r"wireless", r"rf\b",
    r"fingerprint", r"channel impulse", r"channel state", r"signal processing", r"telecom",
    r"machine learning", r"deep learning", r"transformer", r"neural", r"transfer learning",
    r"domain adaptation", r"tracking", r"navigation", r"sensor fusion", r"slam",
    r"autonomous", r"odometry", r"imu\b", r"fpga", r"sdr", r"embedded", r"edge ai",
    r"tinyml", r"iot\b", r"vhdl", r"verilog", r"daq\b", r"data acquisition",
    r"موقعیت", r"مکان[- ]?یابی", r"سیگنال", r"مخابرات", r"یادگیری", r"هوش مصنوعی",
    r"نهفته", r"سنسور", r"رادار", r"دکتری", r"فاند", r"بورسیه", r"پوزیشن"
]

EXCLUDE_PATTERNS = [
    r"ویزای همسر", r"ویزای کاری", r"تعیین وقت سفارت", r"کلاس زبان", r"آموزش آیلتس",
    r"ielts class", r"immigration lawyer"
]

ACADEMIC_SEARCH_QUERIES = [
    "UWB localization", "indoor positioning", "wireless sensing",
    "channel state information", "sensor fusion localization",
    "FPGA signal processing", "edge AI sensing", "radar positioning"
]

SYSTEM_PROMPT = """You are an expert academic evaluator. Assess if a PhD vacancy post matches the candidate's research profile.

Candidate Profile:
- Tier 1 (Direct Match): UWB, Ultra-Wideband, indoor positioning/localization (موقعیت‌یابی داخل ساختمان), wireless sensing, RF sensing, CIR, CSI, ToF, TDoA, NLOS conditions, fingerprinting-based positioning. Combinations of localization with ML/Deep Learning.
- Tier 2 (Adjacent): Robust wireless localization, wireless communications, signal processing (پردازش سیگنال), domain adaptation, self-supervised learning, Transformers, uncertainty estimation, GNNs, sensor fusion (UWB+IMU/Vision), Autonomous robot localization/SLAM, medical signal processing.
- Tier 3 (Electronics/Hardware): FPGA acceleration, SDR, embedded AI, Edge AI / TinyML, low-power digital systems design, DAQ (Data Acquisition), IoT sensor networks.
- Disqualifiers (REJECT): Pure mechanical robotics (robotic arms manipulation), prosthetics, medical wet-lab biology, pure chemistry, theoretical math without RF/sensing/signal focus, visa advertisements, language courses. (Note: Robot indoor localization/SLAM IS acceptable).

Respond ONLY in valid JSON:
{
  "is_relevant": true,
  "tier": 1,
  "confidence_score": 9,
  "title": "Short title",
  "key_topics": ["topic1", "topic2"],
  "reason": "1-line explanation of fit"
}"""

def load_seen():
    if os.path.exists(SEEN_FILE):
        try:
            with open(SEEN_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()

def mark_as_seen(uid: str, seen: set):
    seen.add(uid)
    try:
        with open(SEEN_FILE, "w", encoding="utf-8") as f:
            json.dump(list(seen), f)
    except Exception:
        pass

def fast_prefilter(text: str) -> bool:
    lower_t = text.lower()
    if any(re.search(pat, lower_t) for pat in EXCLUDE_PATTERNS):
        return False
    return any(re.search(pat, lower_t) for pat in BROAD_PATTERNS)

def is_recent_date(dt_obj) -> bool:
    if not dt_obj:
        return True
    try:
        now = datetime.now(timezone.utc)
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        return (now - dt_obj) <= timedelta(days=MAX_POST_AGE_DAYS)
    except Exception:
        return True

def parse_iso_time(time_str: str):
    try:
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except Exception:
        return None

def extract_clean_json(raw_text: str) -> dict:
    try:
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            return json.loads(raw_text[start_idx:end_idx + 1])
    except Exception:
        pass
    return {"is_relevant": False, "reason": "Failed to parse JSON output"}

def evaluate_with_gemini(text: str) -> dict:
    # Use the active models matching your API project
    models_to_try = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
    
    for attempt in range(3):
        for model_name in models_to_try:
            try:
                response = ai_client.models.generate_content(
                    model=model_name,
                    contents=f"Evaluate this PhD post:\n\n{text[:2000]}",
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                        safety_settings=[
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                        ]
                    )
                )
                time.sleep(4.2)  # Maintain 15 RPM rate pacing
                
                if not response.candidates or not response.candidates[0].content.parts:
                    return {"is_relevant": False, "reason": "Blocked by Gemini Safety Filters"}
                    
                return extract_clean_json(response.text)

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = 25
                    print(f"    [!] Quota rate limit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    break
                else:
                    print(f"    [-] LLM call error ({model_name}): {err_str[:120]}")
                    continue

    return {"is_relevant": False, "reason": "API error after retries"}

def send_alert(url: str, analysis: dict, snippet: str):
    if not isinstance(analysis, dict):
        return
    tier_badges = {1: "🔥 Tier 1", 2: "⚡ Tier 2", 3: "🛠 Tier 3"}
    tier_val = analysis.get("tier", 2)
    label = tier_badges.get(tier_val, "Relevant Opportunity")
    
    score = analysis.get("confidence_score", "N/A")
    title = analysis.get("title", "PhD Opportunity")
    topics_list = analysis.get("key_topics", [])
    topics = ", ".join(topics_list) if isinstance(topics_list, list) else str(topics_list)
    reason = analysis.get("reason", "Matches research profile.")

    msg = (
        f"🎯 *New PhD Match!*\n"
        f"*Category:* {label}\n"
        f"*Score:* {score}/10\n"
        f"*Title:* {title}\n"
        f"*Topics:* `{topics}`\n\n"
        f"💡 *Reason:* {reason}\n\n"
        f"🔗 [Open Vacancy / Post]({url})\n"
        f"────────────────────\n"
        f"📝 *Preview:*\n{snippet[:250]}..."
    )
    
    endpoint = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown",
        "disable_web_page_preview": False
    }
    try:
        session.post(endpoint, json=payload, timeout=20)
    except Exception as e:
        print(f"[-] Alert sending error: {e}")

def scrape_telegram_channel_deep(channel: str, seen: set):
    print(f"\n📡 Scanning @{channel}...")
    current_url = f"https://t.me/s/{channel}"
    
    for _ in range(PAGES_PER_CHANNEL):
        try:
            r = session.get(current_url, timeout=25)
            if r.status_code != 200:
                break
            soup = BeautifulSoup(r.text, "html.parser")
            messages = soup.find_all("div", class_="tgme_widget_message")
            if not messages:
                break
            
            oldest_id_on_page = None
            for msg in reversed(messages):
                msg_id = msg.get("data-post")
                if not msg_id:
                    continue
                
                raw_num = msg_id.split("/")[-1]
                if raw_num.isdigit():
                    oldest_id_on_page = int(raw_num) if oldest_id_on_page is None else min(oldest_id_on_page, int(raw_num))
                
                if msg_id in seen:
                    continue
                
                time_tag = msg.find("time")
                if time_tag and time_tag.get("datetime"):
                    dt = parse_iso_time(time_tag.get("datetime"))
                    if dt and not is_recent_date(dt):
                        mark_as_seen(msg_id, seen)
                        continue
                        
                text_elem = msg.find("div", class_="tgme_widget_message_text")
                if not text_elem:
                    mark_as_seen(msg_id, seen)
                    continue
                    
                text = text_elem.get_text(separator="\n").strip()
                if len(text) < 40 or not fast_prefilter(text):
                    mark_as_seen(msg_id, seen)
                    continue
                    
                print(f"[+] Evaluating Telegram post: {msg_id}")
                analysis = evaluate_with_gemini(text)
                
                if analysis.get("is_relevant"):
                    send_alert(f"https://t.me/{msg_id}", analysis, text)
                    print(f"    [✓] MATCH CONFIRMED (Tier {analysis.get('tier')} - Score {analysis.get('confidence_score')}/10) -> Alert Sent!")
                else:
                    print(f"    [-] Skipped: {analysis.get('reason', 'Filtered by LLM')}")
                
                mark_as_seen(msg_id, seen)
                
            if oldest_id_on_page:
                current_url = f"https://t.me/s/{channel}?before={oldest_id_on_page}"
            else:
                break
        except Exception as e:
            print(f"[-] @{channel} issue: {e}")
            break

def scrape_findaphd_direct(seen: set):
    print("\n🌍 Scanning FindAPhD.com directly...")
    for kw in ACADEMIC_SEARCH_QUERIES:
        try:
            search_url = f"https://www.findaphd.com/phds/?Keywords={urllib.parse.quote_plus(kw)}"
            resp = session.get(search_url, timeout=25)
            if resp.status_code != 200:
                continue
            
            soup = BeautifulSoup(resp.text, "html.parser")
            cards = soup.find_all("div", class_="phd-result-row") or soup.find_all("div", class_="w-100")
            valid_cards = [c for c in cards if c.find("a", class_="apply-link") or c.find("h3")]
            
            for card in valid_cards[:6]:
                title_elem = card.find("a", class_="apply-link") or card.find("h3") or card.find("a")
                if not title_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                link = title_elem.get("href", "")
                if link and not link.startswith("http"):
                    link = f"https://www.findaphd.com{link}"
                
                desc_elem = card.find("div", class_="desc") or card.find("div", class_="phd-result-row__desc")
                desc = desc_elem.get_text(strip=True) if desc_elem else ""
                
                full_text = f"{title}\n{desc}"
                uid = link or title
                
                if uid in seen or len(full_text) < 30 or not fast_prefilter(full_text):
                    mark_as_seen(uid, seen)
                    continue

                print(f"[+] Evaluating FindAPhD vacancy: {title[:55]}...")
                analysis = evaluate_with_gemini(full_text)
                if analysis.get("is_relevant"):
                    send_alert(link, analysis, full_text)
                    print(f"    [✓] MATCH CONFIRMED (Tier {analysis.get('tier')}) -> Alert Sent!")
                else:
                    print(f"    [-] Skipped: {analysis.get('reason', 'Filtered by LLM')}")
                    
                mark_as_seen(uid, seen)
        except Exception:
            pass

def scrape_euraxess_direct(seen: set):
    print("\n🇪🇺 Scanning EURAXESS (EU MSCA & Funded Positions)...")
    for kw in ["localization", "sensing", "ultra-wideband", "FPGA"]:
        try:
            search_url = f"https://euraxess.ec.europa.eu/jobs/search?keywords={urllib.parse.quote_plus(kw)}"
            resp = session.get(search_url, timeout=25)
            if resp.status_code != 200:
                continue
            
            soup = BeautifulSoup(resp.text, "html.parser")
            job_links = soup.find_all("a", href=re.compile(r"^/jobs/\d+"))
            
            unique_jobs = {}
            for a in job_links:
                href = a.get("href")
                title = a.get_text(strip=True)
                if len(title) > 15 and href not in unique_jobs:
                    unique_jobs[href] = title
            
            for href, title in list(unique_jobs.items())[:6]:
                link = f"https://euraxess.ec.europa.eu{href}"
                if link in seen:
                    continue
                
                job_desc = ""
                try:
                    job_resp = session.get(link, timeout=15)
                    if job_resp.status_code == 200:
                        jsoup = BeautifulSoup(job_resp.text, "html.parser")
                        desc_div = jsoup.find("div", class_="node__content") or jsoup.find("main")
                        if desc_div:
                            job_desc = desc_div.get_text(separator="\n", strip=True)
                except Exception:
                    pass
                    
                full_text = f"{title}\n{job_desc}"[:2000]
                
                if not fast_prefilter(full_text):
                    mark_as_seen(link, seen)
                    continue

                print(f"[+] Evaluating EURAXESS vacancy: {title[:55]}...")
                analysis = evaluate_with_gemini(full_text)
                if analysis.get("is_relevant"):
                    send_alert(link, analysis, full_text)
                    print(f"    [✓] MATCH CONFIRMED (Tier {analysis.get('tier')}) -> Alert Sent!")
                else:
                    print(f"    [-] Skipped: {analysis.get('reason', 'Filtered by LLM')}")
                    
                mark_as_seen(link, seen)
        except Exception:
            pass

def main():
    print("🚀 Running PhD Finder Agent on Cloud...")
    seen = load_seen()
    print(f"📂 Cached database has {len(seen)} items.")
    
    for ch in CHANNELS_TO_SCRAPE:
        scrape_telegram_channel_deep(ch, seen)
        
    scrape_findaphd_direct(seen)
    scrape_euraxess_direct(seen)
    
    print("✅ Run complete. Exiting cleanly.")

if __name__ == "__main__":
    main()
