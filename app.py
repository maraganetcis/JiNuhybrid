# ji_nu_hybrid_improved.py
import streamlit as st
import google.generativeai as genai
import requests
import sqlite3
import hashlib
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import json
import os
import threading

# 검색 라이브러리
try:
    from duckduckgo_search import DDGS
    search_available = True
except ImportError:
    search_available = False

# 로깅
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="JiNu hybrid AI",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# 🚀 하이브리드 AI\n 검색(KJS1) + 코딩(Qwen) + 일반(Gemini)"
    }
)

# CSS: 라이트/다크 모드 및 채팅 버블
BASE_CSS = """
<style>
:root{
  --bg:#f6f7fb;
  --card:#ffffff;
  --muted:#6c757d;
  --accent:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
}
body { background: var(--bg); }
.chat-container { max-width: 1000px; margin: 0 auto; }
.header { text-align:center; font-weight:800; margin-bottom:8px; font-size:2.4rem; background:-webkit-linear-gradient(135deg,#667eea,#764ba2); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.sub { text-align:center; color:var(--muted); margin-bottom:18px; font-size:1rem; }

.chat-bubble { padding: 12px 16px; border-radius: 14px; margin: 8px 0; display:inline-block; max-width:78%; line-height:1.4; }
.user-bubble { background:#e7f0ff; border-radius: 14px 14px 2px 14px; align-self:flex-end; }
.assistant-bubble { background: #fff; border-radius: 14px 14px 14px 2px; box-shadow: 0 2px 8px rgba(16,24,40,0.04); }

.meta-card { margin-top:8px; padding:10px; border-radius:8px; background:var(--card); font-size:0.85rem; border:1px solid rgba(0,0,0,0.04); }
.timestamp { font-size:0.72rem; color:#999; margin-left:8px; }

.sidebar-card { background:var(--card); padding:10px; border-radius:8px; border:1px solid rgba(0,0,0,0.04); margin-bottom:8px; }
.btn-clear { margin-top:8px; }

.dark body, .dark :root { --bg:#0b1020; --card:#081123; --muted:#9aa7c7; }
.dark .chat-bubble.user-bubble { background: rgba(56,102,255,0.12); }
.dark .chat-bubble.assistant-bubble { background: #07122a; color:#e6eef8; box-shadow:none; }
</style>
"""

# 적용
st.markdown(BASE_CSS, unsafe_allow_html=True)

class FreePlanAISystem:
    def __init__(self):
        self.rate_limit_lock = threading.Lock()
        self.setup_api_keys()
        self.setup_database()
        self.initialize_session_state()
        self.setup_rate_limiting()
        logger.info("하이브리드 AI 시스템 초기화 완료")
    
    def setup_api_keys(self):
        """API 키 설정"""
        try:
            # Gemini
            if 'GEMINI_API_KEY' in st.secrets:
                genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
                self.gemini_available = True
            else:
                self.gemini_available = False
            
            # Groq
            self.groq_key = st.secrets.get('GROQ_API_KEY', '')
            self.groq_available = bool(self.groq_key)
                
            self.available_models = []
            if self.gemini_available: self.available_models.append('gemini')
            if self.groq_available: 
                self.available_models.append('qwen')
                self.available_models.append('llama')
                
            logger.info(f"사용 가능한 모델: {self.available_models}")
                
        except Exception as e:
            logger.error(f"API 키 설정 중 오류: {e}")
            st.error("API 키 설정 오류. secrets.toml을 확인하세요.")
    
    def setup_rate_limiting(self):
        """요청 제한 설정 (thread-safe 사용)"""
        self.rate_limits = {
            'gemini': {'count': 0, 'last_reset': time.time(), 'max_per_minute': 15},
            'qwen': {'count': 0, 'last_reset': time.time(), 'max_per_minute': 30},
            'llama': {'count': 0, 'last_reset': time.time(), 'max_per_minute': 30}
        }
    
    def check_rate_limit(self, model: str) -> bool:
        """요청 제한 확인 (스레드 안전)"""
        with self.rate_limit_lock:
            current_time = time.time()
            limit_info = self.rate_limits.get(model, self.rate_limits['gemini'])
            
            if current_time - limit_info['last_reset'] > 60:
                limit_info['count'] = 0
                limit_info['last_reset'] = current_time
            
            if limit_info['count'] >= limit_info['max_per_minute']:
                return False
            
            limit_info['count'] += 1
            return True
    
    def setup_database(self):
        """SQLite 데이터베이스 설정 (확장된 컬럼 포함)"""
        try:
            self.conn = sqlite3.connect('ai_system.db', check_same_thread=False)
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_message TEXT,
                    bot_response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_used TEXT,
                    intent_detected TEXT,
                    processing_time REAL,
                    tokens_used INTEGER,
                    rate_limited BOOLEAN DEFAULT FALSE
                )
            ''')
            self.conn.commit()
        except Exception as e:
            logger.error(f"데이터베이스 설정 오류: {e}")
    
    def initialize_session_state(self):
        """Streamlit 세션 상태 초기화"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'user_id' not in st.session_state:
            st.session_state.user_id = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()
        if 'conversation_count' not in st.session_state:
            st.session_state.conversation_count = 0
        if 'model_usage' not in st.session_state:
            st.session_state.model_usage = {}
        if 'dark_mode' not in st.session_state:
            st.session_state.dark_mode = False
        if 'pending' not in st.session_state:
            st.session_state.pending = False

    # 검색 캐시 적용
    @st.cache_data(ttl=60 * 10, show_spinner=False)
    def perform_web_search(self, query: str) -> str:
        """DuckDuckGo를 이용한 웹 검색 (list 변환 포함)"""
        if not search_available:
            return "검색 라이브러리(duckduckgo-search)가 설치되지 않았습니다."
        
        try:
            # DDGS().text는 generator일 수 있으니 list로 변환
            raw = DDGS().text(query, max_results=6)
            results = list(raw) if raw is not None else []
            if not results:
                return "검색 결과가 없습니다."
            
            search_summary = ""
            for i, r in enumerate(results):
                title = r.get('title') or r.get('heading') or 'No title'
                body = r.get('body') or r.get('snippet') or ''
                href = r.get('href') or r.get('url') or ''
                search_summary += f"{i+1}. {title}\n{body}\nURL: {href}\n\n"
            return search_summary
        except Exception as e:
            logger.exception("검색 중 오류")
            return f"검색 중 오류 발생: {str(e)}"

    def advanced_intent_analysis(self, user_input: str) -> Dict:
        """사용자 의도 분석 (한국어 키워드 중심으로 확장)"""
        intent_keywords = {
            'search': ['검색','찾아줘','최신','뉴스','사건','정리','요약','무엇','무슨','언제','어디','알려줘','조사','출처'],
            'technical': ['코드','프로그래밍','알고리즘','파이썬','에러','api','json','디버그','버그'],
            'mathematical': ['계산','수학','공식','확률','통계','연산'],
            'creative': ['작성','생성','글쓰기','시','소설','아이디어','카피']
        }
        
        intent_scores = {}
        user_lower = user_input.lower()
        
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in user_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # 간단한 길이 기반 복잡도 판단
        word_count = len(user_input.split())
        complexity = 'high' if word_count > 25 else 'medium' if word_count > 7 else 'low'
        
        primary_intent = 'general'
        if intent_scores:
            primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return {
            'primary_intent': primary_intent,
            'complexity': complexity,
            'intent_scores': intent_scores
        }

    def select_optimal_model(self, intent_analysis: Dict) -> Dict:
        """
        모델 선택 전략
        """
        intent_model_mapping = {
            'search': {
                'primary': 'llama', 'backup': 'gemini', 'fallback': 'qwen',
                'reason': '🌐 KJS 1이 검색 결과를 읽고 답변합니다.', 'icon': '🌐'
            },
            'technical': {
                'primary': 'qwen', 'backup': 'llama', 'fallback': 'gemini',
                'reason': '💻 코딩/기술 문제는 Qwen이 우수', 'icon': '💻'
            },
            'mathematical': {
                'primary': 'qwen', 'backup': 'gemini', 'fallback': 'llama',
                'reason': '🧮 수학 연산은 Qwen이 강력', 'icon': '🧮'
            },
            'general': {
                'primary': 'gemini', 'backup': 'llama', 'fallback': 'qwen',
                'reason': '⚡ 일반 대화는 Gemini 사용', 'icon': '⚡'
            }
        }
        
        primary_intent = intent_analysis['primary_intent']
        model_choice = intent_model_mapping.get(primary_intent, intent_model_mapping['general'])
        
        selected_model = None
        for tier in ['primary', 'backup', 'fallback']:
            candidate = model_choice[tier]
            if candidate in self.available_models and self.check_rate_limit(candidate):
                selected_model = candidate
                if tier != 'primary':
                    model_choice['reason'] += f" ({tier} 전환)"
                break
        
        if not selected_model:
            selected_model = 'gemini'
            model_choice['reason'] = "⚠️ 가용량 초과로 기본 모델 사용"
        
        model_choice['selected'] = selected_model
        return model_choice

    def call_gemini_api(self, prompt: str) -> Dict:
        """Gemini 호출 (기본/예외 처리 포함)"""
        if not self.gemini_available: 
            return {'success': False, 'error': 'Gemini 미설정'}
        try:
            start_time = time.time()
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            st.session_state.model_usage['gemini'] = st.session_state.model_usage.get('gemini', 0) + 1
            content = response.text if hasattr(response, 'text') else str(response)
            tokens = len(prompt + content) // 4
            return {
                'success': True, 'content': content, 'model': "Gemini 2.5 Flash",
                'processing_time': time.time() - start_time, 'tokens': tokens
            }
        except Exception as e:
            # quota 같은 에러를 더 명확히 분류
            msg = str(e).lower()
            if 'quota' in msg or 'quota' in getattr(e, 'message', '').lower():
                return {'success': False, 'error': 'Gemini 할당량 초과'}
            logger.exception("Gemini 호출 오류")
            return {'success': False, 'error': str(e)}

    def call_groq_api(self, prompt: str, model_type: str = 'llama') -> Dict:
        """Groq API 호출 (Qwen / Llama)"""
        if not self.groq_available: return {'success': False, 'error': 'Groq 키 미설정'}
        try:
            start_time = time.time()
            
            # 안전한 모델 ID 후보 목록
            if model_type == 'qwen':
                model_id = "qwen-2.5-72b-32k"
                display_name = "Qwen 2.5 (72B)"
            else:
                # Llama 후보 목록 중 사용 가능한 것을 선택(예시)
                LAMMA_CANDIDATES = [
                    "llama-3.3-70b-versatile",
                    "llama-3.3-70b-specdec"
                ]
                model_id = LAMMA_CANDIDATES[0]
                display_name = "Llama 3.3 (70B)"
            
            data = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6
            }
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_key}",
                    "Content-Type": "application/json"
                },
                json=data, timeout=60
            )
            
            if response.status_code == 200:
                key = 'qwen' if model_type == 'qwen' else 'llama'
                st.session_state.model_usage[key] = st.session_state.model_usage.get(key, 0) + 1
                result = response.json()
                content = result['choices'][0]['message']['content']
                tokens = result.get('usage', {}).get('total_tokens', 0)
                return {
                    'success': True,
                    'content': content,
                    'model': display_name,
                    'processing_time': time.time() - start_time,
                    'tokens': tokens
                }
            logger.error(f"Groq 응답 상태: {response.status_code} / {response.text}")
            return {'success': False, 'error': f"Status {response.status_code}"}
        except Exception as e:
            logger.exception("Groq 호출 오류")
            return {'success': False, 'error': str(e)}

    def intelligent_model_orchestration(self, user_input: str) -> Dict:
        """모델 오케스트레이션 실행"""
        intent_analysis = self.advanced_intent_analysis(user_input)
        model_choice = self.select_optimal_model(intent_analysis)
        selected_model = model_choice.get('selected')
        
        final_prompt = user_input
        search_results = None
        if intent_analysis['primary_intent'] == 'search' and search_available:
            # 검색 수행 (캐시 적용)
            search_results = self.perform_web_search(user_input)
            final_prompt = f"""
[Instructions]
User asked: "{user_input}"

Here is the information found on the web:
{search_results}

Please answer the user's question in Korean, basing your answer on the search results above.
- If results conflict, state the disagreement and cite the most relevant points.
- If results are insufficient, say you couldn't find a satisfying answer and then give your best-guess.
- Keep answer concise and provide sources when appropriate.
"""
        response = {'success': False}
        
        # 모델 호출
        if selected_model == 'qwen':
            response = self.call_groq_api(final_prompt, model_type='qwen')
        elif selected_model == 'llama':
            response = self.call_groq_api(final_prompt, model_type='llama')
        elif selected_model == 'gemini':
            response = self.call_gemini_api(final_prompt)
            
        # 백업 로직
        used_model = selected_model
        rate_limited_flag = False
        if not response.get('success'):
            # fallback to gemini if possible
            if selected_model != 'gemini' and self.gemini_available:
                response = self.call_gemini_api(final_prompt)
                if response.get('success'):
                    used_model = 'gemini'
                    model_choice['reason'] += " (⚠️ 백업 사용)"
            else:
                # rate-limited 경고
                rate_limited_flag = True
        
        if response.get('success'):
            return {
                'success': True,
                'content': response['content'],
                'model_name': response['model'],
                'intent_analysis': intent_analysis,
                'model_reason': model_choice['reason'],
                'processing_time': response['processing_time'],
                'tokens_used': response.get('tokens', 0),
                'model_icon': model_choice['icon'],
                'used_model_key': used_model,
                'rate_limited': rate_limited_flag,
                'search_results': search_results
            }
        else:
            return {'success': False, 'error': f"실패: {response.get('error')}"}

    def save_conversation(self, session_id: str, user_message: str, bot_response: str,
                          model_used: str, intent: str, processing_time: float, tokens: int, rate_limited: bool):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO conversations 
                (session_id, user_message, bot_response, model_used, intent_detected, processing_time, tokens_used, rate_limited)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, user_message, bot_response, model_used, intent, processing_time, tokens, int(rate_limited)))
            self.conn.commit()
        except Exception as e:
            logger.exception("DB 저장 실패")

    # UI 구성 함수들
    def display_beautiful_sidebar(self):
        with st.sidebar:
            st.markdown('<div class="header">JiNu AI</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub">검색(KJS), 코딩(Qwen), 일반(Gemini) 완전체</div>', unsafe_allow_html=True)
            
            if not search_available:
                st.warning("⚠️ duckduckgo-search 미설치됨 (검색 불가)")
            
            st.markdown("### 🔧 연결 상태")
            c1, c2 = st.columns(2)
            c1.metric("Gemini", "ON" if self.gemini_available else "OFF")
            c2.metric("Groq", "ON" if self.groq_available else "OFF")
            st.markdown("---")
            
            st.markdown('<div class="sidebar-card"><strong>팀 구성</strong><br>⚡ Gemini 2.5 (General)<br>💻 Qwen 2.5 (Code)<br>🌐 KJS 1 (Search)</div>', unsafe_allow_html=True)
            if st.button("🗑️ 대화 내용 지우기", key="clear_btn"):
                st.session_state.messages = []
            st.checkbox("다크 모드", value=st.session_state.dark_mode, key="dark_mode", on_change=self.toggle_dark)
            st.markdown("---")
            st.markdown("### 📊 사용 통계")
            mu = st.session_state.model_usage
            st.write(mu)
            st.markdown("---")
            self.display_footer()

    def toggle_dark(self):
        st.session_state.dark_mode = not st.session_state.dark_mode
        # body에 dark 클래스 토글 (간단한 방식)
        if st.session_state.dark_mode:
            st.markdown("<script>document.documentElement.classList.add('dark');</script>", unsafe_allow_html=True)
        else:
            st.markdown("<script>document.documentElement.classList.remove('dark');</script>", unsafe_allow_html=True)

    def display_beautiful_chat(self):
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown('<div class="header">JiNu Hybrid AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub">검색(KJS), 코딩(KJS), 일반(Gemini) — 안정화 & UI 개선 버전</div>', unsafe_allow_html=True)
        
        # 메시지 렌더링 (채팅 버블 스타일)
        for msg in st.session_state.messages:
            role = msg.get("role")
            content = msg.get("content")
            ts = msg.get("timestamp", "")
            if role == "user":
                st.markdown(f'<div style="display:flex; justify-content:flex-end;"><div class="chat-bubble user-bubble">{content}<div class="timestamp">{ts}</div></div></div>', unsafe_allow_html=True)
            else:
                # 메타 표시
                meta_html = ""
                if msg.get("metadata"):
                    meta = msg["metadata"]
                    model_icon = meta.get("model_icon", "🤖")
                    model_name = meta.get("model_name", "")
                    reason = meta.get("model_reason", "")
                    proc = meta.get("processing_time", 0.0)
                    intent = meta.get("intent_analysis", {}).get("primary_intent", "")
                    meta_html = f'''
                    <div class="meta-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div><strong>{model_icon} {model_name}</strong> · <span style="color:#6c757d;">{intent}</span></div>
                            <div style="font-size:0.78rem;color:#999;">⏱ {proc:.2f}s</div>
                        </div>
                        <div style="margin-top:6px;color:#666;">{reason}</div>
                    </div>
                    '''
                st.markdown(f'<div style="display:flex; justify-content:flex-start;"><div><div class="chat-bubble assistant-bubble">{content}<div class="timestamp">{ts}</div></div>{meta_html}</div></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # 입력창 처리: 입력 -> 곧바로 처리 (rerun 최소화)
        if not st.session_state.pending:
            prompt = st.chat_input("질문을 입력하세요...")
            if prompt:
                st.session_state.pending = True
                # 바로 사용자 메시지 추가
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": now})
                # 호출 및 응답 처리
                with st.spinner("🤔 하이브리드 AI가 생각하는 중 입니다..."):
                    result = self.intelligent_model_orchestration(prompt)
                    if result.get('success'):
                        assistant_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result['content'],
                            "timestamp": assistant_ts,
                            "metadata": result
                        })
                        # DB 저장 (비동기 X, 바로 저장)
                        try:
                            self.save_conversation(
                                st.session_state.user_id,
                                prompt,
                                result['content'],
                                result.get('used_model_key', result.get('model_name', 'unknown')),
                                result['intent_analysis']['primary_intent'],
                                result['processing_time'],
                                result.get('tokens_used', 0),
                                result.get('rate_limited', False)
                            )
                        except Exception:
                            pass
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"오류 발생: {result.get('error')}",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                st.session_state.pending = False
                # 한 번에 한 상태에서 rerun하지 않고 화면에 반영되게 함

    def display_footer(self):
        st.markdown("""
        <div style="text-align:center; color:#aaa; font-size:0.85rem; margin-top:18px;">
           copyright © 2025 <strong>Synox Studios</strong>. <br>
            <span style="color: #667eea;">Gemini 2.5</span> • 
            <span style="color: #f25c54;">Qwen 2.5</span> •
            <span style="color: #d97757;">Synox Studios KJS 1</span>
        </div>
        """, unsafe_allow_html=True)

def main():
    ai_system = FreePlanAISystem()
    ai_system.display_beautiful_sidebar()
    ai_system.display_beautiful_chat()

if __name__ == "__main__":
    main()
