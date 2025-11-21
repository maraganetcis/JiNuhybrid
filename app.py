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

# ✅ 검색 라이브러리 (설치 필요: pip install duckduckgo-search)
try:
    from duckduckgo_search import DDGS
    search_available = True
except ImportError:
    search_available = False

# ✅ 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ✅ 페이지 설정
st.set_page_config(
    page_title="JiNu hybrid AI",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# 🚀 하이브리드 AI\n 검색(KJS1) + 코딩(Qwen) + 일반(Gemini)"
    }
)

# ✅ CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .free-badge {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: bold;
        margin-left: 0.5rem;
        vertical-align: middle;
    }
    .model-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .intent-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.1rem;
    }
    .complexity-high { background: #ff6b6b; color: white; }
    .complexity-medium { background: #ffd93d; color: black; }
    .complexity-low { background: #6bcf7f; color: white; }
    
    .metadata-box {
        margin-top: 0.8rem; 
        padding: 0.8rem; 
        background: white; 
        border-radius: 10px; 
        border: 1px solid #e0e0e0;
        font-size: 0.85rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        padding: 2rem 0;
        margin-top: 2rem;
        border-top: 1px solid #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

class FreePlanAISystem:
    def __init__(self):
        self.setup_api_keys()
        self.setup_database()
        self.initialize_session_state()
        self.setup_rate_limiting()
        logger.info("하이브리드 AI 시스템 초기화 완료")
    
    def setup_api_keys(self):
        """API 키 설정"""
        try:
            # Google Gemini (기본)
            if 'GEMINI_API_KEY' in st.secrets:
                genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
                self.gemini_available = True
            else:
                self.gemini_available = False
            
            # Groq (Qwen & Llama)
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
        """요청 제한 설정"""
        self.rate_limits = {
            'gemini': {'count': 0, 'last_reset': time.time(), 'max_per_minute': 15},
            'qwen': {'count': 0, 'last_reset': time.time(), 'max_per_minute': 30},
            'llama': {'count': 0, 'last_reset': time.time(), 'max_per_minute': 30}
        }
    
    def check_rate_limit(self, model: str) -> bool:
        """요청 제한 확인"""
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
        """SQLite 데이터베이스 설정"""
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

    # ✅ [기능 추가] 웹 검색 함수
    def perform_web_search(self, query: str) -> str:
        """DuckDuckGo를 이용한 웹 검색"""
        if not search_available:
            return "검색 라이브러리(duckduckgo-search)가 설치되지 않았습니다."
        
        try:
            results = DDGS().text(query, max_results=4)
            if not results:
                return "검색 결과가 없습니다."
            
            search_summary = "Web Search Results:\n\n"
            for i, r in enumerate(results):
                search_summary += f"{i+1}. {r['title']}: {r['body']}\nURL: {r['href']}\n\n"
            return search_summary
        except Exception as e:
            return f"검색 중 오류 발생: {str(e)}"

    def advanced_intent_analysis(self, user_input: str) -> Dict:
        """사용자 의도 분석 (검색 의도 추가)"""
        intent_keywords = {
            'search': ['검색', '찾아줘', '누구', '최신', '날씨', '뉴스', '사건', '알려줘', '조사', '언제', '어디'],
            'technical': ['코드', '프로그래밍', '알고리즘', '파이썬', '에러', 'api', 'json'],
            'mathematical': ['계산', '수학', '공식', '확률'],
            'creative': ['작성', '생성', '글쓰기', '시', '소설'],
        }
        
        intent_scores = {}
        user_lower = user_input.lower()
        
        for intent, keywords in intent_keywords.items():
            score = sum(10 for keyword in keywords if keyword in user_lower)
            if score > 0:
                intent_scores[intent] = score
        
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
        [전략]
        1. 검색(Search): DuckDuckGo + Llama 3.3 (RAG)
        2. 전문가(Logic/Code): Qwen 2.5
        3. 기본(General): Gemini 2.5
        """
        intent_model_mapping = {
            'search': {
                'primary': 'llama', 'backup': 'gemini', 'fallback': 'qwen',
                'reason': '🌐 KJS 1이 검색 결과를 읽고 답변합니다.', 'icon': '🌐'
            },
            'technical': {
                'primary': 'qwen', 'backup': 'llama', 'fallback': 'gemini',
                'reason': '💻 코딩/기술 문제는 Qwen 2.5가 해결', 'icon': '💻'
            },
            'mathematical': {
                'primary': 'qwen', 'backup': 'gemini', 'fallback': 'llama',
                'reason': '🧮 수학 연산은 Qwen 2.5가 강력함', 'icon': '🧮'
            },
            'general': {
                'primary': 'gemini', 'backup': 'llama', 'fallback': 'qwen',
                'reason': '⚡ 일반 대화는 기본 모델(Gemini 2.5) 사용', 'icon': '⚡'
            }
        }
        
        primary_intent = intent_analysis['primary_intent']
        # 창작(creative) 등은 general 맵핑이나 별도 처리가 없다면 general로 갈 수 있음.
        # 여기서는 창작도 Gemini가 잘하므로 General로 처리됨.
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
        """Gemini API"""
        if not self.gemini_available: return {'success': False}
        try:
            start_time = time.time()
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            st.session_state.model_usage['gemini'] = st.session_state.model_usage.get('gemini', 0) + 1
            return {
                'success': True, 'content': response.text, 'model': "Gemini 2.5 Flash",
                'processing_time': time.time() - start_time, 'tokens': len(prompt + response.text) // 4
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def call_groq_api(self, prompt: str, model_type: str = 'llama') -> Dict:
        """Groq API (Qwen & Llama)"""
        if not self.groq_available: return {'success': False}
        try:
            start_time = time.time()
            
            # 모델 선택
            if model_type == 'qwen':
                model_id = "qwen-2.5-72b-32k"
                display_name = "Qwen 2.5 (72B)"
            else:
                model_id = "llama-3.3-70b-versatile" # 검색 결과 처리에 매우 강함
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
                return {
                    'success': True,
                    'content': result['choices'][0]['message']['content'],
                    'model': display_name,
                    'processing_time': time.time() - start_time,
                    'tokens': result.get('usage', {}).get('total_tokens', 0)
                }
            return {'success': False, 'error': f"Status {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def intelligent_model_orchestration(self, user_input: str) -> Dict:
        """모델 오케스트레이션 실행"""
        intent_analysis = self.advanced_intent_analysis(user_input)
        model_choice = self.select_optimal_model(intent_analysis)
        selected_model = model_choice.get('selected')
        
        # ✅ [핵심] 검색 의도인 경우: 검색 실행 -> 결과를 프롬프트에 추가
        final_prompt = user_input
        if intent_analysis['primary_intent'] == 'search' and search_available:
            with st.spinner("🌐 인터넷 검색 중..."):
                search_results = self.perform_web_search(user_input)
                final_prompt = f"""
                [Instructions]
                User asked: "{user_input}"
                
                Here is the information found on the web:
                {search_results}
                
                Please answer the user's question in Korean, basing your answer on the search results above.
                If the search results are not relevant, use your own knowledge.
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
        if not response.get('success'):
            if selected_model != 'gemini' and self.gemini_available:
                 response = self.call_gemini_api(final_prompt)
                 if response.get('success'):
                     selected_model = 'gemini'
                     model_choice['reason'] += " (⚠️ 백업 사용)"

        if response.get('success'):
            return {
                'success': True,
                'content': response['content'],
                'model_name': response['model'],
                'intent_analysis': intent_analysis,
                'model_reason': model_choice['reason'],
                'processing_time': response['processing_time'],
                'tokens_used': response.get('tokens', 0),
                'model_icon': model_choice['icon']
            }
        else:
            return {'success': False, 'error': f"실패: {response.get('error')}"}

    def display_footer(self):
        st.markdown("""
        <div class="footer">
           copyright © 2025 <strong>Synox Studios</strong>. <br>
            <span style="color: #667eea;">Gemini 2.5</span> (Basic) • 
            <span style="color: #f25c54;">Qwen 2.5</span> (Code) •
            <span style="color: #d97757;">Synox Studios KJS 1</span> (Search)
        </div>
        """, unsafe_allow_html=True)

    def display_beautiful_sidebar(self):
        with st.sidebar:
            st.markdown('<div class="main-header">JiNu AI</div>', unsafe_allow_html=True)
            
            if not search_available:
                st.warning("⚠️ duckduckgo-search 미설치됨 (검색 불가)")
            
            st.markdown("### 🔧 연결 상태")
            c1, c2 = st.columns(2)
            c1.metric("Gemini", "ON" if self.gemini_available else "OFF")
            c2.metric("Groq", "ON" if self.groq_available else "OFF")
            
            st.markdown("---")
            st.markdown("### 🏆 팀 구성")
            team_specs = [
                {"icon": "⚡", "name": "Gemini 2.5", "desc": "기본: 일반/창작", "type": "MAIN"},
                {"icon": "💻", "name": "Qwen 2.5", "desc": "전문: 코딩/수학", "type": "FREE"},
                {"icon": "🌐", "name": "KJS 1", "desc": "정보: 웹검색/요약", "type": "FREE"}
            ]
            for spec in team_specs:
                st.markdown(f"""
                <div class="model-card">
                    <div style="display: flex; justify-content: space-between;">
                        <div>{spec['icon']} <strong>{spec['name']}</strong></div>
                        <span class="free-badge">{spec['type']}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #666; margin-top: 4px;">{spec['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("🗑️ 대화 내용 지우기", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            self.display_footer()

    def display_beautiful_chat(self):
        st.markdown('<div class="main-header">JiNu Hybrid AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">검색(KJS), 코딩(KJS), 일반(Gemini) 완전체</div>', unsafe_allow_html=True)
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("role") == "assistant" and "metadata" in msg:
                    meta = msg["metadata"]
                    try:
                        complexity_class = f"complexity-{meta['intent_analysis']['complexity']}"
                        intent_text = meta['intent_analysis']['primary_intent']
                        meta_html = f"""
                        <div class="metadata-box">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="color: #667eea; font-weight: bold;">{meta.get('model_icon', '🤖')} {meta['model_name']}</span>
                                <span class="intent-badge {complexity_class}">{intent_text}</span>
                            </div>
                            <hr style="margin: 0.5rem 0; opacity: 0.2;">
                            <div style="color: #666;">💡 {meta['model_reason']}</div>
                            <div style="text-align: right; font-size: 0.7rem; color: #999; margin-top: 0.3rem;">⏱️ {meta['processing_time']:.2f}s</div>
                        </div>
                        """
                        st.markdown(meta_html, unsafe_allow_html=True)
                    except KeyError: pass
        
        if st.session_state.messages:
            self.display_footer()

        if prompt := st.chat_input("질문을 입력하세요..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            with st.spinner("🤔 하이브리드 AI가 생각하는 중 입니다..."):
                result = self.intelligent_model_orchestration(st.session_state.messages[-1]["content"])
                if result['success']:
                    st.session_state.messages.append({
                        "role": "assistant", "content": result['content'], "metadata": result
                    })
                    try:
                        cursor = self.conn.cursor()
                        cursor.execute('INSERT INTO conversations (session_id, user_message, bot_response) VALUES (?, ?, ?)', 
                                     (st.session_state.user_id, st.session_state.messages[-2]["content"], result['content']))
                        self.conn.commit()
                    except: pass
                    st.rerun()
                else:
                    st.error(f"오류: {result.get('error')}")

def main():
    ai_system = FreePlanAISystem()
    ai_system.display_beautiful_sidebar()
    ai_system.display_beautiful_chat()

if __name__ == "__main__":
    main()
