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
        'Get Help': 'https://github.com/your-repo',
        'About': "# 🚀 하이브리드 AI\n 최적의 AI 모델 자동 선택 시스템"
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
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        border-bottom-right-radius: 5px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .assistant-message {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 15px;
        border-bottom-left-radius: 5px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        text-align: center;
    }
    .rate-limit-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    .metadata-box {
        margin-top: 1rem; 
        padding: 0.8rem; 
        background: white; 
        border-radius: 10px; 
        border: 1px solid #e0e0e0;
        font-size: 0.85rem;
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
            # Google Gemini
            if 'GEMINI_API_KEY' in st.secrets:
                genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
                self.gemini_available = True
                logger.info("Gemini API 설정 완료")
            else:
                self.gemini_available = False # 수정: 오타 수정 (=I False -> = False)
                logger.warning("Gemini API 키 없음")
            
            # OpenRouter
            self.openrouter_key = st.secrets.get('OPENROUTER_API_KEY', '')
            self.openrouter_available = bool(self.openrouter_key)
            
            # DeepSeek
            self.deepseek_key = st.secrets.get('DEEPSEEK_API_KEY', '')
            self.deepseek_available = bool(self.deepseek_key)
                
            # 사용 가능한 모델 목록
            self.available_models = []
            if self.gemini_available: self.available_models.append('gemini')
            if self.openrouter_available: self.available_models.append('claude')
            if self.deepseek_available: self.available_models.append('deepseek')
                
            logger.info(f"사용 가능한 모델: {self.available_models}")
                
        except Exception as e:
            logger.error(f"API 키 설정 중 오류: {e}")
            st.error("API 키 설정 중 오류가 발생했습니다.")
    
    def setup_rate_limiting(self):
        """요청 제한 설정"""
        self.rate_limits = {
            'gemini': {'count': 0, 'last_reset': time.time(), 'max_per_minute': 15},
            'claude': {'count': 0, 'last_reset': time.time(), 'max_per_minute': 10},
            'deepseek': {'count': 0, 'last_reset': time.time(), 'max_per_minute': 20}
        }
    
    def check_rate_limit(self, model: str) -> bool:
        """요청 제한 확인"""
        current_time = time.time()
        limit_info = self.rate_limits[model]
        
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
        if 'rate_limit_hits' not in st.session_state:
            st.session_state.rate_limit_hits = 0

    def advanced_intent_analysis(self, user_input: str) -> Dict:
        """사용자 의도 분석"""
        intent_keywords = {
            'complex_reasoning': ['논리', '추론', '분석', '비교', '평가', '비판', '이유', '근거', '복잡', '심층'],
            'technical': ['코드', '프로그래밍', '알고리즘', '파이썬', '자바', '함수', '에러', '디버깅', 'api', 'json'],
            'creative': ['작성', '생성', '글쓰기', '시', '소설', '아이디어', '기획', '창작'],
            'mathematical': ['계산', '수학', '공식', '확률', '통계', '수식'],
            'research': ['연구', '논문', '이론', '역사', '과학', '조사', '데이터'],
        }
        
        intent_scores = {}
        user_lower = user_input.lower()
        
        for intent, keywords in intent_keywords.items():
            score = sum(10 for keyword in keywords if keyword in user_lower)
            if score > 0:
                intent_scores[intent] = score
        
        word_count = len(user_input.split())
        if word_count > 25: complexity = 'high'
        elif word_count > 7: complexity = 'medium'
        else: complexity = 'low'
        
        primary_intent = 'general'
        if intent_scores:
            primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return {
            'primary_intent': primary_intent,
            'complexity': complexity,
            'intent_scores': intent_scores
        }

    def select_optimal_model(self, intent_analysis: Dict) -> Dict:
        """의도에 따른 최적 모델 선택"""
        intent_model_mapping = {
            'complex_reasoning': {
                'primary': 'claude', 'backup': 'deepseek', 'fallback': 'gemini',
                'reason': '🧠 복잡한 논리/추론에는 Claude 3.5가 우수', 'icon': '🧠'
            },
            'technical': {
                'primary': 'deepseek', 'backup': 'gemini', 'fallback': 'claude',
                'reason': '💻 코드/기술 문제에는 DeepSeek V3가 최적화', 'icon': '💻'
            },
            'mathematical': {
                'primary': 'deepseek', 'backup': 'gemini', 'fallback': 'claude',
                'reason': '🧮 수학적 연산에는 DeepSeek가 강력', 'icon': '🧮'
            },
            'creative': {
                'primary': 'claude', 'backup': 'gemini', 'fallback': 'deepseek',
                'reason': '🎨 창의적 작문은 Claude가 뛰어남', 'icon': '🎨'
            },
            'general': {
                'primary': 'gemini', 'backup': 'deepseek', 'fallback': 'claude',
                'reason': '⚡ 일반 질문에는 빠르고 경제적인 Gemini', 'icon': '⚡'
            }
        }
        
        primary_intent = intent_analysis['primary_intent']
        model_choice = intent_model_mapping.get(primary_intent, intent_model_mapping['general'])
        
        # 모델 가용성 및 제한 체크 로직
        selected_model = None
        
        for tier in ['primary', 'backup', 'fallback']:
            candidate = model_choice[tier]
            if candidate in self.available_models and self.check_rate_limit(candidate):
                selected_model = candidate
                if tier != 'primary':
                    model_choice['reason'] += f" ({tier} 모델 사용)"
                break
        
        if not selected_model:
            for model in self.available_models:
                if self.check_rate_limit(model):
                    selected_model = model
                    model_choice['reason'] = f"⚠️ 가용 모델 제한으로 {model} 사용"
                    break
        
        model_choice['selected'] = selected_model
        return model_choice

    def call_gemini_api(self, prompt: str) -> Dict:
        """Gemini API 호출"""
        if not self.gemini_available: return {'success': False}
        try:
            start_time = time.time()
            # 수정: 1.5-flash -> 2.5-flash (사용가능한 api로)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            
            st.session_state.model_usage['gemini'] = st.session_state.model_usage.get('gemini', 0) + 1
            return {
                'success': True,
                'content': response.text,
                'model': "Google Gemini Flash",
                'processing_time': time.time() - start_time,
                'tokens': len(prompt + response.text) // 4
            }
        except Exception as e:
            logger.error(f"Gemini API 오류: {e}")
            return {'success': False, 'error': str(e)}

    def call_openrouter_api(self, prompt: str) -> Dict:
        """OpenRouter API 호출"""
        if not self.openrouter_available: return {'success': False}
        try:
            start_time = time.time()
            data = {
                "model": "anthropic/claude-3.5-sonnet",
                "messages": [{"role": "user", "content": prompt}],
            }
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.openrouter_key}"},
                json=data, timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.model_usage['claude'] = st.session_state.model_usage.get('claude', 0) + 1
                return {
                    'success': True,
                    'content': result['choices'][0]['message']['content'],
                    'model': "Claude 3.5 Sonnet",
                    'processing_time': time.time() - start_time,
                    'tokens': result.get('usage', {}).get('total_tokens', 0)
                }
            return {'success': False, 'error': f"Status {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def call_deepseek_api(self, prompt: str) -> Dict:
        """DeepSeek API 호출"""
        if not self.deepseek_available: return {'success': False}
        try:
            start_time = time.time()
            data = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
            }
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {self.deepseek_key}"},
                json=data, timeout=60
            )
            
            if response.status_code == 200:
                st.session_state.model_usage['deepseek'] = st.session_state.model_usage.get('deepseek', 0) + 1
                result = response.json()
                return {
                    'success': True,
                    'content': result['choices'][0]['message']['content'],
                    'model': "DeepSeek V3",
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
        selected_model = model_choice['selected']
        
        response = {'success': False}
        
        if selected_model == 'claude':
            response = self.call_openrouter_api(user_input)
        elif selected_model == 'deepseek':
            response = self.call_deepseek_api(user_input)
        elif selected_model == 'gemini':
            response = self.call_gemini_api(user_input)
            
        # 실패 시 백업 시도 (간소화된 로직)
        if not response.get('success') and selected_model != 'gemini' and self.gemini_available:
             response = self.call_gemini_api(user_input)
             selected_model = 'gemini'
             model_choice['reason'] += " (오류로 인해 Gemini 백업 사용)"

        if response.get('success'):
            return {
                'success': True,
                'content': response['content'],
                'model_name': response['model'],
                'intent_analysis': intent_analysis,
                'model_reason': model_choice['reason'],
                'processing_time': response['processing_time'],
                'tokens_used': response['tokens'],
                'model_icon': model_choice['icon']
            }
        else:
            return {'success': False, 'error': "모든 모델 호출 실패"}

    def display_beautiful_sidebar(self):
        """사이드바 UI"""
        with st.sidebar:
            st.markdown('<div class="main-header">💠JiNu AI</div>', unsafe_allow_html=True)
            st.markdown('<div style="text-align: center; margin-bottom: 1rem;"><span class="free-badge">HYBRID ENGINE</span></div>', unsafe_allow_html=True)
            
            st.markdown("### 🔧 연결 상태")
            c1, c2, c3 = st.columns(3)
            c1.metric("Gemini", "ON" if self.gemini_available else "OFF")
            c2.metric("Claude", "ON" if self.openrouter_available else "OFF")
            c3.metric("DeepSeek", "ON" if self.deepseek_available else "OFF")
            
            st.markdown("---")
            st.markdown("### 📈 사용 통계")
            st.markdown(f"""
            <div class="stats-card">
                <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{st.session_state.conversation_count}</div>
                <div style="color: #6c757d;">총 대화 수</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.model_usage:
                st.markdown("#### 모델별 사용량")
                for m, c in st.session_state.model_usage.items():
                    st.caption(f"{m.title()}: {c}회")

            st.markdown("---")
            st.markdown("### 🏆 모델 라인업")
            
            # 수정: type 키 추가하여 KeyError 방지
            free_model_specs = [
                {"icon": "🧠", "name": "Claude 3.5", "desc": "논리, 작문", "type": "CREDIT"},
                {"icon": "⚡", "name": "Gemini Flash", "desc": "빠른 응답", "type": "FREE"}, 
                {"icon": "💻", "name": "DeepSeek V3", "desc": "코딩, 수학", "type": "FREE"}
            ]
            
            for spec in free_model_specs:
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
                st.session_state.conversation_count = 0
                st.rerun()

    def display_beautiful_chat(self):
        """채팅 UI"""
        st.markdown('<div class="main-header">💠 JiNu Hybrid AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">질문의 의도를 파악하여 최적의 모델이 자동으로 답변합니다.</div>', unsafe_allow_html=True)
        
        # 대화 기록 표시
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end;">
                    <div class="user-message">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                meta = msg.get('metadata', {})
                meta_html = ""
                if meta:
                    meta_html = f"""
                    <div class="metadata-box">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #667eea; font-weight: bold;">{meta.get('model_icon', '🤖')} {meta['model_name']}</span>
                            <span class="intent-badge complexity-{meta['intent_analysis']['complexity']}">{meta['intent_analysis']['primary_intent']}</span>
                        </div>
                        <hr style="margin: 0.5rem 0; opacity: 0.2;">
                        <div style="color: #666;">💡 {meta['model_reason']}</div>
                        <div style="text-align: right; font-size: 0.7rem; color: #999; margin-top: 0.3rem;">⏱️ {meta['response_time']:.2f}s | {meta['tokens_used']} tokens</div>
                    </div>
                    """
                
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start;">
                    <div class="assistant-message" style="max-width: 85%;">
                        {msg["content"]}
                        {meta_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # 입력창
        if prompt := st.chat_input("질문을 입력하세요... (예: 파이썬 코드 짜줘, 시 써줘, 이게 뭐야?)"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

        # 답변 생성 로직 (마지막 메시지가 유저일 경우 실행)
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            with st.spinner("🤔 하이브리드 AI가 생각 중입니다..."):
                result = self.intelligent_model_orchestration(st.session_state.messages[-1]["content"])
                
                if result['success']:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['content'],
                        "metadata": result
                    })
                    
                    # DB 저장
                    try:
                        cursor = self.conn.cursor()
                        cursor.execute('''
                            INSERT INTO conversations 
                            (session_id, user_message, bot_response, model_used, intent_detected, processing_time, tokens_used)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            st.session_state.user_id,
                            st.session_state.messages[-2]["content"],
                            result['content'],
                            result['model_name'],
                            result['intent_analysis']['primary_intent'],
                            result['processing_time'],
                            result['tokens_used']
                        ))
                        self.conn.commit()
                        st.session_state.conversation_count += 1
                    except Exception as e:
                        logger.error(f"DB 저장 실패: {e}")
                        
                    st.rerun()
                else:
                    st.error(f"오류가 발생했습니다: {result.get('error', 'Unknown error')}")

def main():
    ai_system = FreePlanAISystem()
    ai_system.display_beautiful_sidebar()
    ai_system.display_beautiful_chat()

if __name__ == "__main__":
    main()
