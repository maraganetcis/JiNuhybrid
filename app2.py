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
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ✅ 페이지 설정 - 더 예쁜 디자인
st.set_page_config(
    page_title="JiNu hybrid AI",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# 🚀 하이브리드 AI/n 최적의 AI 모델"
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
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
    }
    .model-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
        margin: 0.5rem 0;
    }
    .assistant-message {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 15px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .rate-limit-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
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
        """API 키 설정 - 무료 플랜 최적화"""
        try:
            # Google Gemini (무료 플랜)
            if 'GEMINI_API_KEY' in st.secrets:
                genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
                self.gemini_available = True
                logger.info("Gemini API")
            else:
                self.gemini_available =I False
                logger.warning("Gemini API 키 없음")
            
            # OpenRouter (무료 크레딧 있는 경우)
            self.openrouter_key = st.secrets.get('OPENROUTER_API_KEY', '')
            self.openrouter_available = bool(self.openrouter_key)
            if self.openrouter_available:
                logger.info("OpenRouter 설정 완료")
            
            # DeepSeek (무료)
            self.deepseek_key = st.secrets.get('DEEPSEEK_API_KEY', '')
            self.deepseek_available = bool(self.deepseek_key)
            if self.deepseek_available:
                logger.info("DeepSeek API 설정 완료")
                
            # 사용 가능한 모델 목록
            self.available_models = []
            if self.gemini_available:
                self.available_models.append('gemini')
            if self.openrouter_available:
                self.available_models.append('claude')
            if self.deepseek_available:
                self.available_models.append('deepseek')
                
            logger.info(f"사용 가능한 모델: {self.available_models}")
                
        except Exception as e:
            logger.error(f"API 키 설정 중 오류: {e}")
            st.error("API 키 설정 중 오류가 발생했습니다.")
    
    def setup_rate_limiting(self):
        """무료 플랜을 위한 요청 제한 설정"""
        self.rate_limits = {
            'gemini': {'count': 0, 'last_reset': time.time(), 'max_per_minute': 15},
            'claude': {'count': 0, 'last_reset': time.time(), 'max_per_minute': 10},
            'deepseek': {'count': 0, 'last_reset': time.time(), 'max_per_minute': 20}
        }
    
    def check_rate_limit(self, model: str) -> bool:
        """요청 제한 확인"""
        current_time = time.time()
        limit_info = self.rate_limits[model]
        
        # 1분 이상 지났으면 리셋
        if current_time - limit_info['last_reset'] > 60:
            limit_info['count'] = 0
            limit_info['last_reset'] = current_time
        
        # 제한 체크
        if limit_info['count'] >= limit_info['max_per_minute']:
            return False
        
        limit_info['count'] += 1
        return True
    
    def setup_database(self):
        """데이터베이스 설정"""
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
        """세션 상태 초기화"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'user_id' not in st.session_state:
            st.session_state.user_id = hashlib.md5(
                str(datetime.now().timestamp()).encode()
            ).hexdigest()
        
        if 'conversation_count' not in st.session_state:
            st.session_state.conversation_count = 0
        
        if 'model_usage' not in st.session_state:
            st.session_state.model_usage = {}
        
        if 'rate_limit_hits' not in st.session_state:
            st.session_state.rate_limit_hits = 0

    def advanced_intent_analysis(self, user_input: str) -> Dict:
        """고급 의도 분석 시스템 - 무료 모델에 최적화"""
        intent_keywords = {
            'complex_reasoning': [
                '논리', '추론', '분석', '비교', '평가', '판단', '결론', '가정',
                '전제', '논증', '타당성', '비판적', '사고', '이유', '근거',
                '복잡한', '난이도', '심층', '다단계', '종합', '통합', '철학'
            ],
            'technical': [
                '코드', '프로그래밍', '알고리즘', '개발', '설계', '파이썬', '자바', 
                '함수', '클래스', '디버깅', 'API', 'JSON', '리팩토링'
            ],
            'creative': [
                '작성', '생성', '만들', '글쓰기', '시', '이야기', '창의', '소설',
                '아이디어', '기획', '콘텐츠'
            ],
            'mathematical': [
                '계산', '수학', '공식', '방정식', '통계', '확률', '미분', '적분'
            ],
            'research': [
                '연구', '논문', '참고문헌', '학술', '이론', '실험', '데이터', '조사'
            ],
            'factual': [
                '뭐야', '무엇', '알려줘', '정보', '사실', '정의', '설명'
            ],
            'casual': [
                '안녕', '하이', '잘지내', '고마워', '반가워'
            ]
        }
        
        # 의도 점수 계산
        intent_scores = {}
        user_lower = user_input.lower()
        
        for intent, keywords in intent_keywords.items():
            score = sum(10 for keyword in keywords if keyword in user_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # 복잡도 분석
        word_count = len(user_input.split())
        if word_count > 25:
            complexity = 'very_high'
        elif word_count > 15:
            complexity = 'high'
        elif word_count > 7:
            complexity = 'medium'
        else:
            complexity = 'low'
        
        # 주요 의도 선택
        primary_intent = 'general'
        if intent_scores:
            if 'complex_reasoning' in intent_scores:
                primary_intent = 'complex_reasoning'
            else:
                primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return {
            'primary_intent': primary_intent,
            'all_intents': list(intent_scores.keys()),
            'intent_scores': intent_scores,
            'complexity': complexity,
            'word_count': word_count,
            'is_complex': complexity in ['high', 'very_high']
        }

    def select_optimal_model(self, intent_analysis: Dict) -> Dict:
        """무료 플랜에 최적화된 모델 선택"""
        
        # 무료 플랜 모델 매핑 (Gemini Advanced 제거)
        intent_model_mapping = {
            'complex_reasoning': {
                'primary': 'claude',
                'backup': 'deepseek',
                'fallback': 'gemini',
                'reason': '🧠 복잡한 논리/추론에는 Claude 3.5 Sonnet이 가장 우수',
                'specialization': '논리적 추론, 체계적 분석',
                'icon': '🧠'
            },
            'technical': {
                'primary': 'deepseek',
                'backup': 'gemini', 
                'fallback': 'claude',
                'reason': '💻 코드 및 기술적 문제 해결에는 DeepSeek V3가 최적화',
                'specialization': '프로그래밍, 알고리즘, 개발',
                'icon': '💻'
            },
            'mathematical': {
                'primary': 'deepseek',
                'backup': 'gemini',
                'fallback': 'claude',
                'reason': '🧮 수학적/논리적 연산에는 DeepSeek V3가 강력',
                'specialization': '수학, 계산, 공식',
                'icon': '🧮'
            },
            'research': {
                'primary': 'gemini',
                'backup': 'claude',
                'fallback': 'deepseek',
                'reason': '📚 방대한 텍스트/연구 분석에는 Gemini가 유리',
                'specialization': '연구, 논문, 학술 분석',
                'icon': '📚'
            },
            'creative': {
                'primary': 'claude',
                'backup': 'gemini',
                'fallback': 'deepseek',
                'reason': '🎨 자연스럽고 창의적인 작문은 Claude가 뛰어남',
                'specialization': '창의성, 아이디어, 글쓰기',
                'icon': '🎨'
            },
            'general': {
                'primary': 'gemini',
                'backup': 'deepseek',
                'fallback': 'claude',
                'reason': '⚡ 일반적인 질문에는 빠르고 효율적인 Gemini Flash 사용',
                'specialization': '일반 대화, 기본 질문',
                'icon': '⚡'
            }
        }
        
        primary_intent = intent_analysis['primary_intent']
        model_choice = intent_model_mapping.get(primary_intent, intent_model_mapping['general'])
        
        # 사용 가능한 모델에 따라 선택 조정
        selected_model = None
        
        # 1순위 모델 체크
        if model_choice['primary'] in self.available_models:
            if self.check_rate_limit(model_choice['primary']):
                selected_model = model_choice['primary']
        
        # 2순위 백업 모델 체크
        if not selected_model and model_choice['backup'] in self.available_models:
            if self.check_rate_limit(model_choice['backup']):
                selected_model = model_choice['backup']
                model_choice['reason'] += " (1순위 모델 제한으로 백업 사용)"
        
        # 3순위 폴백 모델 체크
        if not selected_model and model_choice['fallback'] in self.available_models:
            if self.check_rate_limit(model_choice['fallback']):
                selected_model = model_choice['fallback']
                model_choice['reason'] += " (백업 모델 제한으로 폴백 사용)"
        
        # 모든 모델이 제한된 경우
        if not selected_model:
            # 제한이 가장 적은 모델 찾기
            for model in self.available_models:
                if self.check_rate_limit(model):
                    selected_model = model
                    model_choice['reason'] = f"⚠️ 모든 최적 모델 제한으로 {model} 강제 사용"
                    break
        
        model_choice['primary'] = selected_model
        return model_choice

    def call_gemini_api(self, prompt: str, intent: str) -> Dict:
        """Gemini API 호출 - 무료 플랜용"""
        if not self.gemini_available:
            return {'success': False, 'error': 'Gemini API를 사용할 수 없습니다.'}
        
        try:
            start_time = time.time()
            
            # 무료 플랜에서는 항상 gemini-2.5-flash 사용
            model_name = 'gemini-2.5-flash'
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            
            processing_time = time.time() - start_time
            
            # 모델 사용 통계 업데이트
            st.session_state.model_usage['gemini'] = st.session_state.model_usage.get('gemini', 0) + 1
            
            return {
                'success': True,
                'content': response.text,
                'model': f"Google {model_name}",
                'processing_time': processing_time,
                'tokens': len(prompt + response.text) // 4
            }
            
        except Exception as e:
            logger.error(f"Gemini API 오류: {e}")
            return {'success': False, 'error': f'Gemini API 오류: {str(e)}'}

    def call_openrouter_api(self, prompt: str, intent: str) -> Dict:
        """OpenRouter API 호출 - Claude (무료 크레딧)"""
        if not self.openrouter_available:
            return {'success': False, 'error': 'OpenRouter API를 사용할 수 없습니다.'}
        
        try:
            start_time = time.time()
            
            data = {
                "model": "anthropic/claude-3.5-sonnet",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000,
                "temperature": 0.3
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json"
                },
                json=data,
                timeout=60
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                tokens = result.get('usage', {}).get('total_tokens', 0)
                
                # 모델 사용 통계 업데이트
                st.session_state.model_usage['claude'] = st.session_state.model_usage.get('claude', 0) + 1
                
                return {
                    'success': True,
                    'content': content,
                    'model': "Claude 3.5 Sonnet",
                    'processing_time': processing_time,
                    'tokens': tokens
                }
            else:
                return {
                    'success': False, 
                    'error': f'OpenRouter API 오류: {response.status_code}'
                }
                
        except Exception as e:
            logger.error(f"OpenRouter 연결 오류: {e}")
            return {'success': False, 'error': f'OpenRouter 연결 오류: {str(e)}'}

    def call_deepseek_api(self, prompt: str, intent: str) -> Dict:
        """DeepSeek API 호출 - 무료"""
        if not self.deepseek_available:
            return {'success': False, 'error': 'DeepSeek API를 사용할 수 없습니다.'}
        
        try:
            start_time = time.time()
            
            data = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000,
                "temperature": 0.1
            }
            
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {self.deepseek_key}"},
                json=data,
                timeout=60
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                tokens = result.get('usage', {}).get('total_tokens', 0)
                
                # 모델 사용 통계 업데이트
                st.session_state.model_usage['deepseek'] = st.session_state.model_usage.get('deepseek', 0) + 1
                
                return {
                    'success': True,
                    'content': content,
                    'model': "DeepSeek V3",
                    'processing_time': processing_time,
                    'tokens': tokens
                }
            else:
                return {
                    'success': False,
                    'error': f'DeepSeek API 오류: {response.status_code}'
                }
                
        except Exception as e:
            logger.error(f"DeepSeek API 오류: {e}")
            return {'success': False, 'error': f'DeepSeek API 오류: {str(e)}'}

    def intelligent_model_orchestration(self, user_input: str) -> Dict:
        """무료 플랜용 지능형 모델 오케스트레이션"""
        start_time = time.time()
        
        # 1. 고급 의도 분석
        intent_analysis = self.advanced_intent_analysis(user_input)
        
        # 2. 최적 모델 선택 (무료 플랜 고려)
        model_choice = self.select_optimal_model(intent_analysis)
        
        # 3. 선택된 모델로 응답 생성
        selected_model = model_choice['primary']
        responses = {}
        
        if selected_model == 'claude':
            response = self.call_openrouter_api(user_input, intent_analysis['primary_intent'])
            if response['success']:
                responses['claude'] = response
        elif selected_model == 'deepseek':
            response = self.call_deepseek_api(user_input, intent_analysis['primary_intent'])
            if response['success']:
                responses['deepseek'] = response
        elif selected_model == 'gemini':
            response = self.call_gemini_api(user_input, intent_analysis['primary_intent'])
            if response['success']:
                responses['gemini'] = response
        
        # 4. 기본 모델 실패 시 백업 모델 시도
        if not responses and model_choice.get('backup'):
            backup_model = model_choice['backup']
            if backup_model == 'claude' and self.openrouter_available and self.check_rate_limit('claude'):
                response = self.call_openrouter_api(user_input, intent_analysis['primary_intent'])
                if response['success']:
                    responses['claude'] = response
                    selected_model = 'claude'
                    model_choice['reason'] += " (주 모델 실패로 백업 사용)"
            elif backup_model == 'gemini' and self.gemini_available and self.check_rate_limit('gemini'):
                response = self.call_gemini_api(user_input, intent_analysis['primary_intent'])
                if response['success']:
                    responses['gemini'] = response
                    selected_model = 'gemini'
                    model_choice['reason'] += " (주 모델 실패로 백업 사용)"
            elif backup_model == 'deepseek' and self.deepseek_available and self.check_rate_limit('deepseek'):
                response = self.call_deepseek_api(user_input, intent_analysis['primary_intent'])
                if response['success']:
                    responses['deepseek'] = response
                    selected_model = 'deepseek'
                    model_choice['reason'] += " (주 모델 실패로 백업 사용)"
        
        # 5. 최종 응답 선택
        final_response = self.get_final_response(responses)
        total_processing_time = time.time() - start_time
        
        result = {
            'final_response': final_response,
            'selected_model': selected_model,
            'model_reason': model_choice['reason'],
            'model_specialization': model_choice.get('specialization', '일반'),
            'model_icon': model_choice.get('icon', '🤖'),
            'intent_analysis': intent_analysis,
            'processing_time': total_processing_time,
            'success': bool(responses),
            'rate_limited': not bool(responses) and selected_model is not None
        }
        
        # 성공한 응답이 있으면 상세 정보 추가
        if responses:
            first_response = next(iter(responses.values()))
            result.update({
                'content': first_response['content'],
                'model_name': first_response['model'],
                'response_time': first_response['processing_time'],
                'tokens_used': first_response['tokens']
            })
        
        return result

    def get_final_response(self, responses: Dict) -> str:
        """최종 응답 선택"""
        if responses:
            for response in responses.values():
                if response['success']:
                    return response['content']
        
        return "⚠️ 현재 모든 AI 모델의 요청 제한에 도달했거나 일시적으로 사용할 수 없습니다. 잠시 후 다시 시도해주세요."

    def save_conversation(self, user_input: str, result: Dict):
        """대화 저장"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO conversations 
                (session_id, user_message, bot_response, model_used, intent_detected, processing_time, tokens_used, rate_limited)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                st.session_state.user_id,
                user_input,
                result.get('content', ''),
                result.get('model_name', ''),
                result['intent_analysis']['primary_intent'],
                result['processing_time'],
                result.get('tokens_used', 0),
                result.get('rate_limited', False)
            ))
            
            self.conn.commit()
            st.session_state.conversation_count += 1
            
        except Exception as e:
            logger.error(f"대화 저장 오류: {e}")

    def display_beautiful_sidebar(self):
        """무료 플랜 최적화 사이드바"""
        with st.sidebar:
            # 헤더
            st.markdown('<div class="main-header">💠하이브리드 AI</div>', unsafe_allow_html=True)
            
            # 무료 플랜 배지
            st.markdown(
                '<div style="text-align: center; margin-bottom: 1rem;">'
                '<span class="free-badge">HYBRID</span>'
                '</div>', 
                unsafe_allow_html=True
            )
            
            # 시스템 상태 카드
            st.markdown("### 🔧 모델 상태")
            col1, col2, col3 = st.columns(3)
            with col1:
                status = "✅" if self.gemini_available else "❌"
                st.metric("Gemini Flash", status, help="Google Gemini")
            with col2:
                status = "✅" if self.openrouter_available else "❌"
                st.metric("Claude", status, help="OpenRouter ")
            with col3:
                status = "✅" if self.deepseek_available else "❌"
                st.metric("DeepSeek", status, help="DeepSeek API")
            
            st.markdown("---")
            
            # 요청 제한 정보
            st.markdown("### 📊 현재 사용량")
            
            for model, limits in self.rate_limits.items():
                if model in self.available_models:
                    remaining = max(0, limits['max_per_minute'] - limits['count'])
                    st.progress(
                        limits['count'] / limits['max_per_minute'],
                        text=f"{model}: {limits['count']}/{limits['max_per_minute']} 회"
                    )
            
            st.markdown("---")
            
            # 통계 카드
            st.markdown("### 📈 사용 통계")
            st.markdown(f"""
            <div class="stats-card">
                <div style="font-size: 2rem; font-weight: bold; color: #667eea; text-align: center;">
                    {st.session_state.conversation_count}
                </div>
                <div style="text-align: center; color: #6c757d;">총 대화 수</div>
            </div>
            """, unsafe_allow_html=True)
            
            # 모델별 사용량
            if st.session_state.model_usage:
                st.markdown("#### 🎯 모델 사용 비율")
                total_usage = sum(st.session_state.model_usage.values())
                for model, count in st.session_state.model_usage.items():
                    percentage = (count / total_usage * 100) if total_usage > 0 else 0
                    st.write(f"{model}: {count}회 ({percentage:.1f}%)")
            
            st.markdown("---")
            
            # 무료 모델 특기 안내
            st.markdown("### 🏆 모델 특기")
            
           # free_model_specs 리스트 수정 (type 키 제거)
free_model_specs = [
    {"icon": "🧠", "name": "Claude 3.5", "desc": "논리적 추론, 창의적 작문"},
    {"icon": "⚡", "name": "Gemini Flash", "desc": "빠른 응답, 일반 질문"}, 
    {"icon": "💻", "name": "DeepSeek V3", "desc": "코딩, 수학, 기술 질문"}
]
            
            for spec in free_model_specs:
                st.markdown(f"""
                <div class="model-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="font-size: 1.2rem;">
                            {spec['icon']} <strong>{spec['name']}</strong>
                        </div>
                        <span class="free-badge" style="font-size: 0.6rem; padding: 0.2rem 0.5rem;">{spec['type']}</span>
                    </div>
                    <div style="color: #6c757d; font-size: 0.9rem; margin-top: 0.5rem;">
                        {spec['desc']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 무료 플랜 사용 팁
            st.markdown("### 💡 하이브리드 AI 팁")
            tips = [
                "🔄 **모델 순환**: 시스템이 자동으로 모델을 전환해요",
                "⏱️ **요청 분산**: 분당 요청 제한을 초과하지 않도록 해요", 
                "🎯 **의도 명확히**: 질문을 명확히 하면 더 좋은 응답을 받아요",
                "⚡ **가벼운 질문**: Gemini Flash가 가장 빠르고 경제적이에요"
            ]
            
            for tip in tips:
                st.markdown(f"<div style='margin: 0.5rem 0; font-size: 0.9rem;'>{tip}</div>", unsafe_allow_html=True)
            
            # 대화 지우기 버튼
            if st.button("🗑️ 대화 기록 지우기", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.session_state.conversation_count = 0
                st.session_state.model_usage = {}
                st.rerun()

    def display_beautiful_chat(self):
        """무료 플랜 최적화 채팅 인터페이스"""
        # 헤더
        st.markdown('<div class="main-header">💠 하이브리드 AI</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">'
            '하이브리드 모델을 자동으로 선택합니다<br>'
            '<small>요청 제한을 고려하여 최적의 모델을 자동 선택</small>'
            '</div>', 
            unsafe_allow_html=True
        )
        
        # 요청 제한 경고 표시
        total_requests = sum(limit['count'] for limit in self.rate_limits.values())
        if total_requests > 50:
            st.markdown("""
            <div class="rate-limit-warning">
                ⚠️ <strong>요청 제한 접근 중</strong><br>
                현재 많은 요청을 보내고 있습니다. 무료 플랜 제한을 초과하면 일시적으로 사용이 제한될 수 있습니다.
            </div>
            """, unsafe_allow_html=True)
        
        # 채팅 컨테이너
        chat_container = st.container()
        
        with chat_container:
            # 대화 기록
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-end; margin: 1rem 0;">
                        <div class="user-message" style="max-width: 70%;">
                            <div style="font-weight: 600; margin-bottom: 0.5rem;">👤 You</div>
                            {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # AI 응답 메타정보
                    metadata_html = ""
                    if "metadata" in message:
                        meta = message['metadata']
                        complexity_class = f"complexity-{meta['intent_analysis']['complexity']}"
                        
                        # 무료 배지 추가
                        free_badge = ""
                        if 'Gemini' in meta['model_name']:
                            free_badge = '<span class="free-badge" style="font-size: 0.6rem; margin-left: 0.5rem;">FREE</span>'
                        elif 'DeepSeek' in meta['model_name']:
                            free_badge = '<span class="free-badge" style="font-size: 0.6rem; margin-left: 0.5rem;">FREE</span>'
                        
                        metadata_html = f"""
                        <div style="margin-top: 1rem; padding: 1rem; background: white; border-radius: 10px; border: 1px solid #e0e0e0;">
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                                <div>
                                    <div style="font-size: 0.8rem; color: #6c757d;">선택 모델</div>
                                    <div style="font-weight: 600; color: #667eea; display: flex; align-items: center;">
                                        {meta['model_name']} {free_badge}
                                    </div>
                                </div>
                                <div>
                                    <div style="font-size: 0.8rem; color: #6c757d;">감지 의도</div>
                                    <div style="font-weight: 600;">{meta['intent_analysis']['primary_intent']}</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.8rem; color: #6c757d;">복잡도</div>
                                    <span class="intent-badge {complexity_class}">
                                        {meta['intent_analysis']['complexity']}
                                    </span>
                                </div>
                                <div>
                                    <div style="font-size: 0.8rem; color: #6c757d;">처리 시간</div>
                                    <div style="font-weight: 600;">{meta['response_time']:.2f}s</div>
                                </div>
                            </div>
                            <div style="font-size: 0.9rem; color: #495057;">
                                <strong>선택 이유:</strong> {meta['model_reason']}
                            </div>
                        </div>
                        """
                    
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-start; margin: 1rem 0;">
                        <div class="assistant-message" style="max-width: 70%;">
                            <div style="font-weight: 600; margin-bottom: 0.5rem;">🤖 Assistant</div>
                            {message["content"]}
                            {metadata_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # 입력창
        st.markdown("---")
        prompt = st.chat_input(
            "하이브리드 AI에게 질문을 입력하세요...",
            key="chat_input"
        )
        
        if prompt:
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # AI 응답 생성
            with st.spinner("🤔 하이브리드 AI가 답변 생성 중..."):
                result = self.intelligent_model_orchestration(prompt)
            
            if result['success']:
                # 성공한 응답 표시
                complexity_class = f"complexity-{result['intent_analysis']['complexity']}"
                
                # 무료 배지
                free_badge = ""
                if 'Gemini' in result['model_name'] or 'DeepSeek' in result['model_name']:
                    free_badge = '<span class="free-badge" style="font-size: 0.6rem; margin-left: 0.5rem;">FREE</span>'
                elif 'Claude' in result['model_name']:
                    free_badge = '<span class="free-badge" style="font-size: 0.6rem; margin-left: 0.5rem;">FREE CREDIT</span>'
                
                response_html = f"""
                <div style="display: flex; justify-content: flex-start; margin: 1rem 0;">
                    <div class="assistant-message" style="max-width: 70%;">
                        <div style="font-weight: 600; margin-bottom: 0.5rem;">🤖 Assistant</div>
                        {result['content']}
                        <div style="margin-top: 1rem; padding: 1rem; background: white; border-radius: 10px; border: 1px solid #e0e0e0;">
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                                <div>
                                    <div style="font-size: 0.8rem; color: #6c757d;">선택 모델</div>
                                    <div style="font-weight: 600; color: #667eea; display: flex; align-items: center;">
                                        {result['model_name']} {free_badge}
                                    </div>
                                </div>
                                <div>
                                    <div style="font-size: 0.8rem; color: #6c757d;">감지 의도</div>
                                    <div style="font-weight: 600;">{result['intent_analysis']['primary_intent']}</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.8rem; color: #6c757d;">복잡도</div>
                                    <span class="intent-badge {complexity_class}">
                                        {result['intent_analysis']['complexity']}
                                    </span>
                                </div>
                                <div>
                                    <div style="font-size: 0.8rem; color: #6c757d;">토큰</div>
                                    <div style="font-weight: 600;">{result['tokens_used']}</div>
                                </div>
                            </div>
                            <div style="font-size: 0.9rem; color: #495057;">
                                <strong>선택 이유:</strong> {result['model_reason']}
                            </div>
                            <div style="font-size: 0.8rem; color: #6c757d; margin-top: 0.5rem;">
                                처리 시간: {result['processing_time']:.2f}초
                            </div>
                        </div>
                    </div>
                </div>
                """
                
                st.markdown(response_html, unsafe_allow_html=True)
                
                # 세션에 메시지 저장
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result['content'],
                    "metadata": {
                        'model_name': result['model_name'],
                        'model_reason': result['model_reason'],
                        'intent_analysis': result['intent_analysis'],
                        'response_time': result.get('response_time', 0),
                        'tokens_used': result.get('tokens_used', 0)
                    }
                })
                
                # 데이터베이스 저장
                self.save_conversation(prompt, result)
                
            else:
                # 실패한 응답 표시 (요청 제한 등)
                if result.get('rate_limited'):
                    st.error(f"⏱️ {result['final_response']}")
                    st.session_state.rate_limit_hits += 1
                else:
                    st.error(f"❌ {result['final_response']}")
            
            st.rerun()

def main():
    # 시스템 초기화
    ai_system = FreePlanAISystem()
    
    # 사이드바 표시
    ai_system.display_beautiful_sidebar()
    
    # 메인 채팅 인터페이스
    ai_system.display_beautiful_chat()
    
    # 푸터
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6c757d; font-size: 0.9rem;'>"
        "copyright ©️ 2025. Synox Studios"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()