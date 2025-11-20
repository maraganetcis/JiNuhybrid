import streamlit as st
import google.generativeai as genai
import requests
from typing import Dict, List
import json
import time
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="하이브리드 멀티 LLM 시스템",
    page_icon="🧠",
    layout="wide"
)

class StreamlitAISystem:
    def __init__(self):
        self.setup_api_keys()
        self.initialize_session_state()
    
    def setup_api_keys(self):
        """Streamlit secrets에서 API 키 설정 (없을 경우 처리)"""
        self.gemini_available = False
        self.openrouter_available = False
        self.deepseek_available = False

        try:
            # Google Gemini
            if 'GOOGLE_API_KEY' in st.secrets:
                genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
                self.gemini_available = True
            
            # OpenRouter (Claude 등)
            if 'OPENROUTER_API_KEY' in st.secrets:
                self.openrouter_key = st.secrets['OPENROUTER_API_KEY']
                self.openrouter_available = True
            
            # DeepSeek
            if 'DEEPSEEK_API_KEY' in st.secrets:
                self.deepseek_key = st.secrets['DEEPSEEK_API_KEY']
                self.deepseek_available = True
                
        except FileNotFoundError:
            st.warning("⚠️ `.streamlit/secrets.toml` 파일이 발견되지 않았습니다. API 키를 설정해주세요.")
        except Exception as e:
            st.error(f"API 키 설정 중 오류: {e}")
    
    def initialize_session_state(self):
        """세션 상태 초기화"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
            
        if 'model_usage' not in st.session_state:
            st.session_state.model_usage = {}

    def advanced_intent_analysis(self, user_input: str) -> Dict:
        """고급 의도 분석 시스템"""
        intent_keywords = {
            'complex_reasoning': [
                '논리', '추론', '분석', '비교', '평가', '판단', '결론', '가정',
                '전제', '논증', '타당성', '비판적', '사고', '이유', '근거',
                '복잡한', '난이도', '심층', '다단계', '종합', '통합', '철학'
            ],
            'technical': ['코드', '프로그래밍', '알고리즘', '개발', '설계', '파이썬', '자바', '에러', '버그', 'api', 'json'],
            'creative': ['작성', '생성', '만들', '글쓰기', '시', '이야기', '창의', '소설', '아이디어'],
            'mathematical': ['계산', '수학', '공식', '방정식', '통계', '확률', '미분', '적분', '수치'],
            'research': ['연구', '논문', '참고문헌', '학술', '이론', '실험', '데이터', '동향'],
            'factual': ['뭐야', '무엇', '알려줘', '정보', '사실', '정의', '설명'],
            'casual': ['안녕', '하이', '잘지내', '고마워', '반가워']
        }
        
        # 의도 점수 계산
        intent_scores = {}
        user_lower = user_input.lower()
        
        for intent, keywords in intent_keywords.items():
            score = sum(10 for keyword in keywords if keyword in user_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # 복잡도 분석 강화
        word_count = len(user_input.split())
        has_complex_indicators = any(word in user_lower for word in [
            '분석', '비교', '평가', '논리', '추론', '전제', '결론'
        ])
        
        if word_count > 30 or has_complex_indicators:
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
            # 복잡한 추론 키워드가 있으면 우선순위 부여
            if 'complex_reasoning' in intent_scores and intent_scores['complex_reasoning'] >= 10:
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
        """최적의 AI 모델 선택 로직 (최신 모델 반영)"""
        
        # 기본 매핑 전략
        intent_model_mapping = {
            'complex_reasoning': {
                'primary': 'claude',
                'reason': '🧠 복잡한 논리/추론에는 Claude 3.5 Sonnet이 가장 우수',
                'backup': 'gemini_advanced'
            },
            'technical': {
                'primary': 'deepseek',  # 코딩은 DeepSeek V3가 강력하고 저렴
                'reason': '💻 코드 및 기술적 문제 해결에는 DeepSeek V3가 최적화',
                'backup': 'claude'
            },
            'mathematical': {
                'primary': 'gemini_advanced', # 또는 DeepSeek
                'reason': '🧮 수학적/논리적 연산에는 Gemini 1.5 Pro가 강력',
                'backup': 'deepseek'
            },
            'research': {
                'primary': 'gemini_advanced', # 긴 컨텍스트 강점
                'reason': '📚 방대한 텍스트/연구 분석에는 Gemini의 긴 컨텍스트 창이 유리',
                'backup': 'claude'
            },
            'creative': {
                'primary': 'claude',
                'reason': '🎨 자연스럽고 창의적인 작문은 Claude가 뛰어남',
                'backup': 'gemini'
            },
            'general': {
                'primary': 'gemini',
                'reason': '⚡ 일반적인 질문에는 빠르고 효율적인 Gemini Flash 사용',
                'backup': 'claude'
            }
        }
        
        # 복잡도가 매우 높으면 고성능 모델 강제 사용
        if intent_analysis['complexity'] == 'very_high':
            if intent_analysis['primary_intent'] == 'technical':
                 # 복잡한 코딩은 Claude나 DeepSeek 유지
                 pass
            else:
                # 그 외 복잡한건 Claude 우선
                primary_intent = 'complex_reasoning'
                model_choice = intent_model_mapping['complex_reasoning']
        else:
            model_choice = intent_model_mapping.get(intent_analysis['primary_intent'], intent_model_mapping['general'])
        
        # 모델 가용성 체크 및 폴백(Fallback) 로직
        selected_model = model_choice['primary']
        
        # 1. Claude 선택 시
        if selected_model == 'claude' and not self.openrouter_available:
            selected_model = model_choice['backup']
            model_choice['reason'] += " (Claude 키 없음 -> 백업 모델 사용)"

        # 2. DeepSeek 선택 시
        if selected_model == 'deepseek' and not self.deepseek_available:
            selected_model = 'gemini' if self.gemini_available else 'backup_unavailable'
            model_choice['reason'] += " (DeepSeek 키 없음 -> Gemini 사용)"

        # 3. Gemini 선택 시
        if 'gemini' in selected_model and not self.gemini_available:
            selected_model = 'claude' if self.openrouter_available else 'backup_unavailable'
            model_choice['reason'] += " (Gemini 키 없음 -> Claude 사용)"

        model_choice['primary'] = selected_model
        return model_choice

    def call_advanced_models(self, prompt: str, intent: str, model_type: str) -> Dict:
        """모델 API 호출 실행"""
        
        # 시스템 프롬프트 강화
        reasoning_prompts = {
            'complex_reasoning': """당신은 논리적 분석 전문가입니다. 심호흡을 하고 차근차근 생각해보세요(Chain of Thought).
            1. 핵심 주장과 전제를 식별하세요.
            2. 논리적 허점이나 모순을 찾으세요.
            3. 다각도로 분석한 뒤 결론을 내리세요.""",
            
            'technical': """당신은 수석 소프트웨어 엔지니어입니다.
            1. 요구사항을 명확히 분석하세요.
            2. 효율적이고 안전한 코드를 작성하세요.
            3. 코드에 대한 간결한 설명을 덧붙이세요.""",
            
            'mathematical': """당신은 수학자입니다.
            1. 문제를 수식으로 정의하세요.
            2. 풀이 과정을 단계별로 보여주세요.
            3. 최종 답안을 검증하세요."""
        }
        
        system_instruction = reasoning_prompts.get(intent, "당신은 유능한 AI 어시스턴트입니다. 질문에 명확하고 도움이 되도록 답변하세요.")
        full_prompt = f"{system_instruction}\n\n질문: {prompt}"
        
        try:
            # --- Google Gemini ---
            if 'gemini' in model_type and self.gemini_available:
                # Gemini 모델 버전 업데이트 (1.0 -> 1.5)
                if model_type == 'gemini_advanced':
                    model_name = 'gemini-1.5-pro'
                else:
                    model_name = 'gemini-1.5-flash' # 1.0-pro 대신 Flash 사용 (더 빠르고 성능 좋음)
                
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(full_prompt)
                
                return {
                    'success': True,
                    'content': response.text,
                    'model': f"Google {model_name}",
                    'tokens': self._estimate_tokens(full_prompt + response.text)
                }
                
            # --- Claude (via OpenRouter) ---
            elif model_type == 'claude' and self.openrouter_available:
                data = {
                    "model": "anthropic/claude-3.5-sonnet",
                    "messages": [{"role": "user", "content": full_prompt}],
                    "max_tokens": 4000, # 토큰 수 증가
                    "temperature": 0.2
                }
                
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openrouter_key}",
                        "HTTP-Referer": "http://localhost:8501", 
                    },
                    json=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    tokens = result.get('usage', {}).get('total_tokens', 0)
                    return {'success': True, 'content': content, 'model': 'Claude 3.5 Sonnet', 'tokens': tokens}
                else:
                    return {'success': False, 'error': f'OpenRouter API 오류: {response.status_code} - {response.text}'}
            
            # --- DeepSeek ---
            elif model_type == 'deepseek' and self.deepseek_available:
                data = {
                    "model": "deepseek-chat", # V3 Chat
                    "messages": [{"role": "user", "content": full_prompt}],
                    "max_tokens": 4000,
                    "temperature": 0.1 # 코딩/논리는 낮은 온도
                }
                
                response = requests.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={"Authorization": f"Bearer {self.deepseek_key}"},
                    json=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    tokens = result.get('usage', {}).get('total_tokens', 0)
                    return {'success': True, 'content': content, 'model': 'DeepSeek V3', 'tokens': tokens}
                else:
                    return {'success': False, 'error': f'DeepSeek API 오류: {response.status_code}'}
            
            return {'success': False, 'error': f'선택된 모델({model_type})을 사용할 수 없습니다. API 키를 확인하세요.'}
                
        except Exception as e:
            return {'success': False, 'error': f'모델 호출 중 예외 발생: {str(e)}'}

    def _estimate_tokens(self, text):
        """간이 토큰 계산 (Gemini용)"""
        return len(text) // 4

# --- UI 컴포넌트 함수들 ---

def display_sidebar_info(intent_analysis, model_choice):
    """사이드바 정보 표시"""
    st.sidebar.header("🎯 의도 및 모델 분석")
    
    # 의도
    st.sidebar.subheader("User Intent")
    primary = intent_analysis['primary_intent']
    st.sidebar.info(f"**의도**: {primary.upper()}")
    st.sidebar.text(f"복잡도: {intent_analysis['complexity']}")
    
    # 모델
    st.sidebar.subheader("Selected Model")
    model_name = model_choice['primary']
    st.sidebar.success(f"**모델**: {model_name}")
    st.sidebar.caption(f"💡 {model_choice['reason']}")
    
    # 모델별 특징 안내
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏆 모델별 특기")
    st.sidebar.markdown("""
    - **Claude 3.5 Sonnet**: 
      논리적 추론, 뉘앙스 파악, 작문
    - **Gemini 1.5 Pro/Flash**: 
      긴 문맥 처리, 멀티모달, 속도
    - **DeepSeek V3**: 
      코딩, 수학, 가성비 최강
    """)

def main():
    st.title("🧠 하이브리드 AI 오케스트레이터")
    st.markdown("사용자의 질문 의도와 복잡도를 분석하여 **Claude, Gemini, DeepSeek** 중 최적의 모델을 자동으로 연결합니다.")
    
    # 시스템 초기화
    ai_system = StreamlitAISystem()
    
    # API 키 상태 확인 UI
    with st.expander("🔑 시스템 상태 및 API 키 확인", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Gemini", "Ready" if ai_system.gemini_available else "Missing")
        col2.metric("Claude(OpenRouter)", "Ready" if ai_system.openrouter_available else "Missing")
        col3.metric("DeepSeek", "Ready" if ai_system.deepseek_available else "Missing")
        if not any([ai_system.gemini_available, ai_system.openrouter_available, ai_system.deepseek_available]):
            st.error("설정된 API 키가 없습니다. `.streamlit/secrets.toml` 파일을 확인해주세요.")

    # 채팅 인터페이스
    st.divider()
    
    # 대화 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "model" in message:
                st.caption(f"🛠 {message['model']} | 의도: {message.get('intent', 'N/A')}")

    # 입력창
    if prompt := st.chat_input("질문을 입력하세요 (예: '파이썬으로 뱀 게임 코드 짜줘', '이 논리의 모순점은?')"):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 분석 및 생성
        with st.spinner("🤔 의도 분석 및 최적 모델 선정 중..."):
            # 1. 의도 분석
            intent_analysis = ai_system.advanced_intent_analysis(prompt)
            # 2. 모델 선택
            model_choice = ai_system.select_optimal_model(intent_analysis)
            
            # 사이드바 업데이트
            display_sidebar_info(intent_analysis, model_choice)
            
        with st.chat_message("assistant"):
            msg_placeholder = st.empty()
            
            # 3. 모델 호출
            with st.spinner(f"🚀 {model_choice['primary']} 모델이 답변을 생성하고 있습니다..."):
                response = ai_system.call_advanced_models(
                    prompt, 
                    intent_analysis['primary_intent'],
                    model_choice['primary']
                )
            
            if response['success']:
                msg_placeholder.markdown(response['content'])
                
                # 메타데이터 표시
                st.success(f"Used: **{response['model']}** (Tokens: approx. {response['tokens']})")
                
                # 기록 저장
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response['content'],
                    "model": response['model'],
                    "intent": intent_analysis['primary_intent']
                })
                
                # 통계 업데이트
                if response['model'] not in st.session_state.model_usage:
                    st.session_state.model_usage[response['model']] = 0
                st.session_state.model_usage[response['model']] += 1
                
            else:
                st.error(f"❌ 오류 발생: {response.get('error')}")
                st.info("제안: 다른 모델을 사용하도록 질문을 변경하거나 API 키를 확인하세요.")

if __name__ == "__main__":
    main()
