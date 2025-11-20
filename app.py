import streamlit as st
import google.generativeai as genai
import requests
from typing import Dict, List
import json
import time
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="하이브리드 AI 시스템",
    page_icon="🧠",
    layout="wide"
)

class StreamlitAISystem:
    def __init__(self):
        self.setup_api_keys()
        self.initialize_session_state()
    
    def setup_api_keys(self):
        """Streamlit secrets에서 API 키 설정"""
        try:
            # Google Gemini
            if 'GOOGLE_API_KEY' in st.secrets:
                genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
                self.gemini_available = True
            else:
                self.gemini_available = False
            
            # OpenRouter
            self.openrouter_key = st.secrets.get('OPENROUTER_API_KEY', '')
            self.openrouter_available = bool(self.openrouter_key)
            
            # DeepSeek
            self.deepseek_key = st.secrets.get('DEEPSEEK_API_KEY', '')
            self.deepseek_available = bool(self.deepseek_key)
            
        except Exception as e:
            st.error(f"API 키 설정 중 오류: {e}")
    
    def initialize_session_state(self):
        """세션 상태 초기화"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        if 'total_cost' not in st.session_state:
            st.session_state.total_cost = 0.0
            
        if 'model_usage' not in st.session_state:
            st.session_state.model_usage = {}

    def advanced_intent_analysis(self, user_input: str) -> Dict:
        """고급 의도 분석 시스템 - 처음 코드 기준"""
        intent_keywords = {
            'complex_reasoning': [
                '논리', '추론', '분석', '비교', '평가', '판단', '결론', '가정',
                '전제', '논증', '타당성', '비판적', '사고', '이유', '근거',
                '복잡한', '난이도', '심층', '다단계', '종합', '통합'
            ],
            'technical': ['코드', '프로그래밍', '알고리즘', '개발', '설계', '파이썬', '자바'],
            'creative': ['작성', '생성', '만들', '글쓰기', '시', '이야기', '창의'],
            'mathematical': ['계산', '수학', '공식', '방정식', '통계', '확률', '미분'],
            'research': ['연구', '논문', '참고문헌', '학술', '이론', '실험', '데이터'],
            'factual': ['뭐야', '무엇', '알려줘', '정보', '사실', '정의'],
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
        
        if word_count > 20 or has_complex_indicators:
            complexity = 'very_high'
        elif word_count > 12:
            complexity = 'high'
        elif word_count > 6:
            complexity = 'medium'
        else:
            complexity = 'low'
        
        # 주요 의도 선택 (복잡한 추론 우선)
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
        """최적의 AI 모델 선택 - 처음 코드 기준"""
        intent_model_mapping = {
            'complex_reasoning': {
                'primary': 'claude',
                'reason': '🧠 복잡한 추론에는 Claude 3.5 Sonnet이 가장 우수함',
                'backup': 'gemini_advanced'
            },
            'technical': {
                'primary': 'gemini',
                'reason': '🔧 기술/코드 관련 질문에는 Gemini가 최적화됨',
                'backup': 'claude'
            },
            'mathematical': {
                'primary': 'gemini_advanced',
                'reason': '🧮 수학적 추론에는 Gemini Advanced가 정확도 높음',
                'backup': 'claude'
            },
            'research': {
                'primary': 'claude', 
                'reason': '📊 연구/학술 분석에는 Claude의 깊은 이해력이 적합',
                'backup': 'gemini_advanced'
            },
            'analytical': {
                'primary': 'gemini_advanced',
                'reason': '🔍 분석적 사고에는 Gemini의 논리력이 뛰어남',
                'backup': 'claude'
            },
            'creative': {
                'primary': 'claude',
                'reason': '🎨 창의적 사고에는 Claude의 유연성이 좋음',
                'backup': 'gemini'
            },
            'general': {
                'primary': 'gemini',
                'reason': '⚡ 일반 질문에는 Gemini의 빠른 응답이 적합',
                'backup': 'claude'
            }
        }
        
        # 복잡도가 매우 높으면 복잡한 추론 모델 강제 사용
        if intent_analysis['complexity'] == 'very_high':
            primary_intent = 'complex_reasoning'
        else:
            primary_intent = intent_analysis['primary_intent']
        
        model_choice = intent_model_mapping.get(primary_intent, intent_model_mapping['general'])
        
        # 사용 가능한 모델 확인
        if model_choice['primary'] == 'claude' and not self.openrouter_available:
            model_choice['primary'] = model_choice['backup']
        elif model_choice['primary'] == 'gemini_advanced' and not self.gemini_available:
            model_choice['primary'] = 'gemini'
        
        return model_choice

    def call_advanced_models(self, prompt: str, intent: str, model_type: str) -> Dict:
        """고급 모델 호출 - 복잡한 추론 특화 (처음 코드 기준)"""
        
        reasoning_prompts = {
            'complex_reasoning': """
            당신은 논리적 추론 전문가입니다. 다음 단계로 접근해주세요:
            1. 문제의 핵심 요소 분석
            2. 가정과 전제 확인  
            3. 논리적 연결고리 도출
            4. 결론 도출 및 검증
            
            질문: {prompt}
            """,
            'mathematical': """
            당신은 수학적 사고 전문가입니다. 체계적으로 접근해주세요:
            1. 문제 이해 및 변수 정의
            2. 관련 공식/이론 적용
            3. 단계별 계산
            4. 결과 검증
            
            질문: {prompt}
            """,
            'technical': """
            당신은 소프트웨어 엔지니어링 전문가입니다:
            1. 문제 분석 및 요구사항 이해
            2. 최적의 솔루션 설계
            3. 실용적인 코드 구현
            4. 테스트 및 검증 방법 제시
            
            질문: {prompt}
            """
        }
        
        specialized_prompt = reasoning_prompts.get(
            intent, 
            "체계적으로 분석하고 논리적으로 답변해주세요: {prompt}"
        ).format(prompt=prompt)
        
        try:
            if model_type in ['gemini', 'gemini_advanced'] and self.gemini_available:
                if model_type == 'gemini_advanced':
                    model = genai.GenerativeModel('gemini-1.5-pro')
                else:
                    model = genai.GenerativeModel('gemini-1.0-pro')
                
                response = model.generate_content(specialized_prompt)
                
                return {
                    'success': True,
                    'content': response.text,
                    'model': 'Gemini ' + ('Advanced' if model_type == 'gemini_advanced' else 'Flash'),
                    'tokens': len(prompt.split()) + len(response.text.split())
                }
                
            elif model_type == 'claude' and self.openrouter_available:
                data = {
                    "model": "anthropic/claude-3.5-sonnet",
                    "messages": [{"role": "user", "content": specialized_prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.3
                }
                
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.openrouter_key}"},
                    json=data,
                    timeout=45
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    tokens = result.get('usage', {}).get('total_tokens', 0)
                    
                    return {
                        'success': True,
                        'content': content,
                        'model': 'Claude 3.5 Sonnet',
                        'tokens': tokens
                    }
                else:
                    return {'success': False, 'error': f'Claude API 오류: {response.status_code}'}
            
            elif model_type == 'deepseek' and self.deepseek_available:
                data = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": specialized_prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.3
                }
                
                response = requests.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={"Authorization": f"Bearer {self.deepseek_key}"},
                    json=data,
                    timeout=45
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    tokens = result.get('usage', {}).get('total_tokens', 0)
                    
                    return {
                        'success': True,
                        'content': content,
                        'model': 'DeepSeek V3',
                        'tokens': tokens
                    }
                else:
                    return {'success': False, 'error': f'DeepSeek API 오류: {response.status_code}'}
            
            return {'success': False, 'error': '사용 가능한 모델이 없습니다.'}
                
        except Exception as e:
            return {'success': False, 'error': f'모델 호출 중 오류: {str(e)}'}

def display_intent_analysis(intent_analysis: Dict):
    """의도 분석 결과 표시 - 처음 UI 디자인대로"""
    st.sidebar.markdown("### 🎯 의도 분석 결과")
    
    # 주요 의도
    intent_icons = {
        'complex_reasoning': '🧠',
        'technical': '🔧', 
        'creative': '🎨',
        'mathematical': '🧮',
        'research': '📊',
        'factual': 'ℹ️',
        'casual': '💬',
        'general': '⚡'
    }
    
    primary_intent = intent_analysis['primary_intent']
    icon = intent_icons.get(primary_intent, '⚡')
    
    st.sidebar.markdown(f"**주요 의도**: {icon} {primary_intent}")
    st.sidebar.markdown(f"**복잡도**: {intent_analysis['complexity']}")
    st.sidebar.markdown(f"**단어 수**: {intent_analysis['word_count']}")
    
    # 모든 의도 점수
    if intent_analysis['intent_scores']:
        st.sidebar.markdown("**의도 점수**:")
        for intent, score in intent_analysis['intent_scores'].items():
            icon = intent_icons.get(intent, '⚡')
            st.sidebar.markdown(f"- {icon} {intent}: {score}점")

def display_model_selection(model_choice: Dict, intent_analysis: Dict):
    """모델 선택 정보 표시 - 처음 UI 디자인대로"""
    st.sidebar.markdown("### 🤖 모델 선택 정보")
    
    model_icons = {
        'claude': '🧠',
        'gemini': '🔧',
        'gemini_advanced': '🧮',
        'deepseek': '💰'
    }
    
    primary_model = model_choice['primary']
    icon = model_icons.get(primary_model, '⚡')
    
    st.sidebar.markdown(f"**선택된 모델**: {icon} {primary_model}")
    st.sidebar.markdown(f"**선택 이유**: {model_choice['reason']}")
    
    # 복잡도에 따른 모델 선택 로직 표시
    st.sidebar.markdown("### ⚙️ 모델 선택 로직")
    if intent_analysis['is_complex']:
        if intent_analysis['primary_intent'] == 'complex_reasoning':
            st.sidebar.markdown("`model = 'claude'  # 가장 강력한 추론 모델`")
        elif intent_analysis['primary_intent'] == 'mathematical':
            st.sidebar.markdown("`model = 'gemini_advanced'  # 수학적 추론 특화`")
        else:
            st.sidebar.markdown("`model = 'claude'  # 일반적 복잡 추론`")
    else:
        st.sidebar.markdown("`model = 'gemini'  # 일반 질문용`")

def display_model_comparison():
    """모델 비교표 표시 - 처음 UI 디자인대로"""
    st.sidebar.markdown("### 💡 모델 추론 강점")
    
    comparison_data = {
        "모델": ["Claude 3.5 Sonnet", "Gemini Thinking", "Llama 3 70B"],
        "추론 강점": ["논리적 일관성, 비판적 사고", "체계적 접근, 단계적 추론", "광범위한 지식 통합"],
        "최적 사용처": ["철학적 논증, 복잡한 분석", "수학적 문제, 알고리즘", "연구 분석, 종합적 판단"]
    }
    
    for i in range(len(comparison_data["모델"])):
        st.sidebar.markdown(f"**{comparison_data['모델'][i]}**")
        st.sidebar.markdown(f"- 강점: {comparison_data['추론 강점'][i]}")
        st.sidebar.markdown(f"- 사용처: {comparison_data['최적 사용처'][i]}")

def main():
    st.title("🧠 복잡한 추론 작업에 최적화된 AI 모델")
    
    st.markdown("""
    **현재 복잡한 추론 작업에는 주로 다음과 같은 모델들이 사용됩니다:**
    
    🏆 **복잡한 추론에 가장 강력한 모델들**
    1. Google Gemini 2.0/2.5 Pro 시리즈
    2. Anthropic Claude 3.5 Sonnet  
    3. Meta Llama 3 70B
    """)
    
    # AI 시스템 초기화
    ai_system = StreamlitAISystem()
    
    # 사이드바에 모델 비교표 표시
    display_model_comparison()
    
    # 채팅 인터페이스
    st.markdown("---")
    st.subheader("💬 대화하기")
    
    # 대화 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "model" in message:
                st.caption(f"모델: {message['model']}")
            if "intent" in message:
                st.caption(f"의도: {message['intent']}")
    
    # 사용자 입력
    if prompt := st.chat_input("복잡한 추론 질문을 입력해주세요..."):
        # 사용자 메시지 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 의도 분석 및 모델 선택
        with st.spinner("의도 분석 중..."):
            intent_analysis = ai_system.advanced_intent_analysis(prompt)
            model_choice = ai_system.select_optimal_model(intent_analysis)
        
        # 사이드바에 분석 결과 표시
        display_intent_analysis(intent_analysis)
        display_model_selection(model_choice, intent_analysis)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner(f"{model_choice['primary']} 모델로 응답 생성 중..."):
                response = ai_system.call_advanced_models(
                    prompt, 
                    intent_analysis['primary_intent'],
                    model_choice['primary']
                )
                
                if response['success']:
                    # 응답 표시
                    message_placeholder.markdown(response['content'])
                    
                    # 모델 정보 표시
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"**모델**: {response['model']}")
                    with col2:
                        st.caption(f"**토큰 사용량**: {response['tokens']}")
                    
                    # 세션 상태에 메시지 저장
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response['content'],
                        "model": response['model'],
                        "intent": intent_analysis['primary_intent']
                    })
                    
                    # 대화 기록 저장
                    st.session_state.conversation_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "user_input": prompt,
                        "ai_response": response['content'],
                        "model_used": response['model'],
                        "intent": intent_analysis['primary_intent'],
                        "tokens_used": response['tokens']
                    })
                    
                    # 사용 통계 업데이트
                    if response['model'] not in st.session_state.model_usage:
                        st.session_state.model_usage[response['model']] = 0
                    st.session_state.model_usage[response['model']] += 1
                    
                else:
                    st.error(f"❌ 오류: {response.get('error', '알 수 없는 오류')}")

    # 사용 통계 표시
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 사용 통계")
    st.sidebar.markdown(f"**총 대화**: {len(st.session_state.conversation_history)}")
    
    if st.session_state.model_usage:
        st.sidebar.markdown("**모델 사용량**:")
        for model, count in st.session_state.model_usage.items():
            st.sidebar.markdown(f"- {model}: {count}회")

    # 복잡한 추론 예시
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 복잡한 추론 예시")
    st.sidebar.markdown("""
    - "이 논리의 타당성을 분석해줘"
    - "다음 주장의 전제와 결론을 비판적으로 평가해줘" 
    - "이 복잡한 문제를 단계별로 추론해줘"
    - "A와 B 접근법의 장단점을 깊이 있게 비교분석해줘"
    """)

if __name__ == "__main__":
    main()
