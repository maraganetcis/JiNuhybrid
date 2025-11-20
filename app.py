import streamlit as st
import asyncio
import google.generativeai as genai
import requests
from typing import Dict, List
import json
import time
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="JiNu hybrid AI",
    page_icon="⌘",
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
        
        if 'usage_stats' not in st.session_state:
            st.session_state.usage_stats = {
                'total_queries': 0,
                'total_tokens': 0,
                'model_usage': {},
                'cost_estimate': 0.0
            }
    
    def advanced_intent_analysis(self, user_input: str) -> Dict:
        """고급 의도 분석 시스템"""
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
        
        # 복잡도 분석
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
        if 'complex_reasoning' in intent_scores:
            primary_intent = 'complex_reasoning'
        else:
            primary_intent = max(intent_scores, key=intent_scores.get()) if intent_scores else 'general'
        
        return {
            'primary_intent': primary_intent,
            'all_intents': list(intent_scores.keys()),
            'intent_scores': intent_scores,
            'complexity': complexity,
            'word_count': word_count,
            'is_complex': complexity in ['high', 'very_high']
        }
    
    def select_optimal_model(self, intent_analysis: Dict) -> Dict:
        """최적의 AI 모델 선택"""
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
            'creative': {
                'primary': 'claude',
                'reason': '🎨 창의적 사고에는 Claude의 유연성이 좋음',
                'backup': 'gemini'
            },
            'general': {
                'primary': 'gemini',
                'reason': '⚡ 일반 질문에는 Gemini의 빠른 응답이 적합',
                'backup': 'deepseek'
            }
        }
        
        # 복잡도가 매우 높으면 복잡한 추론 모델 강제 사용
        if intent_analysis['complexity'] == 'very_high':
            primary_intent = 'complex_reasoning'
        else:
            primary_intent = intent_analysis['primary_intent']
        
        model_choice = intent_model_mapping.get(primary_intent, intent_model_mapping['general'])
        return model_choice
    
    def call_ai_model(self, prompt: str, model_type: str, intent: str) -> Dict:
        """AI 모델 호출"""
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
            """
        }
        
        specialized_prompt = reasoning_prompts.get(
            intent, 
            "체계적으로 분석하고 논리적으로 답변해주세요: {prompt}"
        ).format(prompt=prompt)
        
        try:
            if model_type in ['gemini', 'gemini_advanced'] and self.gemini_available:
                return self._call_gemini(specialized_prompt, model_type)
            elif model_type == 'claude' and self.openrouter_available:
                return self._call_claude(specialized_prompt)
            elif model_type == 'deepseek' and self.deepseek_available:
                return self._call_deepseek(specialized_prompt)
            else:
                return {'error': '사용 가능한 모델이 없습니다.'}
                
        except Exception as e:
            return {'error': f'모델 호출 중 오류: {str(e)}'}
    
    def _call_gemini(self, prompt: str, model_type: str) -> Dict:
        """Gemini 모델 호출"""
        try:
            if model_type == 'gemini_advanced':
                model = genai.GenerativeModel('gemini-2.5-pro')
            else:
                model = genai.GenerativeModel('gemini-2.5-flash')
            
            response = model.generate_content(prompt)
            return {
                'success': True,
                'content': response.text,
                'model': 'Gemini ' + ('Advanced' if model_type == 'gemini_advanced' else 'Flash'),
                'tokens': len(prompt.split()) + len(response.text.split())
            }
        except Exception as e:
            return {'error': f'Gemini 오류: {str(e)}'}
    
    def _call_claude(self, prompt: str) -> Dict:
        """Claude 모델 호출"""
        try:
            data = {
                "model": "anthropic/claude-3.5-sonnet",
                "messages": [{"role": "user", "content": prompt}],
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
                return {'error': f'Claude API 오류: {response.status_code}'}
                
        except Exception as e:
            return {'error': f'Claude 호출 오류: {str(e)}'}
    
    def _call_deepseek(self, prompt: str) -> Dict:
        """DeepSeek 모델 호출"""
        try:
            data = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
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
                return {'error': f'DeepSeek API 오류: {response.status_code}'}
                
        except Exception as e:
            return {'error': f'DeepSeek 호출 오류: {str(e)}'}
    
    def update_usage_stats(self, model: str, tokens: int):
        """사용 통계 업데이트"""
        st.session_state.usage_stats['total_queries'] += 1
        st.session_state.usage_stats['total_tokens'] += tokens
        
        if model not in st.session_state.usage_stats['model_usage']:
            st.session_state.usage_stats['model_usage'][model] = 0
        st.session_state.usage_stats['model_usage'][model] += 1
        
        # 간단한 비용 추정 (토큰당 평균 $0.00001)
        st.session_state.usage_stats['cost_estimate'] += tokens * 0.00001

def main():
    st.title("🚀 하이브리드 AI 어시스턴트")
    st.markdown("질문 유형에 따라 최적의 AI 모델이 자동으로 선택됩니다!")
    
    # 사이드바
    with st.sidebar:
        st.header("🔧 설정")
        budget_mode = st.selectbox("비용 모드", ["비용 효율", "성능 최대"])
        
        st.header("📊 사용 통계")
        if 'usage_stats' in st.session_state:
            stats = st.session_state.usage_stats
            st.metric("총 질문", stats['total_queries'])
            st.metric("총 토큰", f"{stats['total_tokens']:,}")
            st.metric("예상 비용", f"${stats['cost_estimate']:.4f}")
            
            if stats['model_usage']:
                st.subheader("모델 사용량")
                for model, count in stats['model_usage'].items():
                    st.write(f"- {model}: {count}회")
    
    # AI 시스템 초기화
    ai_system = StreamlitAISystem()
    
    # 채팅 인터페이스
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "model" in message:
                st.caption(f"모델: {message['model']}")
    
    # 사용자 입력
    if prompt := st.chat_input("무엇이 궁금하신가요?"):
        # 사용자 메시지 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 의도 분석 및 모델 선택
        with st.spinner("최적의 AI 모델을 선택 중..."):
            intent_analysis = ai_system.advanced_intent_analysis(prompt)
            model_choice = ai_system.select_optimal_model(intent_analysis)
            
            # 비용 효율 모드에서는 일부 모델 변경
            if budget_mode == "비용 효율" and model_choice['primary'] == 'claude':
                model_choice['primary'] = model_choice['backup']
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner(f"{model_choice['primary']} 모델로 응답 생성 중..."):
                response = ai_system.call_ai_model(
                    prompt, 
                    model_choice['primary'], 
                    intent_analysis['primary_intent']
                )
                
                if 'error' in response:
                    st.error(response['error'])
                else:
                    # 응답 표시
                    st.markdown(response['content'])
                    st.caption(f"모델: {response['model']} | 토큰: {response['tokens']}")
                    
                    # 의도 분석 정보
                    with st.expander("의도 분석 결과 보기"):
                        st.json(intent_analysis)
                    
                    # 통계 업데이트
                    ai_system.update_usage_stats(response['model'], response['tokens'])
                    
                    # 세션 상태에 메시지 저장
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response['content'],
                        "model": response['model']
                    })

if __name__ == "__main__":
    main()
