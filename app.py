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

# ✅ 페이지 설정
st.set_page_config(
    page_title="🧠 하이브리드 멀티 LLM 시스템",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AdvancedAISystem:
    def __init__(self):
        self.setup_api_keys()
        self.setup_database()
        self.initialize_session_state()
        logger.info("고급 AI 시스템 초기화 완료")
    
    def setup_api_keys(self):
        """API 키 설정 - Streamlit Cloud Secrets 사용"""
        try:
            # Google Gemini
            if 'GEMINI_API_KEY' in st.secrets:
                genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
                self.gemini_available = True
                logger.info("Gemini API 설정 완료")
            else:
                self.gemini_available = False
                logger.warning("Gemini API 키 없음")
            
            # OpenRouter (Claude 등)
            self.openrouter_key = st.secrets.get('OPENROUTER_API_KEY', '')
            self.openrouter_available = bool(self.openrouter_key)
            if self.openrouter_available:
                logger.info("OpenRouter API 설정 완료")
            
            # DeepSeek
            self.deepseek_key = st.secrets.get('DEEPSEEK_API_KEY', '')
            self.deepseek_available = bool(self.deepseek_key)
            if self.deepseek_available:
                logger.info("DeepSeek API 설정 완료")
                
        except Exception as e:
            logger.error(f"API 키 설정 중 오류: {e}")
            st.error("API 키 설정 중 오류가 발생했습니다.")
    
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
                    tokens_used INTEGER
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    intent_type TEXT,
                    success_count INTEGER DEFAULT 0,
                    total_count INTEGER DEFAULT 0,
                    avg_response_time REAL,
                    last_used DATETIME
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
        
        if 'model_stats' not in st.session_state:
            st.session_state.model_stats = {}
        
        if 'intent_stats' not in st.session_state:
            st.session_state.intent_stats = {}

    def advanced_intent_analysis(self, user_input: str) -> Dict:
        """고급 의도 분석 시스템 - 강화된 버전"""
        intent_keywords = {
            'complex_reasoning': [
                '논리', '추론', '분석', '비교', '평가', '판단', '결론', '가정',
                '전제', '논증', '타당성', '비판적', '사고', '이유', '근거',
                '복잡한', '난이도', '심층', '다단계', '종합', '통합', '철학',
                '모순', '논쟁', '주장', '반박', '입증', '체계적'
            ],
            'technical': [
                '코드', '프로그래밍', '알고리즘', '개발', '설계', '파이썬', '자바', 
                '함수', '클래스', '디버깅', '컴파일', '인터페이스', '데이터베이스',
                'API', 'JSON', 'XML', 'HTML', 'CSS', 'JavaScript', '리팩토링'
            ],
            'creative': [
                '작성', '생성', '만들', '글쓰기', '시', '이야기', '창의', '소설',
                '아이디어', '기획', '콘텐츠', '스토리', '플롯', '캐릭터', '시나리오'
            ],
            'mathematical': [
                '계산', '수학', '공식', '방정식', '통계', '확률', '미분', '적분',
                '수치', '삼각함수', '기하', '대수', '수열', '행렬', '벡터'
            ],
            'research': [
                '연구', '논문', '참고문헌', '학술', '이론', '실험', '데이터', '조사',
                '분석', '결과', '가설', '방법론', '참고', '인용', '문헌'
            ],
            'factual': [
                '뭐야', '무엇', '알려줘', '정보', '사실', '정의', '설명', '개념',
                '역사', '백과사전', '사전', '의미'
            ],
            'analytical': [
                '분석', '비교', '장단점', '왜', '어떻게', '원인', '결과', '해석',
                '평가', '의견', '관점', '시사점', '함의'
            ],
            'casual': [
                '안녕', '하이', '잘지내', '고마워', '반가워', '헤이', '굿', '좋아',
                'ㅋㅋ', 'ㅎㅎ', 'ㅠㅠ', 'ㅜㅜ', '하루', '기분'
            ]
        }
        
        # 의도 점수 계산
        intent_scores = {}
        user_lower = user_input.lower()
        
        for intent, keywords in intent_keywords.items():
            score = sum(10 for keyword in keywords if keyword in user_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # 고급 복잡도 분석
        word_count = len(user_input.split())
        char_count = len(user_input)
        
        # 복잡도 지표
        has_complex_words = any(word in user_lower for word in [
            '분석', '비교', '평가', '논리', '추론', '전제', '결론', '체계적', '다단계'
        ])
        has_technical_terms = any(word in user_lower for word in [
            '알고리즘', '함수', '클래스', '데이터베이스', 'API'
        ])
        has_research_terms = any(word in user_lower for word in [
            '연구', '논문', '이론', '가설', '방법론'
        ])
        
        # 복잡도 점수 계산
        complexity_score = 0
        complexity_score += min(word_count // 3, 20)  # 단어 수 기여
        complexity_score += 15 if has_complex_words else 0
        complexity_score += 10 if has_technical_terms else 0
        complexity_score += 10 if has_research_terms else 0
        
        # 복잡도 레벨 결정
        if complexity_score >= 40:
            complexity = 'very_high'
        elif complexity_score >= 25:
            complexity = 'high'
        elif complexity_score >= 15:
            complexity = 'medium'
        else:
            complexity = 'low'
        
        # 주요 의도 선택 (복잡한 추론 우선)
        primary_intent = 'general'
        if intent_scores:
            if 'complex_reasoning' in intent_scores and intent_scores['complex_reasoning'] >= 15:
                primary_intent = 'complex_reasoning'
            else:
                primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return {
            'primary_intent': primary_intent,
            'all_intents': list(intent_scores.keys()),
            'intent_scores': intent_scores,
            'complexity': complexity,
            'complexity_score': complexity_score,
            'word_count': word_count,
            'char_count': char_count,
            'is_complex': complexity in ['high', 'very_high']
        }

    def select_optimal_model(self, intent_analysis: Dict) -> Dict:
        """최적의 AI 모델 선택 - 개선된 버전"""
        
        # 기본 모델 매핑
        intent_model_mapping = {
            'complex_reasoning': {
                'primary': 'claude',
                'reason': '🧠 복잡한 논리/추론에는 Claude 3.5 Sonnet이 가장 우수',
                'backup': 'gemini_advanced',
                'specialization': '논리적 추론, 체계적 분석'
            },
            'technical': {
                'primary': 'deepseek',
                'reason': '💻 코드 및 기술적 문제 해결에는 DeepSeek V3가 최적화',
                'backup': 'gemini',
                'specialization': '프로그래밍, 알고리즘, 개발'
            },
            'mathematical': {
                'primary': 'gemini_advanced',
                'reason': '🧮 수학적/논리적 연산에는 Gemini 1.5 Pro가 강력',
                'backup': 'deepseek',
                'specialization': '수학, 계산, 공식'
            },
            'research': {
                'primary': 'gemini_advanced',
                'reason': '📚 방대한 텍스트/연구 분석에는 Gemini의 긴 컨텍스트 창이 유리',
                'backup': 'claude',
                'specialization': '연구, 논문, 학술 분석'
            },
            'analytical': {
                'primary': 'claude',
                'reason': '🔍 분석적 사고와 다각도 접근에는 Claude가 뛰어남',
                'backup': 'gemini_advanced',
                'specialization': '분석, 비교, 평가'
            },
            'creative': {
                'primary': 'claude',
                'reason': '🎨 자연스럽고 창의적인 작문은 Claude가 뛰어남',
                'backup': 'gemini',
                'specialization': '창의성, 아이디어, 글쓰기'
            },
            'factual': {
                'primary': 'gemini',
                'reason': '📖 사실적 정보 검색에는 Gemini의 정확도가 높음',
                'backup': 'claude',
                'specialization': '사실, 정보, 정의'
            },
            'general': {
                'primary': 'gemini',
                'reason': '⚡ 일반적인 질문에는 빠르고 효율적인 Gemini Flash 사용',
                'backup': 'claude',
                'specialization': '일반 대화, 기본 질문'
            }
        }
        
        # 복잡도가 매우 높으면 고성능 모델 우선
        primary_intent = intent_analysis['primary_intent']
        if intent_analysis['complexity'] == 'very_high':
            if primary_intent in ['technical', 'mathematical']:
                # 기술/수학 복잡 문제는 그대로 유지
                pass
            else:
                # 그 외 복잡한 문제는 복잡한 추론으로 취급
                primary_intent = 'complex_reasoning'
        
        model_choice = intent_model_mapping.get(primary_intent, intent_model_mapping['general'])
        
        # 모델 가용성 체크 및 폴백 로직
        selected_model = model_choice['primary']
        original_reason = model_choice['reason']
        
        # 가용성 체크 및 조정
        if selected_model == 'claude' and not self.openrouter_available:
            selected_model = model_choice['backup']
            model_choice['reason'] = f"🚫 Claude 사용 불가 → {original_reason} (백업 모델 사용)"
        
        if selected_model == 'deepseek' and not self.deepseek_available:
            selected_model = 'gemini' if self.gemini_available else 'claude' if self.openrouter_available else 'none'
            model_choice['reason'] = f"🚫 DeepSeek 사용 불가 → {original_reason} (백업 모델 사용)"
        
        if 'gemini' in selected_model and not self.gemini_available:
            if selected_model == 'gemini_advanced':
                selected_model = 'claude' if self.openrouter_available else 'deepseek' if self.deepseek_available else 'none'
            else:
                selected_model = 'claude' if self.openrouter_available else 'deepseek' if self.deepseek_available else 'none'
            model_choice['reason'] = f"🚫 Gemini 사용 불가 → {original_reason} (백업 모델 사용)"
        
        # 사용 가능한 모델이 없는 경우
        if selected_model == 'none':
            model_choice['reason'] = "❌ 사용 가능한 AI 모델이 없습니다. API 키를 설정해주세요."
        
        model_choice['primary'] = selected_model
        return model_choice

    def call_gemini_api(self, prompt: str, intent: str, is_advanced: bool = False) -> Dict:
        """Gemini API 호출 - 안정적인 버전"""
        if not self.gemini_available:
            return {'success': False, 'error': 'Gemini API를 사용할 수 없습니다.'}
        
        try:
            start_time = time.time()
            
            # 모델 선택
            if is_advanced:
                model_name = 'gemini-1.5-pro'
            else:
                model_name = 'gemini-1.5-flash'
            
            # 의도별 특화 프롬프트
            reasoning_prompts = {
                'complex_reasoning': """
                당신은 논리적 추론 전문가입니다. 체계적으로 접근해주세요:

                💭 **사고 프레임워크**
                1. 핵심 문제 식별 및 구조화
                2. 명시적/암묵적 전제 분석  
                3. 다각도 논리 전개
                4. 결론 도출 및 검증
                5. 함의와 한계 명시

                질문: {prompt}
                """,
                'technical': """
                당신은 수석 소프트웨어 엔지니어입니다:

                🔧 **개발 방법론**
                1. 요구사항 명확화
                2. 아키텍처 설계
                3. 효율적 구현
                4. 에러 처리 및 최적화
                5. 사용법 설명

                질문: {prompt}
                """,
                'mathematical': """
                당신은 수학 전문가입니다:

                🧮 **문제 해결 접근법**
                1. 문제 재정의 및 변수 설정
                2. 관련 이론/공식 적용
                3. 단계별 계산 과정
                4. 결과 검증
                5. 일반화 가능성 탐구

                질문: {prompt}
                """,
                'research': """
                당신은 연구 분석 전문가입니다:

                📊 **학문적 분석**
                1. 연구 질문 명확화
                2. 방법론 적절성 평가
                3. 증거 수준 분석
                4. 결론 도출 및 함의
                5. 한계점과 향후 방향

                질문: {prompt}
                """
            }
            
            system_prompt = reasoning_prompts.get(
                intent, 
                "명확하고 체계적으로 답변해주세요: {prompt}"
            ).format(prompt=prompt)
            
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(system_prompt)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'content': response.text,
                'model': f"Google {model_name}",
                'processing_time': processing_time,
                'tokens': len(system_prompt + response.text) // 4
            }
            
        except Exception as e:
            logger.error(f"Gemini API 오류: {e}")
            return {'success': False, 'error': f'Gemini API 오류: {str(e)}'}

    def call_openrouter_api(self, prompt: str, intent: str) -> Dict:
        """OpenRouter API 호출 - Claude 등"""
        if not self.openrouter_available:
            return {'success': False, 'error': 'OpenRouter API를 사용할 수 없습니다.'}
        
        try:
            start_time = time.time()
            
            # 의도별 최적 모델 선택
            intent_models = {
                'complex_reasoning': 'anthropic/claude-3.5-sonnet',
                'research': 'anthropic/claude-3.5-sonnet', 
                'analytical': 'anthropic/claude-3.5-sonnet',
                'creative': 'anthropic/claude-3.5-sonnet',
                'technical': 'google/gemini-2.0-flash',
                'mathematical': 'google/gemini-2.0-flash',
                'general': 'google/gemini-2.0-flash'
            }
            
            selected_model = intent_models.get(intent, 'anthropic/claude-3.5-sonnet')
            
            # Claude 특화 프롬프트
            claude_prompts = {
                'complex_reasoning': """
                <thinking_framework>
                당신은 복잡한 문제 해결 전문가입니다. 심층적 분석을 수행해주세요:
                
                1. 문제 구조화: 핵심 이슈와 하위 문제 분해
                2. 다중 관점: 다양한 렌즈를 통한 접근
                3. 연역적 추론: 일반 원리에서 특수 결론 도출  
                4. 귀납적 일반화: 구체적 사례에서 패턴 발견
                5. 비판적 검토: 가정과 결론의 타당성 평가
                </thinking_framework>

                질문: {prompt}
                """,
                'research': """
                <research_methodology>
                당신은 연구 분석 전문가입니다. 학문적 엄밀성을 유지해주세요:
                
                1. 문헌 검토: 기존 연구와의 연계성 분석
                2. 방법론 검증: 분석 접근법의 적절성 평가
                3. 증거 수준: 주장의 근거 강도 판단
                4. 함의 도출: 연구 결과의 실제적 의미
                5. 한계 인식: 분석의 제한점 명시
                </research_methodology>

                질문: {prompt}
                """
            }
            
            if selected_model.startswith('anthropic/claude'):
                base_prompt = claude_prompts.get(intent, "심도 있게 분석하고 체계적으로 답변해주세요: {prompt}")
                final_prompt = base_prompt.format(prompt=prompt)
            else:
                final_prompt = prompt
            
            data = {
                "model": selected_model,
                "messages": [{"role": "user", "content": final_prompt}],
                "max_tokens": 4000,
                "temperature": 0.3
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://streamlit.io",
                    "X-Title": "AI Orchestrator"
                },
                json=data,
                timeout=60
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                tokens = result.get('usage', {}).get('total_tokens', 0)
                
                model_name = "Claude 3.5 Sonnet" if "claude" in selected_model else "Gemini 2.0 Flash"
                
                return {
                    'success': True,
                    'content': content,
                    'model': model_name,
                    'processing_time': processing_time,
                    'tokens': tokens
                }
            else:
                return {
                    'success': False, 
                    'error': f'OpenRouter API 오류: {response.status_code} - {response.text}'
                }
                
        except Exception as e:
            logger.error(f"OpenRouter 연결 오류: {e}")
            return {'success': False, 'error': f'OpenRouter 연결 오류: {str(e)}'}

    def call_deepseek_api(self, prompt: str, intent: str) -> Dict:
        """DeepSeek API 호출"""
        if not self.deepseek_available:
            return {'success': False, 'error': 'DeepSeek API를 사용할 수 없습니다.'}
        
        try:
            start_time = time.time()
            
            # DeepSeek 특화 프롬프트
            deepseek_prompts = {
                'technical': """
                [코딩 전문가 모드]
                당신은 수석 개발자입니다. 다음 원칙을 따라주세요:
                
                1. 효율적이고 읽기 쉬운 코드 작성
                2. 에러 처리와 예외 상황 고려
                3. 모범 사례와 패턴 적용
                4. 상세한 주석과 설명 제공
                5. 확장성과 유지보수성 고려
                
                질문: {prompt}
                """,
                'mathematical': """
                [수학 전문가 모드] 
                체계적인 문제 해결:
                
                1. 문제 이해 및 변수 정의
                2. 관련 공식/알고리즘 적용
                3. 단계별 계산 과정
                4. 결과 검증 및 설명
                5. 실용적 응용 제시
                
                질문: {prompt}
                """
            }
            
            system_prompt = deepseek_prompts.get(
                intent, 
                "명확하고 정확하게 답변해주세요: {prompt}"
            ).format(prompt=prompt)
            
            data = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": system_prompt}],
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
        """지능형 모델 오케스트레이션"""
        start_time = time.time()
        
        # 1. 고급 의도 분석
        intent_analysis = self.advanced_intent_analysis(user_input)
        
        # 2. 최적 모델 선택
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
        elif selected_model == 'gemini_advanced':
            response = self.call_gemini_api(user_input, intent_analysis['primary_intent'], True)
            if response['success']:
                responses['gemini_advanced'] = response
        elif selected_model == 'gemini':
            response = self.call_gemini_api(user_input, intent_analysis['primary_intent'], False)
            if response['success']:
                responses['gemini'] = response
        
        # 4. 기본 모델 실패 시 백업 모델 시도
        if not responses:
            backup_model = model_choice.get('backup')
            if backup_model == 'claude' and self.openrouter_available:
                response = self.call_openrouter_api(user_input, intent_analysis['primary_intent'])
                if response['success']:
                    responses['claude'] = response
                    selected_model = 'claude'
                    model_choice['reason'] += " (주 모델 실패 → 백업 사용)"
            elif backup_model == 'gemini' and self.gemini_available:
                response = self.call_gemini_api(user_input, intent_analysis['primary_intent'], False)
                if response['success']:
                    responses['gemini'] = response
                    selected_model = 'gemini'
                    model_choice['reason'] += " (주 모델 실패 → 백업 사용)"
            elif backup_model == 'deepseek' and self.deepseek_available:
                response = self.call_deepseek_api(user_input, intent_analysis['primary_intent'])
                if response['success']:
                    responses['deepseek'] = response
                    selected_model = 'deepseek'
                    model_choice['reason'] += " (주 모델 실패 → 백업 사용)"
        
        # 5. 최종 응답 선택
        final_response = self.get_final_response(responses)
        total_processing_time = time.time() - start_time
        
        result = {
            'final_response': final_response,
            'selected_model': selected_model,
            'model_reason': model_choice['reason'],
            'model_specialization': model_choice.get('specialization', '일반'),
            'intent_analysis': intent_analysis,
            'processing_time': total_processing_time,
            'success': bool(responses)
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
            # 첫 번째 성공한 응답 사용
            for response in responses.values():
                if response['success']:
                    return response['content']
        
        # 모든 모델 실패 시 폴백 응답
        return """
        🤖 **AI 서비스에 일시적으로 접속할 수 없습니다**
        
        가능한 원인:
        - API 키가 설정되지 않았거나 만료됨
        - 네트워크 연결 문제
        - 서비스 일시적 중단
        
        ✅ **해결 방법**:
        1. Streamlit Cloud Secrets에서 API 키 확인
        2. 잠시 후 다시 시도
        3. 질문을 더 간단하게 재구성
        """

    def save_conversation(self, user_input: str, result: Dict):
        """대화 저장"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO conversations 
                (session_id, user_message, bot_response, model_used, intent_detected, processing_time, tokens_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                st.session_state.user_id,
                user_input,
                result.get('content', ''),
                result.get('model_name', ''),
                result['intent_analysis']['primary_intent'],
                result['processing_time'],
                result.get('tokens_used', 0)
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"대화 저장 오류: {e}")

    def display_advanced_sidebar(self):
        """고급 사이드바 표시"""
        with st.sidebar:
            st.title("🎯 AI 오케스트레이터")
            
            # 시스템 상태
            st.subheader("🔧 시스템 상태")
            col1, col2, col3 = st.columns(3)
            col1.metric("Gemini", "✅" if self.gemini_available else "❌")
            col2.metric("Claude", "✅" if self.openrouter_available else "❌")
            col3.metric("DeepSeek", "✅" if self.deepseek_available else "❌")
            
            st.markdown("---")
            
            # 모델 특성 안내
            st.subheader("🏆 모델 특기")
            st.markdown("""
            **Claude 3.5 Sonnet**
            - 논리적 추론, 복잡한 분석
            - 창의적 작문, 뉘앙스 이해
            
            **Gemini 1.5 Pro/Flash**  
            - 긴 컨텍스트 처리
            - 멀티모달, 빠른 응답
            - 수학적 문제 해결
            
            **DeepSeek V3**
            - 코딩, 알고리즘 최적화
            - 수학적 계산
            - 가성비 우수
            """)
            
            st.markdown("---")
            
            # 사용 팁
            st.subheader("💡 사용 팁")
            st.markdown("""
            - **복잡한 추론**: Claude 추천
            - **코딩 문제**: DeepSeek 추천  
            - **연구 분석**: Gemini Pro 추천
            - **빠른 응답**: Gemini Flash 추천
            """)
            
            if st.button("🗑️ 대화 기록 지우기", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    def display_chat_interface(self):
        """채팅 인터페이스 표시"""
        st.title("🧠 하이브리드 멀티 LLM 시스템")
        st.markdown("**질문을 분석하여 Claude, Gemini, DeepSeek 중 최적의 모델을 자동 선택합니다**")
        
        # 채팅 기록
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # 메타정보 표시
                if message["role"] == "assistant" and "metadata" in message:
                    with st.expander("🔍 AI 분석 정보"):
                        metadata = message['metadata']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**모델**: {metadata['model_name']}")
                            st.write(f"**의도**: {metadata['intent_analysis']['primary_intent']}")
                        with col2:
                            st.write(f"**복잡도**: {metadata['intent_analysis']['complexity']}")
                            st.write(f"**처리시간**: {metadata['response_time']:.2f}s")
                        
                        st.info(f"**선택 이유**: {metadata['model_reason']}")
                        
                        # 의도 점수 시각화
                        if 'intent_scores' in metadata['intent_analysis']:
                            st.write("**의도 분석**:")
                            for intent, score in metadata['intent_analysis']['intent_scores'].items():
                                progress = min(score / 100, 1.0)
                                st.progress(progress, text=f"{intent} ({score}점)")
        
        # 사용자 입력
        if prompt := st.chat_input("질문을 입력하세요..."):
            # 사용자 메시지
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # AI 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("🧠 질문 분석 및 최적 모델 선택 중..."):
                    result = self.intelligent_model_orchestration(prompt)
                
                if result['success']:
                    st.markdown(result['content'])
                    
                    # 실시간 분석 정보
                    with st.expander("🎯 실시간 라우팅 정보", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("선택 모델", result['model_name'])
                        with col2:
                            st.metric("감지 의도", result['intent_analysis']['primary_intent'])
                        with col3:
                            st.metric("복잡도", result['intent_analysis']['complexity'])
                        with col4:
                            st.metric("토큰", f"{result['tokens_used']}")
                        
                        st.info(f"**선택 이유**: {result['model_reason']}")
                        st.success(f"**전문 분야**: {result['model_specialization']}")
                        
                        # 처리 시간
                        st.write(f"**총 처리 시간**: {result['processing_time']:.2f}초")
                        
                else:
                    st.error("❌ AI 응답 생성에 실패했습니다.")
                    st.info(result['final_response'])
            
            # 세션에 메시지 저장
            if result['success']:
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
            
            st.rerun()

def main():
    # 시스템 초기화
    ai_system = AdvancedAISystem()
    
    # 사이드바 표시
    ai_system.display_advanced_sidebar()
    
    # 메인 채팅 인터페이스
    ai_system.display_chat_interface()
    
    # 푸터
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🧠 하이브리드 멀티 LLM 시스템 • 지능형 모델 오케스트레이션 • 실시간 의도 분석"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()