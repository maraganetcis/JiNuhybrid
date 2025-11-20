import os
import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import google.generativeai as genai
import requests
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    GOOGLE = "google"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    OPENAI = "openai"

@dataclass
class ModelConfig:
    provider: ModelProvider
    model_name: str
    api_key: str
    base_url: Optional[str] = None
    cost_per_input: float = 0.0
    cost_per_output: float = 0.0

class HybridAISystem:
    def __init__(self):
        # API 키 초기화
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.openrouter_key = os.getenv('OPENROUTER_API_KEY')
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        self.deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        
        # 모델 구성
        self.models = {
            'gemini_flash': ModelConfig(
                provider=ModelProvider.GOOGLE,
                model_name='gemini-2.0-flash',
                api_key=self.google_api_key,
                cost_per_input=0.075,  # $0.75 per 1M tokens
                cost_per_output=0.30   # $3.00 per 1M tokens
            ),
            'gemini_pro': ModelConfig(
                provider=ModelProvider.GOOGLE,
                model_name='gemini-1.5-pro',
                api_key=self.google_api_key,
                cost_per_input=3.75,   # $7.5 per 1M tokens
                cost_per_output=15.00  # $15.0 per 1M tokens
            ),
            'claude_sonnet': ModelConfig(
                provider=ModelProvider.OPENROUTER,
                model_name='anthropic/claude-3.5-sonnet',
                api_key=self.openrouter_key,
                base_url='https://openrouter.ai/api/v1',
                cost_per_input=3.0,    # $3.0 per 1M tokens
                cost_per_output=15.0   # $15.0 per 1M tokens
            ),
            'deepseek_v3': ModelConfig(
                provider=ModelProvider.DEEPSEEK,
                model_name='deepseek-chat',
                api_key=self.deepseek_key,
                base_url='https://api.deepseek.com/v1',
                cost_per_input=0.14,   # $1.4 per 1M tokens
                cost_per_output=0.28   # $2.8 per 1M tokens
            ),
            'llama_70b': ModelConfig(
                provider=ModelProvider.OPENROUTER,
                model_name='meta-llama/llama-3-70b-instruct',
                api_key=self.openrouter_key,
                base_url='https://openrouter.ai/api/v1',
                cost_per_input=0.59,   # $0.59 per 1M tokens
                cost_per_output=0.79   # $0.79 per 1M tokens
            )
        }
        
        # 제공자별 클라이언트 초기화
        if self.google_api_key:
            genai.configure(api_key=self.google_api_key)
        
        self.openai_client = OpenAI(api_key=self.deepseek_key) if self.deepseek_key else None
        self.anthropic_client = Anthropic(api_key=self.anthropic_key) if self.anthropic_key else None
        
        self.available_models = self._check_available_models()
    
    def _check_available_models(self) -> Dict:
        """사용 가능한 모델 확인"""
        available = {}
        
        # Google 모델 확인
        if self.google_api_key:
            available['gemini_flash'] = self.models['gemini_flash']
            available['gemini_pro'] = self.models['gemini_pro']
        
        # OpenRouter 모델 확인
        if self.openrouter_key:
            available['claude_sonnet'] = self.models['claude_sonnet']
            available['llama_70b'] = self.models['llama_70b']
        
        # DeepSeek 모델 확인
        if self.deepseek_key:
            available['deepseek_v3'] = self.models['deepseek_v3']
        
        logger.info(f"Available models: {list(available.keys())}")
        return available

    def advanced_intent_analysis(self, user_input: str) -> Dict:
        """고급 의도 분석 시스템 - 복잡한 추론 추가"""
        intent_keywords = {
            'complex_reasoning': [
                '논리', '추론', '분석', '비교', '평가', '판단', '결론', '가정',
                '전제', '논증', '타당성', '비판적', '사고', '이유', '근거',
                '복잡한', '난이도', '심층', '다단계', '종합', '통합', '논문',
                '연구', '실험', '가설', '검증'
            ],
            'technical': [
                '코드', '프로그래밍', '알고리즘', '개발', '설계', '파이썬', 
                '자바', '자바스크립트', '리액트', 'vue', 'html', 'css',
                '디버그', '버그', '오류', '컴파일', '함수', '클래스'
            ],
            'creative': [
                '작성', '생성', '만들', '글쓰기', '시', '이야기', '창의',
                '아이디어', '기획', '콘텐츠', '마케팅', '광고', '브랜드'
            ],
            'mathematical': [
                '계산', '수학', '공식', '방정식', '통계', '확률', '미분',
                '적분', '함수', '기하', '대수', '삼각함수', '행렬'
            ],
            'research': [
                '연구', '논문', '참고문헌', '학술', '이론', '실험', '데이터',
                '분석', '통계', '설문', '조사', '리서치'
            ],
            'factual': [
                '뭐야', '무엇', '알려줘', '정보', '사실', '정의', '의미',
                '개념', '원리', '방법'
            ],
            'casual': [
                '안녕', '하이', '잘지내', '고마워', '반가워', 'ㅎㅎ', 'ㅋㅋ'
            ]
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
        sentence_count = user_input.count('.') + user_input.count('?') + user_input.count('!')
        
        has_complex_indicators = any(word in user_lower for word in [
            '분석', '비교', '평가', '논리', '추론', '전제', '결론', '연구'
        ])
        
        # 복잡도 점수 계산
        complexity_score = word_count * 0.5 + sentence_count * 2
        
        if complexity_score > 25 or has_complex_indicators:
            complexity = 'very_high'
        elif complexity_score > 15:
            complexity = 'high'
        elif complexity_score > 8:
            complexity = 'medium'
        else:
            complexity = 'low'
        
        # 주요 의도 선택 (복잡한 추론 우선)
        if 'complex_reasoning' in intent_scores:
            primary_intent = 'complex_reasoning'
        elif intent_scores:
            primary_intent = max(intent_scores, key=intent_scores.get())
        else:
            primary_intent = 'general'
        
        return {
            'primary_intent': primary_intent,
            'all_intents': list(intent_scores.keys()),
            'intent_scores': intent_scores,
            'complexity': complexity,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'complexity_score': complexity_score,
            'is_complex': complexity in ['high', 'very_high']
        }

    def select_optimal_model(self, intent_analysis: Dict, budget_conscious: bool = True) -> Dict:
        """최적의 AI 모델 선택 - 비용 효율성 고려"""
        
        # 비용 효율적인 모델 매핑
        cost_effective_mapping = {
            'complex_reasoning': {
                'primary': 'claude_sonnet',
                'reason': '🧠 복잡한 추론에는 Claude 3.5 Sonnet이 가장 우수함',
                'backup': 'gemini_pro'
            },
            'technical': {
                'primary': 'gemini_flash',
                'reason': '🔧 기술/코드 관련 질문에는 Gemini Flash가 빠르고 정확함',
                'backup': 'deepseek_v3'
            },
            'mathematical': {
                'primary': 'gemini_flash',
                'reason': '🧮 수학적 문제에는 Gemini Flash의 정확도가 높음',
                'backup': 'deepseek_v3'
            },
            'research': {
                'primary': 'claude_sonnet',
                'reason': '📊 연구/학술 분석에는 Claude의 깊은 이해력이 적합',
                'backup': 'gemini_pro'
            },
            'creative': {
                'primary': 'claude_sonnet',
                'reason': '🎨 창의적 작업에는 Claude의 유연성이 좋음',
                'backup': 'gemini_pro'
            },
            'general': {
                'primary': 'gemini_flash',
                'reason': '⚡ 일반 질문에는 Gemini Flash의 빠른 응답이 적합',
                'backup': 'deepseek_v3'
            },
            'factual': {
                'primary': 'deepseek_v3',
                'reason': '💰 사실 확인에는 가장 저렴한 DeepSeek이 효율적',
                'backup': 'gemini_flash'
            }
        }
        
        # 고성능 모델 매핑 (비용 덜 중요)
        performance_mapping = {
            'complex_reasoning': {
                'primary': 'claude_sonnet',
                'reason': '🧠 최고 수준의 추론 성능을 위한 Claude 3.5 Sonnet',
                'backup': 'gemini_pro'
            },
            'technical': {
                'primary': 'gemini_pro',
                'reason': '🔧 정밀한 기술 작업에는 Gemini Pro가 적합',
                'backup': 'claude_sonnet'
            },
            # ... 나머지 의도들도 유사하게 구성
        }
        
        # 매핑 선택
        model_mapping = cost_effective_mapping if budget_conscious else performance_mapping
        
        # 복잡도가 매우 높으면 복잡한 추론 모델 강제 사용
        if intent_analysis['complexity'] == 'very_high':
            primary_intent = 'complex_reasoning'
        else:
            primary_intent = intent_analysis['primary_intent']
        
        model_choice = model_mapping.get(primary_intent, model_mapping['general'])
        
        # 선택된 모델이 사용 가능한지 확인
        if model_choice['primary'] not in self.available_models:
            model_choice['primary'] = model_choice['backup']
        
        return model_choice

    async def call_model(self, prompt: str, model_config: ModelConfig, intent: str) -> Dict:
        """모델 호출 - 비동기 처리"""
        
        reasoning_prompts = {
            'complex_reasoning': """
            당신은 논리적 추론 전문가입니다. 다음 단계로 체계적으로 접근해주세요:
            
            1. **문제 분석**: 핵심 요소와 주요 개념 파악
            2. **전제 확인**: 명시적/암묵적 가정 식별
            3. **논리 구조**: 주장과 근거의 연결 관계 분석
            4. **비판적 검토**: 타당성과 한계점 평가
            5. **결론 도출**: 체계적인 추론을 통한 최종 판단
            
            질문: {prompt}
            """,
            'technical': """
            당신은 소프트웨어 엔지니어링 전문가입니다. 다음을 확인해주세요:
            
            1. **요구사항 분석**: 기술적 요구사항 명확히 이해
            2. **아키텍처 설계**: 최적의 솔루션 구조 제안
            3. **코드 구현**: 실용적이고 효율적인 코드 작성
            4. **테스트 계획**: 검증 가능한 테스트 케이스 제시
            5. **성능 고려**: 확장성과 유지보수성 고려
            
            질문: {prompt}
            """,
            'mathematical': """
            당신은 수학적 문제 해결 전문가입니다. 단계별로 접근해주세요:
            
            1. **문제 이해**: 주어진 조건과 구해야 하는 값 정의
            2. **접근법 선택**: 적절한 공식/이론/알고리즘 선택
            3. **단계적 계산**: 체계적인 계산 과정 제시
            4. **결과 검증**: 답변의 타당성 확인
            5. **대안 제시**: 다른 접근법 가능성 탐색
            
            질문: {prompt}
            """
        }
        
        specialized_prompt = reasoning_prompts.get(
            intent, 
            "명확하고 체계적으로 답변해주세요: {prompt}"
        ).format(prompt=prompt)
        
        try:
            if model_config.provider == ModelProvider.GOOGLE:
                return await self._call_google_model(specialized_prompt, model_config)
            elif model_config.provider == ModelProvider.OPENROUTER:
                return await self._call_openrouter_model(specialized_prompt, model_config)
            elif model_config.provider == ModelProvider.DEEPSEEK:
                return await self._call_deepseek_model(specialized_prompt, model_config)
            else:
                raise ValueError(f"Unsupported provider: {model_config.provider}")
                
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'content': None,
                'tokens_used': 0,
                'cost': 0.0
            }

    async def _call_google_model(self, prompt: str, config: ModelConfig) -> Dict:
        """Google Gemini 모델 호출"""
        model = genai.GenerativeModel(config.model_name)
        response = model.generate_content(prompt)
        
        return {
            'success': True,
            'content': response.text,
            'tokens_used': len(prompt.split()) + len(response.text.split()),  # 추정치
            'cost': 0.0,  # 실제로는 정확한 토큰 수 계산 필요
            'model': config.model_name
        }

    async def _call_openrouter_model(self, prompt: str, config: ModelConfig) -> Dict:
        """OpenRouter 모델 호출"""
        data = {
            "model": config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "temperature": 0.3
        }
        
        response = requests.post(
            f"{config.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {config.api_key}"},
            json=data,
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            tokens_used = result.get('usage', {}).get('total_tokens', 0)
            
            return {
                'success': True,
                'content': content,
                'tokens_used': tokens_used,
                'cost': (tokens_used / 1000000) * config.cost_per_input,  # 단순화
                'model': config.model_name
            }
        else:
            raise Exception(f"OpenRouter API error: {response.status_code}")

    async def _call_deepseek_model(self, prompt: str, config: ModelConfig) -> Dict:
        """DeepSeek 모델 호출"""
        data = {
            "model": config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "temperature": 0.3
        }
        
        response = requests.post(
            f"{config.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {config.api_key}"},
            json=data,
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            tokens_used = result.get('usage', {}).get('total_tokens', 0)
            
            return {
                'success': True,
                'content': content,
                'tokens_used': tokens_used,
                'cost': (tokens_used / 1000000) * config.cost_per_input,
                'model': config.model_name
            }
        else:
            raise Exception(f"DeepSeek API error: {response.status_code}")

    async def process_query(self, user_input: str, budget_conscious: bool = True) -> Dict:
        """사용자 쿼리 처리 메인 함수"""
        
        # 1. 의도 분석
        intent_analysis = self.advanced_intent_analysis(user_input)
        logger.info(f"Intent analysis: {intent_analysis}")
        
        # 2. 모델 선택
        model_choice = self.select_optimal_model(intent_analysis, budget_conscious)
        logger.info(f"Model choice: {model_choice}")
        
        # 3. 모델 호출
        model_config = self.available_models[model_choice['primary']]
        response = await self.call_model(
            user_input, 
            model_config, 
            intent_analysis['primary_intent']
        )
        
        return {
            'intent_analysis': intent_analysis,
            'model_choice': model_choice,
            'response': response,
            'timestamp': asyncio.get_event_loop().time()
        }

# 사용 예제
async def main():
    system = HybridAISystem()
    
    test_queries = [
        "파이썬에서 다중 상속의 장단점과 MRO(Method Resolution Order)에 대해 설명해줘",
        "기후 변화가 경제 성장에 미치는 영향을 논리적으로 분석해줘",
        "안녕! 오늘 기분이 어때?",
        "미분방정식과 선형대수의 관계를 수학적으로 설명해줘"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        result = await system.process_query(query, budget_conscious=True)
        
        if result['response']['success']:
            print(f"Intent: {result['intent_analysis']['primary_intent']}")
            print(f"Model: {result['model_choice']['primary']}")
            print(f"Reason: {result['model_choice']['reason']}")
            print(f"Response: {result['response']['content'][:200]}...")
            print(f"Tokens used: {result['response']['tokens_used']}")
            print(f"Estimated cost: ${result['response']['cost']:.6f}")
        else:
            print(f"Error: {result['response']['error']}")

if __name__ == "__main__":
    asyncio.run(main())
