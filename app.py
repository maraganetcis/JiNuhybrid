import streamlit as st
import google.generativeai as genai
import sqlite3
import hashlib
import requests
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import sys

# ✅ 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ✅ 페이지 설정
st.set_page_config(
    page_title="JiNu Hybrid",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class UltimateHybridChatbot:
    def __init__(self):
        self.setup_apis()
        self.setup_database()
        self.setup_session_state()
        logger.info("하이브리드 챗봇 초기화 완료")
    
    def setup_apis(self):
        """모든 API 설정"""
        try:
            # Gemini API 설정
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key:
                genai.configure(api_key=gemini_key)
                self.gemini_available = True
                logger.info("Gemini API 설정 완료")
            else:
                self.gemini_available = False
                logger.warning("Gemini API 키 없음")
            
            # OpenRouter API 설정
            self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
            self.openrouter_available = bool(self.openrouter_key)
            
            if self.openrouter_available:
                logger.info("OpenRouter API 설정 완료")
            
        except Exception as e:
            logger.error(f"API 설정 중 오류: {e}")
            st.error("API 설정 중 오류가 발생했습니다.")
    
    def setup_database(self):
        """강력한 데이터베이스 초기화"""
        try:
            self.conn = sqlite3.connect('chatbot_website.db', check_same_thread=False)
            cursor = self.conn.cursor()
            
            # ✅ conversations 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_message TEXT,
                    bot_response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_used TEXT,
                    response_time REAL,
                    intent_detected TEXT
                )
            ''')
            
            # ✅ users 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    first_visit DATETIME,
                    last_visit DATETIME,
                    visit_count INTEGER DEFAULT 0,
                    total_messages INTEGER DEFAULT 0
                )
            ''')
            
            # ✅ performance 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT,
                    response_time REAL,
                    success BOOLEAN,
                    error_message TEXT
                )
            ''')
            
            self.conn.commit()
            logger.info("데이터베이스 초기화 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
            st.error("데이터베이스 초기화에 실패했습니다.")
    
    def setup_session_state(self):
        """세션 상태 초기화"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'user_id' not in st.session_state:
            st.session_state.user_id = hashlib.md5(
                str(datetime.now().timestamp()).encode()
            ).hexdigest()
            self.track_user_visit()
        
        if 'chat_start_time' not in st.session_state:
            st.session_state.chat_start_time = datetime.now()
    
    def track_user_visit(self):
        """사용자 방문 추적"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO users 
                (session_id, first_visit, last_visit, visit_count, total_messages)
                VALUES (?, 
                       COALESCE((SELECT first_visit FROM users WHERE session_id=?), datetime('now')), 
                       datetime('now'), 
                       COALESCE((SELECT visit_count FROM users WHERE session_id=?), 0) + 1,
                       COALESCE((SELECT total_messages FROM users WHERE session_id=?), 0)
                )
            ''', (st.session_state.user_id, st.session_state.user_id, 
                  st.session_state.user_id, st.session_state.user_id))
            
            self.conn.commit()
            logger.info(f"사용자 방문 기록: {st.session_state.user_id}")
            
        except Exception as e:
            logger.error(f"사용자 추적 오류: {e}")
    
    def detect_intent(self, user_input: str) -> Dict:
        """고급 의도 감지 시스템"""
        intents = {
            'creative': ['작성', '생성', '만들', '글쓰기', '시', '이야기', '창의'],
            'technical': ['코드', '프로그래밍', '알고리즘', '개발', '설계', '파이썬', '자바'],
            'factual': ['뭐야', '무엇', '알려줘', '정보', '사실', '정의', '의미'],
            'analytical': ['분석', '비교', '장단점', '왜', '어떻게', '원인', '결과'],
            'casual': ['안녕', '하이', '잘지내', '고마워', 'ㅋㅋ', 'ㅎㅎ', '반가워']
        }
        
        detected_intents = []
        for intent, keywords in intents.items():
            if any(keyword in user_input for keyword in keywords):
                detected_intents.append(intent)
        
        # 복잡도 분석
        complexity = 'high' if len(user_input.split()) > 10 else 'medium'
        complexity = 'low' if len(user_input.split()) < 3 else complexity
        
        return {
            'intents': detected_intents if detected_intents else ['general'],
            'complexity': complexity,
            'requires_context': len(user_input) > 20
        }
    
    def call_gemini_api(self, prompt: str, intent: str) -> str:
        """Gemini API 호출"""
        if not self.gemini_available:
            return "Gemini API를 사용할 수 없습니다. API 키를 확인해주세요."
        
        try:
            start_time = time.time()
            
            # 의도별 프롬프트 최적화
            intent_prompts = {
                'creative': "당신은 창의적인 작가입니다. 창의적이고 흥미로운 내용을 생성해주세요.",
                'technical': "당신은 전문 소프트웨어 엔지니어입니다. 정확하고 실용적인 답변을 제공해주세요.",
                'factual': "당신은 전문 백과사전입니다. 사실적이고 정확한 정보만 제공해주세요.",
                'analytical': "당신은 분석 전문가입니다. 깊이 있고 체계적인 분석을 제공해주세요.",
                'casual': "당신은 친근한 AI 어시스턴트입니다. 따뜻하고 자연스러운 대화를 나눠주세요."
            }
            
            system_prompt = intent_prompts.get(intent, "당신은 유용한 AI 어시스턴트입니다.")
            
            full_prompt = f"{system_prompt}\n\n사용자: {prompt}"
            
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(full_prompt)
            
            response_time = time.time() - start_time
            
            # 성능 로깅
            self.log_performance('gemini-2.5-flash', response_time, True, "")
            
            logger.info(f"Google API 호출 성공: {response_time:.2f}초")
            return response.text
            
        except Exception as e:
            error_msg = f"Gemini API 오류: {str(e)}"
            logger.error(error_msg)
            self.log_performance('gemini-2.5-flash', 0, False, error_msg)
            return f"Gemini API 호출 중 오류: {str(e)}"
    
    def call_openrouter_api(self, prompt: str) -> str:
        """OpenRouter API 호출 (백업)"""
        if not self.openrouter_available:
            return "OpenRouter API를 사용할 수 없습니다."
        
        try:
            start_time = time.time()
            
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://your-website.com",
                "X-Title": "AI Chatbot Website"
            }
            
            data = {
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                self.log_performance('llama-3.1-8b-instruct', response_time, True, "")
                logger.info(f"OpenRouter API 호출 성공: {response_time:.2f}초")
                return content
            else:
                error_msg = f"OpenRouter API 오류: {response.status_code}"
                logger.error(error_msg)
                self.log_performance('llama-3.1-8b-instruct', response_time, False, error_msg)
                return f"OpenRouter API 호출 실패: {response.status_code}"
                
        except Exception as e:
            error_msg = f"OpenRouter API 예외: {str(e)}"
            logger.error(error_msg)
            self.log_performance('llama-3.1-8b-instruct', 0, False, error_msg)
            return f"OpenRouter API 호출 중 오류: {str(e)}"
    
    def log_performance(self, model_name: str, response_time: float, success: bool, error_message: str = ""):
        """성능 로깅"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO performance (model_name, response_time, success, error_message)
                VALUES (?, ?, ?, ?)
            ''', (model_name, response_time, success, error_message))
            self.conn.commit()
        except Exception as e:
            logger.error(f"성능 로깅 오류: {e}")
    
    def hybrid_response_generation(self, user_input: str) -> Dict:
        """하이브리드 응답 생성 시스템"""
        start_time = time.time()
        intent_analysis = self.detect_intent(user_input)
        primary_intent = intent_analysis['intents'][0]
        
        responses = {}
        models_used = []
        
        # 1. 기본: Gemini API 시도
        if self.gemini_available:
            gemini_response = self.call_gemini_api(user_input, primary_intent)
            responses['gemini'] = gemini_response
            models_used.append('gemini')
        
        # 2. 백업: OpenRouter 시도
        if self.openrouter_available and ('gemini' not in responses or "오류" in responses['gemini']):
            openrouter_response = self.call_openrouter_api(user_input)
            responses['openrouter'] = openrouter_response
            models_used.append('openrouter')
        
        # 3. 최후의 백업: 로컬 응답
        if not responses or all("오류" in response for response in responses.values()):
            responses['fallback'] = self.generate_fallback_response(user_input, intent_analysis)
            models_used.append('fallback')
        
        total_time = time.time() - start_time
        
        return {
            'responses': responses,
            'models_used': models_used,
            'processing_time': total_time,
            'intent_analysis': intent_analysis,
            'final_response': self.select_best_response(responses, intent_analysis)
        }
    
    def generate_fallback_response(self, user_input: str, intent_analysis: Dict) -> str:
        """폴백 응답 생성"""
        fallback_responses = {
            'creative': "제가 창의적인 내용을 생성하려면 API 연결이 필요합니다. 현재는 연결에 문제가 있어 간단한 답변만 가능합니다.",
            'technical': "기술적인 질문에는 정확한 API 응답이 필요합니다. 현재 API 연결을 확인 중입니다.",
            'factual': "사실적인 정보를 제공하기 위해선 API 접근이 필요합니다. 잠시 후 다시 시도해주세요.",
            'general': "현재 AI 서비스에 일시적으로 접속할 수 없습니다. 잠시 후 다시 시도해주세요."
        }
        
        for intent in intent_analysis['intents']:
            if intent in fallback_responses:
                return fallback_responses[intent]
        
        return fallback_responses['general']
    
    def select_best_response(self, responses: Dict, intent_analysis: Dict) -> str:
        """최적의 응답 선택"""
        # Gemini 응답이 있으면 우선 사용
        if 'gemini' in responses and responses['gemini'] and "오류" not in responses['gemini']:
            return responses['gemini']
        
        # OpenRouter 응답
        if 'openrouter' in responses and responses['openrouter']:
            return responses['openrouter']
        
        # 폴백 응답
        if 'fallback' in responses:
            return responses['fallback']
        
        return "죄송합니다. 현재 서비스에 접속할 수 없습니다. 잠시 후 다시 시도해주세요."
    
    def save_conversation(self, user_input: str, result: Dict):
        """대화 저장"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO conversations 
                (session_id, user_message, bot_response, model_used, response_time, intent_detected)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                st.session_state.user_id,
                user_input,
                result['final_response'],
                ','.join(result['models_used']),
                result['processing_time'],
                ','.join(result['intent_analysis']['intents'])
            ))
            
            # 사용자 메시지 수 업데이트
            cursor.execute('''
                UPDATE users SET total_messages = total_messages + 1 
                WHERE session_id = ?
            ''', (st.session_state.user_id,))
            
            self.conn.commit()
            logger.info(f"대화 저장 완료: {user_input[:50]}...")
            
        except Exception as e:
            logger.error(f"대화 저장 오류: {e}")
    
    def get_conversation_stats(self) -> Dict:
        """대화 통계 조회"""
        try:
            cursor = self.conn.cursor()
            
            # 총 대화 수
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_conversations = cursor.fetchone()[0] or 0
            
            # 총 사용자 수
            cursor.execute("SELECT COUNT(DISTINCT session_id) FROM users")
            total_users = cursor.fetchone()[0] or 0
            
            # 오늘 대화 수
            cursor.execute("SELECT COUNT(*) FROM conversations WHERE DATE(timestamp) = DATE('now')")
            today_conversations = cursor.fetchone()[0] or 0
            
            # 평균 응답 시간
            cursor.execute("SELECT AVG(response_time) FROM conversations WHERE response_time IS NOT NULL")
            avg_response_time = cursor.fetchone()[0] or 0
            
            return {
                'total_conversations': total_conversations,
                'total_users': total_users,
                'today_conversations': today_conversations,
                'avg_response_time': avg_response_time
            }
            
        except Exception as e:
            logger.error(f"통계 조회 오류: {e}")
            return {
                'total_conversations': 0,
                'total_users': 0,
                'today_conversations': 0,
                'avg_response_time': 0
            }
    
    def display_sidebar(self):
        """고급 사이드바 표시"""
        with st.sidebar:
            st.title("🚀 AI 챗봇 컨트롤")
            
            # API 상태 표시
            st.subheader("🔌 API 상태")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Google", "✅" if self.gemini_available else "❌")
            with col2:
                st.metric("OpenRouter", "✅" if self.openrouter_available else "❌")
            
            st.markdown("---")
            
            # 실시간 통계
            st.subheader("📊 실시간 통계")
            stats = self.get_conversation_stats()
            
            st.metric("총 대화 수", f"{stats['total_conversations']:,}")
            st.metric("총 사용자 수", f"{stats['total_users']:,}")
            st.metric("오늘 대화", f"{stats['today_conversations']:,}")
            st.metric("평균 응답시간", f"{stats['avg_response_time']:.2f}s")
            
            st.markdown("---")
            
            # 관리 기능
            st.subheader("⚙️ 관리")
            
            if st.button("🗑️ 현재 대화 지우기", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            
            if st.button("📊 성능 리포트", use_container_width=True):
                self.show_performance_report()
            
            # 시스템 정보
            st.markdown("---")
            st.markdown("""
            **🔧 시스템 정보**
            - 하이브리드 AI 엔진
            - 실시간 모니터링
            - 자동 장애 조치
            """)
    
    def show_performance_report(self):
        """성능 리포트 표시"""
        try:
            cursor = self.conn.cursor()
            
            # 모델별 성능 통계
            cursor.execute('''
                SELECT model_name, 
                       AVG(response_time) as avg_time,
                       COUNT(*) as total_requests,
                       SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count
                FROM performance 
                GROUP BY model_name
            ''')
            
            performance_data = cursor.fetchall()
            
            with st.expander("📈 상세 성능 리포트", expanded=True):
                st.subheader("모델별 성능 비교")
                
                for model, avg_time, total_requests, success_count in performance_data:
                    success_rate = (success_count / total_requests * 100) if total_requests > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{model}", f"{avg_time:.2f}s")
                    with col2:
                        st.metric("요청 수", total_requests)
                    with col3:
                        st.metric("성공률", f"{success_rate:.1f}%")
                    
        except Exception as e:
            st.error(f"성능 리포트 생성 오류: {e}")
    
    def display_chat_interface(self):
        """고급 채팅 인터페이스"""
        st.title(" JiNuhybrid")
        st.markdown("""
        **최강 성능의 하이브리드 AI 시스템:**
        - 🧠 *Google AI**: Google 최신 모델
        - 🔄 **OpenRouter**: 고급추론 모델(Claude Sonnet, Lema)
        - 🎯 **지능형 라우팅**: 상황에 맞는 최적 모델 선택
        - ⚡ **실시간 처리**: 초고속 응답
        (응답 지연은 텍스트 생성하는데 걸리는 시간으로 하이브리드 AI는 응답생성은 바로합니다)
        """)
        
        # 채팅 컨테이너
        chat_container = st.container()
        
        with chat_container:
            # 대화 기록 표시
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # 메타정보 표시 (AI 응답인 경우)
                    if message["role"] == "assistant" and "metadata" in message:
                        with st.expander("🔍 상세 정보"):
                            st.write(f"**사용 모델:** {message['metadata'].get('models_used', 'N/A')}")
                            st.write(f"**처리 시간:** {message['metadata'].get('processing_time', 0):.2f}초")
                            st.write(f"**의도 분석:** {', '.join(message['metadata'].get('intents', []))}")
        
        # 사용자 입력
        if prompt := st.chat_input("무엇이 궁금하신가요?"):
            # 사용자 메시지 표시
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # AI 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("🤔 하이브리드 AI가 분석 중..."):
                    result = self.hybrid_response_generation(prompt)
                
                # 응답 표시
                st.markdown(result['final_response'])
                
                # 상세 정보
                with st.expander("🔧 기술적 세부사항"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("사용 모델", ', '.join(result['models_used']))
                    with col2:
                        st.metric("처리 시간", f"{result['processing_time']:.2f}초")
                    with col3:
                        st.metric("질문 유형", ', '.join(result['intent_analysis']['intents']))
                    
                    # 모든 응답 보기
                    st.write("**모든 AI 응답:**")
                    for model, response in result['responses'].items():
                        st.write(f"**{model.upper()}:** {response}")
            
            # 세션에 메시지 저장 (메타데이터 포함)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result['final_response'],
                "metadata": {
                    "models_used": result['models_used'],
                    "processing_time": result['processing_time'],
                    "intents": result['intent_analysis']['intents']
                }
            })
            
            # 데이터베이스 저장
            self.save_conversation(prompt, result)
            
            # 페이지 새로고침
            st.rerun()

def main():
    """메인 애플리케이션"""
    try:
        # 앱 초기화
        chatbot = UltimateHybridChatbot()
        
        # 사이드바 표시
        chatbot.display_sidebar()
        
        # 메인 채팅 인터페이스 표시
        chatbot.display_chat_interface()
        
        # 푸터
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                "<div style='text-align: center; color: gray;'>"
                "Copyright ⓒ 2025. Synox Studios"
                "</div>", 
                unsafe_allow_html=True
            )
        
        # 세션 시간 표시
        session_duration = datetime.now() - st.session_state.chat_start_time
        st.sidebar.markdown(f"**세션 시간:** {str(session_duration).split('.')[0]}")
        
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 오류: {e}")
        st.error("애플리케이션 실행 중 오류가 발생했습니다. 콘솔 로그를 확인해주세요.")

if __name__ == "__main__":
    main()
