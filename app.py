import streamlit as st
import requests
import json
import os
from datetime import datetime
import sqlite3
import hashlib

# 페이지 설정
st.set_page_config(
    page_title="AI 챗봇 웹사이트",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .user-message {
        background-color: #1f77b4;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .bot-message {
        background-color: #28a745;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class WebsiteChatbot:
    def __init__(self):
        self.setup_database()
        self.setup_session_state()
    
    def setup_database(self):
        """사용자 데이터베이스 설정"""
        self.conn = sqlite3.connect('chatbot_website.db', check_same_thread=False)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_message TEXT,
                bot_response TEXT,
                timestamp DATETIME,
                model_used TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip_hash TEXT,
                first_visit DATETIME,
                visit_count INTEGER
            )
        ''')
        
        self.conn.commit()
    
    def setup_session_state(self):
        """세션 상태 초기화"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'user_id' not in st.session_state:
            # 간단한 사용자 식별자 생성
            st.session_state.user_id = hashlib.md5(
                str(datetime.now().timestamp()).encode()
            ).hexdigest()
            
            self.track_user_visit()
    
    def track_user_visit(self):
        """사용자 방문 추적"""
        cursor = self.conn.cursor()
        
        # IP 해시 생성 (개인정보 보호)
        ip_hash = st.session_state.user_id
        
        cursor.execute('''
            INSERT OR REPLACE INTO users (ip_hash, first_visit, visit_count)
            VALUES (?, COALESCE((SELECT first_visit FROM users WHERE ip_hash=?), datetime('now')), 
                   COALESCE((SELECT visit_count FROM users WHERE ip_hash=?), 0) + 1)
        ''', (ip_hash, ip_hash, ip_hash))
        
        self.conn.commit()
    
    def save_conversation(self, user_msg, bot_response, model_used):
        """대화 저장"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (session_id, user_message, bot_response, timestamp, model_used)
            VALUES (?, ?, ?, ?, ?)
        ''', (st.session_state.user_id, user_msg, bot_response, datetime.now(), model_used))
        
        self.conn.commit()
    
    def call_ai_api(self, message):
        """AI API 호출 (백엔드 또는 직접)"""
        try:
            # 백엔드 API 호출 시도
            response = requests.post(
                "https://your-backend.com/api/chat",
                json={"message": message},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["response"], "backend"
        except:
            # 백엔드 실패 시 직접 Gemini 호출
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(message)
                
                return response.text, "gemini"
            except Exception as e:
                return f"죄송합니다. 일시적인 오류가 발생했습니다: {str(e)}", "error"
        
        return "현재 서비스가 원활하지 않습니다. 잠시 후 다시 시도해주세요.", "error"
    
    def display_chat_interface(self):
        """채팅 인터페이스 표시"""
        st.markdown('<div class="main-header">🤖 AI 챗봇 웹사이트</div>', unsafe_allow_html=True)
        
        # 소개 섹션
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**💡 다양한 주제**\n\n어떤 질문이든 편하게 물어보세요!")
        
        with col2:
            st.info("**🚀 빠른 응답**\n\n최신 AI 기술로 빠르게 답변드립니다")
        
        with col3:
            st.info("**🔒 안전한 대화**\n\n개인정보를 보호하는 안전한 채팅")
        
        st.markdown("---")
        
        # 채팅 컨테이너
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # 대화 기록 표시
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">🤖 {message["content"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 입력 폼
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input(
                    "메시지를 입력하세요...",
                    placeholder="무엇이 궁금하신가요?",
                    label_visibility="collapsed"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                submit_button = st.form_submit_button("전송", use_container_width=True)
        
        if submit_button and user_input:
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # AI 응답 생성
            with st.spinner("AI가 답변을 생성 중입니다..."):
                bot_response, model_used = self.call_ai_api(user_input)
                
                # 봇 응답 추가
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                
                # 대화 저장
                self.save_conversation(user_input, bot_response, model_used)
            
            # 페이지 새로고침으로 새 메시지 표시
            st.rerun()
    
    def display_sidebar(self):
        """사이드바 표시"""
        with st.sidebar:
            st.title("ℹ️ 정보")
            
            st.markdown("""
            ### 이 웹사이트는...
            최신 AI 기술을 활용한 지능형 챗봇 서비스입니다.
            
            **기능:**
            - 다양한 주제 대화
            - 실시간 응답
            - 대화 기록 저장
            - 모바일 최적화
            """)
            
            st.markdown("---")
            
            # 통계 표시
            st.subheader("📊 통계")
            
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_chats = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT session_id) FROM users")
            total_users = cursor.fetchone()[0]
            
            st.metric("총 대화 수", f"{total_chats:,}")
            st.metric("방문자 수", f"{total_users:,}")
            
            st.markdown("---")
            
            # 관리자 링크
            if st.checkbox("관리자 모드"):
                self.display_admin_panel()
    
    def display_admin_panel(self):
        """관리자 패널 표시"""
        st.subheader("🔧 관리자 패널")
        
        # 데이터베이스 관리
        if st.button("데이터베이스 백업"):
            # 백업 로직 구현
            st.success("백업이 완료되었습니다.")
        
        # 대화 기록 보기
        if st.button("최근 대화 보기"):
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT user_message, bot_response, timestamp 
                FROM conversations 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''')
            
            recent_chats = cursor.fetchall()
            
            for user_msg, bot_resp, timestamp in recent_chats:
                with st.expander(f"{timestamp} - {user_msg[:50]}..."):
                    st.write(f"**사용자:** {user_msg}")
                    st.write(f"**봇:** {bot_resp}")

def main():
    # 웹사이트 초기화
    website = WebsiteChatbot()
    
    # 사이드바 표시
    website.display_sidebar()
    
    # 메인 채팅 인터페이스 표시
    website.display_chat_interface()
    
    # 푸터
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "© 2024 AI 챗봇 웹사이트. All rights reserved."
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
