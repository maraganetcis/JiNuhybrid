# error_handling.py
import logging
import traceback
from typing import Optional

class ChatbotErrorHandler:
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            filename='chatbot_errors.log',
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def handle_api_error(self, error: Exception, user_input: str) -> str:
        """API 에러 처리"""
        error_msg = f"API Error: {str(error)}"
        logging.error(f"{error_msg} - User input: {user_input}")
        
        # 사용자 친화적인 메시지
        user_friendly_messages = {
            "API key": "🔑 API 키를 확인해주세요. 설정에서 다시 입력해주세요.",
            "quota": "📊 오늘 사용량을 초과했습니다. 내일 다시 시도해주세요.",
            "network": "🌐 네트워크 연결을 확인해주세요.",
            "timeout": "⏰ 응답 시간이 초과되었습니다. 다시 시도해주세요."
        }
        
        for key, message in user_friendly_messages.items():
            if key in str(error).lower():
                return message
        
        return "😅 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
    
    def safe_api_call(self, api_func, *args, **kwargs):
        """안전한 API 호출 래퍼"""
        try:
            return api_func(*args, **kwargs)
        except Exception as e:
            return self.handle_api_error(e, kwargs.get('user_input', ''))

# 사용 예시
error_handler = ChatbotErrorHandler()
response = error_handler.safe_api_call(call_gemini_api, user_input=user_message)
