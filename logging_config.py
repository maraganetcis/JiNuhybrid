# logging_config.py
import logging
import sys
from datetime import datetime

def setup_comprehensive_logging():
    """종합 로깅 시스템 설정"""
    
    # 로거 생성
    logger = logging.getLogger('chatbot')
    logger.setLevel(logging.INFO)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 파일 핸들러
    file_handler = logging.FileHandler(
        f'chatbot_logs_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler.setFormatter(formatter)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 사용 예시
logger = setup_comprehensive_logging()

def log_conversation(session_id, user_input, bot_response, response_time):
    """대화 로깅"""
    logger.info(
        f"Session: {session_id} | "
        f"User: {user_input[:50]}... | "
        f"Response: {bot_response[:50]}... | "
        f"Time: {response_time:.2f}s"
    )
