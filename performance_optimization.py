# performance_optimization.py
import functools
import time
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_gemini_response(user_input: str):
    """자주 묻는 질문 캐싱"""
    # 캐시에서 먼저 찾기
    pass

def optimize_streamlit():
    """Streamlit 성능 최적화"""
    # 1. 캐싱 활용
    @st.cache_data(ttl=3600)  # 1시간 캐시
    def load_data():
        return "캐시된 데이터"
    
    # 2. session_state로 재렌더링 방지
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        # 초기화 코드
    
    # 3. 불필요한 계산 방지
    if st.button("클릭할 때만 실행"):
        # 무거운 연산
        pass
