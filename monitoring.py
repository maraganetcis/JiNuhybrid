# monitoring.py
def real_time_monitoring():
    """실시간 모니터링 대시보드"""
    st.sidebar.markdown("## 📊 실시간 모니터링")
    
    # 시스템 상태
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        st.metric("API 응답 시간", "0.8s")
    
    with col2:
        st.metric("오늘 대화 수", "1,234")
    
    with col3:
        st.metric("에러율", "0.2%")
    
    # 최근 로그
    st.sidebar.markdown("### 🔍 최근 활동")
    if os.path.exists('chatbot_errors.log'):
        with open('chatbot_errors.log', 'r') as f:
            recent_logs = f.readlines()[-5:]
            for log in recent_logs:
                st.sidebar.text(log.strip())
