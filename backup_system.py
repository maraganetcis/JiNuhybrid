# backup_system.py
import sqlite3
import shutil
import os
from datetime import datetime, timedelta
import json

class BackupManager:
    def __init__(self, db_path='chatbot.db', backup_dir='backups'):
        self.db_path = db_path
        self.backup_dir = backup_dir
        self.ensure_backup_dir()
    
    def ensure_backup_dir(self):
        """백업 디렉토리 생성"""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
    
    def create_backup(self):
        """데이터베이스 백업 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}.db")
        
        # 데이터베이스 복사
        shutil.copy2(self.db_path, backup_path)
        
        # 백업 메타데이터 저장
        metadata = {
            'timestamp': timestamp,
            'size': os.path.getsize(backup_path),
            'file': f"backup_{timestamp}.db"
        }
        
        with open(os.path.join(self.backup_dir, 'backup_metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        print(f"✅ 백업 생성 완료: {backup_path}")
        return backup_path
    
    def auto_cleanup(self, keep_days=7):
        """오래된 백업 자동 삭제"""
        now = datetime.now()
        
        for filename in os.listdir(self.backup_dir):
            if filename.startswith('backup_') and filename.endswith('.db'):
                filepath = os.path.join(self.backup_dir, filename)
                file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                
                if (now - file_time) > timedelta(days=keep_days):
                    os.remove(filepath)
                    print(f"🗑️ 오래된 백업 삭제: {filename}")
    
    def export_conversations(self):
        """대화 기록 JSON으로 내보내기"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id, user_message, bot_response, timestamp 
            FROM conversations 
            ORDER BY timestamp DESC
        ''')
        
        conversations = cursor.fetchall()
        
        export_data = []
        for conv in conversations:
            export_data.append({
                'session_id': conv[0],
                'user_message': conv[1],
                'bot_response': conv[2],
                'timestamp': conv[3]
            })
        
        export_path = os.path.join(
            self.backup_dir, 
            f"conversations_export_{datetime.now().strftime('%Y%m%d')}.json"
        )
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        conn.close()
        print(f"📤 대화 기록 내보내기 완료: {export_path}")

# 백업 자동화
def setup_auto_backup():
    """자동 백업 설정"""
    backup_manager = BackupManager()
    
    # 매일 자정 백업
    backup_manager.create_backup()
    
    # 오래된 백업 정리
    backup_manager.auto_cleanup()
    
    # 주간 리포트 생성
    if datetime.now().weekday() == 0:  # 월요일
        backup_manager.export_conversations()
