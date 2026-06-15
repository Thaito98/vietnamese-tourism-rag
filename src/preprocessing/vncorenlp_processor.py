"""
VnCoreNLP PROCESSOR
Wrapper for VnCoreNLP with stopwords removal
"""
import subprocess
import tempfile
import os
import re
from pathlib import Path
from typing import List
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.vietnamese_stopwords import remove_stopwords

class VnCoreNLPProcessor:
    """
    VnCoreNLP Processor với stopwords removal
    """
    
    def __init__(self, vncorenlp_dir: str = None):
        """
        Args:
            vncorenlp_dir: Đường dẫn tới folder chứa VnCoreNLP-1.2.jar và models/
                          Mặc định: thư mục gốc project
        """
        if vncorenlp_dir is None:
            # Mặc định: thư mục gốc project
            vncorenlp_dir = Path(__file__).parent.parent.parent
        
        self.vncorenlp_dir = Path(vncorenlp_dir)
        self.jar_path = self.vncorenlp_dir / "VnCoreNLP-1.2.jar"
        self.models_dir = self.vncorenlp_dir / "models"
        
        # Kiểm tra files cần thiết
        if not self.jar_path.exists():
            raise FileNotFoundError(f"VnCoreNLP JAR not found: {self.jar_path}")
        if not self.models_dir.exists():
            raise FileNotFoundError(f"VnCoreNLP models not found: {self.models_dir}")
    
    def tokenize(self, text: str, remove_stopwords_flag: bool = True) -> List[str]:
        """
        Tách từ tiếng Việt sử dụng VnCoreNLP
        
        Args:
            text: Văn bản cần tách từ
            remove_stopwords_flag: Có loại bỏ stopwords không
            
        Returns:
            List các tokens
        """
        if not text or not text.strip():
            return []
        
        try:
            # Tạo file tạm cho input
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', 
                                           suffix='.txt', delete=False) as f_in:
                f_in.write(text)
                input_file = f_in.name
            
            # Tạo file tạm cho output
            output_file = input_file + '.out'
            
            # Chạy VnCoreNLP
            cmd = [
                'java',
                '-jar', str(self.jar_path),
                '-fin', input_file,
                '-fout', output_file,
                '-annotators', 'wseg'
            ]
            
            # Chạy command (timeout 10s)
            result = subprocess.run(
                cmd,
                cwd=str(self.vncorenlp_dir),
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8',
                errors='ignore'
            )
            
            # Đọc kết quả
            tokens = []
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        # Format: số TAB từ TAB ...
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            token = parts[1].strip()
                            if token and token != '_':
                                tokens.append(token)

            # Cleanup
            if os.path.exists(input_file):
                os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)

            # Loại bỏ dấu câu và ký tự đặc biệt
            # Giữ lại token nào có ít nhất 1 ký tự chữ hoặc số
            tokens = [t for t in tokens if re.search(r'\w', t)]

            # Loại bỏ stopwords nếu cần
            if remove_stopwords_flag and tokens:
                tokens = remove_stopwords(tokens)
            
            return tokens
            
        except subprocess.TimeoutExpired:
            print(f"Warning: VnCoreNLP timeout for text: {text[:50]}...")
            tokens = text.lower().split()
            tokens = [t for t in tokens if re.search(r'\w', t)]
            return remove_stopwords(tokens) if remove_stopwords_flag else tokens

        except Exception as e:
            print(f"Warning: VnCoreNLP error: {e}")
            tokens = text.lower().split()
            tokens = [t for t in tokens if re.search(r'\w', t)]
            return remove_stopwords(tokens) if remove_stopwords_flag else tokens
    
    def preprocess(self, text: str) -> str:
        """
        Tiền xử lý text: lowercase + tokenize + remove stopwords
        
        Returns:
            Chuỗi tokens nối bằng dấu cách
        """
        tokens = self.tokenize(text.lower(), remove_stopwords_flag=True)
        return ' '.join(tokens)
