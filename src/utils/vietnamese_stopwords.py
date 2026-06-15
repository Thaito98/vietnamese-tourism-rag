"""
VIETNAMESE STOPWORDS
Danh sách stopwords tiếng Việt 
"""

# Stopwords được chia theo nhóm để dễ quản lý
VIETNAMESE_STOPWORDS = {
    # Liên từ
    'thì', 'là', 'mà', 'và', 'hay', 'hoặc', 'nhưng', 'vì', 'nên',
    'cho_nên', 'tuy_nhiên', 'song', 'nhưng_mà',
    
    # Đại từ
    'tôi', 'bạn', 'anh', 'chị', 'em', 'mình', 'ta', 'chúng_ta',
    'này', 'đó', 'kia', 'nọ', 'ấy', 'đấy',
    
    # Giới từ
    'của', 'với', 'từ', 'đến', 'về', 'theo', 'bởi', 'do', 'vào', 
    'ở', 'tại', 'trong', 'ngoài', 'trên', 'dưới', 'giữa',
    
    # Động từ phổ biến (không mang nghĩa quan trọng)
    'có', 'được', 'đã', 'sẽ', 'đang', 'bị', 'cho', 'làm', 'ra',
    
    # Trợ từ nghi vấn
    'gì', 'nào', 'sao', 'thế_nào', 'như_thế_nào', 'ở_đâu', 
    'bao_nhiêu', 'mấy', 'khi_nào',
    
    # Từ chỉ số lượng không quan trọng
    'nhiều', 'ít', 'một', 'hai', 'ba', 'vài', 'một_số', 'các',
    'những', 'mọi', 'tất_cả',
    
    # Từ không mang thông tin
    'rất', 'khá', 'hơi', 'cũng', 'còn', 'nữa', 'luôn', 'đều',
    'chỉ', 'chỉ_có', 'duy_nhất',
}

def is_stopword(word: str) -> bool:
    """Kiểm tra xem từ có phải stopword không"""
    return word.lower() in VIETNAMESE_STOPWORDS

def remove_stopwords(tokens: list) -> list:
    """Loại bỏ stopwords từ list tokens"""
    return [t for t in tokens if not is_stopword(t)]
