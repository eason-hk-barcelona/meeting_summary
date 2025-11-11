import re
from typing import Dict, List

class TextGridParser:
    def __init__(self):
        self.content_tier = "内容层"
        self.role_tier = "角色层"
    
    def parse(self, textgrid_path: str) -> Dict:
        """
        解析TextGrid文件
        Returns:
            包含内容和角色信息的字典
        """
        with open(textgrid_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取基本信息
        xmin = float(re.search(r'xmin = ([\d.]+)', content).group(1))
        xmax = float(re.search(r'xmax = ([\d.]+)', content).group(1))
        
        # 解析intervals
        intervals = self._parse_intervals(content)
        
        # 清理噪声标记
        cleaned_intervals = [
            interval for interval in intervals 
            if interval['text'] != '<NOISE>'
        ]
        
        return {
            'duration': xmax - xmin,
            'intervals': cleaned_intervals,
            'raw_intervals': intervals
        }
    
    def _parse_intervals(self, content: str) -> List[Dict]:
        """解析所有intervals"""
        intervals = []
        
        # 使用正则表达式提取所有interval
        pattern = r'intervals \[\d+\]:\s*\n\s*xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "(.*?)"'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            intervals.append({
                'xmin': float(match[0]),
                'xmax': float(match[1]),
                'text': match[2].strip()
            })
        
        return intervals
    
    def extract_transcript(self, textgrid_path: str) -> str:
        """提取纯文本转录"""
        data = self.parse(textgrid_path)
        texts = [interval['text'] for interval in data['intervals']]
        return ' '.join(texts)