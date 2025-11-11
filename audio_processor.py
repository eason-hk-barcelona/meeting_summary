import librosa
import soundfile as sf
import numpy as np
from typing import List, Tuple

class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """加载音频文件"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = len(audio) / sr
        return audio, duration
    
    def segment_audio(self, audio: np.ndarray, segment_duration: int = 300) -> List[np.ndarray]:
        """
        将音频分段
        Args:
            audio: 音频数组
            segment_duration: 每段时长（秒）
        Returns:
            音频段列表
        """
        segment_samples = segment_duration * self.sample_rate
        segments = []
        
        for start in range(0, len(audio), segment_samples):
            end = min(start + segment_samples, len(audio))
            segments.append(audio[start:end])
            
        return segments
    
    def merge_speaker_audios(self, audio_paths: List[str], 
                            textgrids: List[dict]) -> Tuple[np.ndarray, List[dict]]:
        """
        合并三个说话人的音频，基于时间轴对齐
        Args:
            audio_paths: 三个音频文件路径
            textgrids: 解析后的TextGrid数据
        Returns:
            合并的音频和对齐的标注
        """
        # 加载所有音频
        audios = [self.load_audio(path)[0] for path in audio_paths]
        
        # 创建混合音频（这里简单相加，实际可能需要更复杂的混音）
        max_len = max(len(audio) for audio in audios)
        merged = np.zeros(max_len)
        
        for audio in audios:
            merged[:len(audio)] += audio / len(audios)  # 平均混音
            
        # 合并标注信息（按时间排序）
        merged_annotations = self._merge_annotations(textgrids)
        
        return merged, merged_annotations
    
    def _merge_annotations(self, textgrids: List[dict]) -> List[dict]:
        """合并多个TextGrid的标注，按时间排序"""
        all_annotations = []
        
        for speaker_id, tg in enumerate(textgrids, start=192):
            for interval in tg['intervals']:
                if interval['text'] != '<NOISE>':
                    all_annotations.append({
                        'start': interval['xmin'],
                        'end': interval['xmax'],
                        'text': interval['text'],
                        'speaker': f'Speaker {speaker_id}'
                    })
        
        # 按开始时间排序
        all_annotations.sort(key=lambda x: x['start'])
        return all_annotations