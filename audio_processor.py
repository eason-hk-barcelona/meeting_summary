"""
audio_processor.py - 带智能混音功能的音频处理器
================================================

修改说明：
- 新增 `intelligent_mix` 方法：基于能量选择策略进行多麦克风智能混音
- 新增 `merge_speaker_audios_intelligent` 方法：替代原有的简单平均混音
- 保留原有的所有功能，确保向后兼容

核心策略：
在每个时间窗口内，选择能量最大的麦克风信号，避免串音干扰。
"""

import librosa
import soundfile as sf
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class MixerConfig:
    """智能混音器配置"""
    window_size_ms: float = 100.0      # 分析窗口大小（毫秒）
    hop_size_ms: float = 50.0          # 窗口跳跃大小（毫秒）
    crossfade_ms: float = 20.0         # 交叉淡入淡出时长（毫秒）
    smoothing_window: int = 3          # 平滑窗口大小（减少频繁切换）
    energy_threshold: float = 0.001    # 能量阈值（低于此值认为静音）
    normalize_output: bool = True      # 是否归一化输出
    output_gain: float = 0.95          # 输出增益


class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, mixer_config: Optional[MixerConfig] = None):
        """
        初始化音频处理器
        
        Args:
            sample_rate: 采样率，默认16000
            mixer_config: 智能混音器配置，如果为None则使用默认配置
        """
        self.sample_rate = sample_rate
        self.mixer_config = mixer_config or MixerConfig()
    
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
    
    # =========================================================================
    # 智能混音相关方法
    # =========================================================================
    
    def compute_short_time_energy(
        self, 
        audio: np.ndarray,
        window_size_ms: Optional[float] = None,
        hop_size_ms: Optional[float] = None
    ) -> np.ndarray:
        """
        计算短时能量 (RMS)
        
        Args:
            audio: 音频信号
            window_size_ms: 窗口大小（毫秒），默认使用配置值
            hop_size_ms: 跳跃大小（毫秒），默认使用配置值
        
        Returns:
            能量数组，每个元素对应一个窗口的RMS能量
        """
        window_ms = window_size_ms or self.mixer_config.window_size_ms
        hop_ms = hop_size_ms or self.mixer_config.hop_size_ms
        
        window_samples = int(window_ms * self.sample_rate / 1000)
        hop_samples = int(hop_ms * self.sample_rate / 1000)
        
        # 确保至少有一个窗口
        if len(audio) < window_samples:
            return np.array([np.sqrt(np.mean(audio ** 2))], dtype=np.float32)
        
        num_windows = (len(audio) - window_samples) // hop_samples + 1
        energies = np.zeros(num_windows, dtype=np.float32)
        
        for i in range(num_windows):
            start = i * hop_samples
            end = start + window_samples
            window = audio[start:end]
            energies[i] = np.sqrt(np.mean(window ** 2))  # RMS能量
        
        return energies
    
    def _smooth_selection(self, selection: np.ndarray) -> np.ndarray:
        """
        平滑麦克风选择，使用众数滤波减少频繁切换
        
        Args:
            selection: 原始选择数组
            
        Returns:
            平滑后的选择数组
        """
        smoothing = self.mixer_config.smoothing_window
        if smoothing <= 1:
            return selection
        
        smoothed = selection.copy()
        half_win = smoothing // 2
        
        for i in range(len(selection)):
            start = max(0, i - half_win)
            end = min(len(selection), i + half_win + 1)
            window = selection[start:end]
            # 众数滤波
            values, counts = np.unique(window, return_counts=True)
            smoothed[i] = values[np.argmax(counts)]
        
        return smoothed
    
    def _create_crossfade(self, length: int, fade_in: bool = True) -> np.ndarray:
        """
        创建余弦淡入淡出曲线
        
        Args:
            length: 淡入淡出长度（采样点数）
            fade_in: True为淡入，False为淡出
            
        Returns:
            淡入淡出曲线
        """
        if length <= 0:
            return np.array([], dtype=np.float32)
        
        t = np.linspace(0, np.pi / 2, length)
        if fade_in:
            return (np.sin(t) ** 2).astype(np.float32)
        else:
            return (np.cos(t) ** 2).astype(np.float32)
    
    def intelligent_mix(
        self, 
        mic_audios: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        智能混音：基于能量选择策略
        
        核心原理：
        - 将音频分成小的时间窗口
        - 在每个窗口内计算各麦克风的能量
        - 选择能量最大的麦克风（即当前说话人的麦克风）
        - 使用交叉淡入淡出平滑切换
        
        Args:
            mic_audios: 各麦克风的音频数组列表，要求长度一致
        
        Returns:
            mixed_audio: 混音后的音频
            selection: 每个窗口选择的麦克风索引
            energies: 各麦克风的能量矩阵 (num_mics, num_windows)
        
        示例:
            >>> processor = AudioProcessor()
            >>> audio1, _ = processor.load_audio("mic1.wav")
            >>> audio2, _ = processor.load_audio("mic2.wav")
            >>> audio3, _ = processor.load_audio("mic3.wav")
            >>> mixed, selection, energies = processor.intelligent_mix([audio1, audio2, audio3])
        """
        num_mics = len(mic_audios)
        if num_mics < 1:
            raise ValueError("至少需要一个麦克风的音频")
        
        logger.info(f"开始智能混音，共 {num_mics} 个麦克风")
        
        # 确保所有音频长度一致
        min_len = min(len(a) for a in mic_audios)
        mic_audios = [a[:min_len].astype(np.float32) for a in mic_audios]
        
        logger.info(f"音频长度: {min_len} 采样点 ({min_len/self.sample_rate:.2f} 秒)")
        
        # 如果只有一个麦克风，直接返回
        if num_mics == 1:
            return mic_audios[0], np.zeros(1, dtype=np.int32), np.zeros((1, 1))
        
        # 1. 计算每个麦克风的短时能量
        all_energies = []
        for i, audio in enumerate(mic_audios):
            energy = self.compute_short_time_energy(audio)
            all_energies.append(energy)
            avg_energy = np.mean(energy)
            logger.debug(f"麦克风 {i+1} 平均能量: {avg_energy:.6f}")
        
        # 对齐能量数组长度
        min_energy_len = min(len(e) for e in all_energies)
        energies = np.array([e[:min_energy_len] for e in all_energies])
        
        # 2. 选择能量最大的麦克风
        selection = np.argmax(energies, axis=0)
        
        # 3. 处理静音段：保持前一个选择
        max_energies = np.max(energies, axis=0)
        for i in range(1, len(selection)):
            if max_energies[i] < self.mixer_config.energy_threshold:
                selection[i] = selection[i-1]
        
        # 4. 平滑选择以减少频繁切换
        selection = self._smooth_selection(selection)
        
        # 统计各麦克风使用比例
        for i in range(num_mics):
            usage = np.sum(selection == i) / len(selection) * 100
            logger.info(f"麦克风 {i+1} 使用比例: {usage:.1f}%")
        
        # 5. 构建混音输出
        hop_samples = int(self.mixer_config.hop_size_ms * self.sample_rate / 1000)
        window_samples = int(self.mixer_config.window_size_ms * self.sample_rate / 1000)
        crossfade_samples = int(self.mixer_config.crossfade_ms * self.sample_rate / 1000)
        
        output = np.zeros(min_len, dtype=np.float32)
        weight = np.zeros(min_len, dtype=np.float32)
        
        prev_mic = -1
        for i, mic_idx in enumerate(selection):
            start = i * hop_samples
            end = min(start + window_samples, min_len)
            actual_len = end - start
            
            # 获取当前窗口的音频
            segment = mic_audios[mic_idx][start:end].copy()
            seg_weight = np.ones(actual_len, dtype=np.float32)
            
            # 麦克风切换时应用淡入
            if prev_mic != -1 and prev_mic != mic_idx:
                fade_len = min(crossfade_samples, actual_len)
                fade_in = self._create_crossfade(fade_len, fade_in=True)
                segment[:fade_len] *= fade_in
                seg_weight[:fade_len] *= fade_in
            
            # 下一窗口切换时应用淡出
            if i < len(selection) - 1 and selection[i + 1] != mic_idx:
                fade_len = min(crossfade_samples, actual_len)
                fade_out = self._create_crossfade(fade_len, fade_in=False)
                segment[-fade_len:] *= fade_out
                seg_weight[-fade_len:] *= fade_out
            
            # 重叠相加
            output[start:end] += segment
            weight[start:end] += seg_weight
            
            prev_mic = mic_idx
        
        # 6. 归一化重叠区域
        weight[weight == 0] = 1
        output = output / weight
        
        # 7. 归一化输出
        if self.mixer_config.normalize_output:
            max_val = np.max(np.abs(output))
            if max_val > 0:
                output = output / max_val * self.mixer_config.output_gain
        
        logger.info(f"混音完成，共 {len(selection)} 个窗口")
        
        return output, selection, energies
    
    def intelligent_mix_from_files(
        self, 
        audio_paths: List[str],
        output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, float]:
        """
        从文件执行智能混音
        
        Args:
            audio_paths: 各麦克风wav文件路径列表
            output_path: 输出文件路径（可选）
        
        Returns:
            mixed_audio: 混音后的音频
            duration: 音频时长（秒）
        """
        # 加载所有音频
        mic_audios = []
        for i, path in enumerate(audio_paths):
            logger.info(f"读取麦克风 {i+1}: {path}")
            audio, _ = self.load_audio(path)
            mic_audios.append(audio)
        
        # 执行智能混音
        mixed_audio, selection, energies = self.intelligent_mix(mic_audios)
        
        # 保存输出（如果指定了路径）
        if output_path:
            sf.write(output_path, mixed_audio, self.sample_rate)
            logger.info(f"混音文件已保存: {output_path}")
        
        duration = len(mixed_audio) / self.sample_rate
        return mixed_audio, duration
    
    def get_selection_timeline(
        self, 
        selection: np.ndarray
    ) -> List[Tuple[float, float, int]]:
        """
        获取麦克风选择的时间线
        
        Args:
            selection: 选择数组（来自 intelligent_mix 的返回值）
        
        Returns:
            timeline: [(start_time, end_time, mic_index), ...]
        """
        hop_samples = int(self.mixer_config.hop_size_ms * self.sample_rate / 1000)
        
        timeline = []
        current_mic = selection[0]
        start_idx = 0
        
        for i in range(1, len(selection)):
            if selection[i] != current_mic:
                start_time = start_idx * hop_samples / self.sample_rate
                end_time = i * hop_samples / self.sample_rate
                timeline.append((start_time, end_time, int(current_mic)))
                
                current_mic = selection[i]
                start_idx = i
        
        # 添加最后一段
        start_time = start_idx * hop_samples / self.sample_rate
        end_time = len(selection) * hop_samples / self.sample_rate
        timeline.append((start_time, end_time, int(current_mic)))
        
        return timeline
    
    # =========================================================================
    # 原有方法（保持向后兼容）
    # =========================================================================
    
    def merge_speaker_audios(self, audio_paths: List[str], 
                            textgrids: List[dict]) -> Tuple[np.ndarray, List[dict]]:
        """
        合并三个说话人的音频，基于时间轴对齐
        
        【已弃用】建议使用 merge_speaker_audios_intelligent 方法
        
        Args:
            audio_paths: 三个音频文件路径
            textgrids: 解析后的TextGrid数据
        Returns:
            合并的音频和对齐的标注
        """
        # 加载所有音频
        audios = [self.load_audio(path)[0] for path in audio_paths]
        
        # 创建混合音频（简单平均混音）
        max_len = max(len(audio) for audio in audios)
        merged = np.zeros(max_len)
        
        for audio in audios:
            merged[:len(audio)] += audio / len(audios)
            
        # 合并标注信息
        merged_annotations = self._merge_annotations(textgrids)
        
        return merged, merged_annotations
    
    def merge_speaker_audios_intelligent(
        self, 
        audio_paths: List[str], 
        textgrids: List[dict],
        output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, List[dict], np.ndarray]:
        """
        【新方法】使用智能混音合并多个说话人的音频
        
        相比原有的平均混音，这个方法：
        - 在每个时间窗口选择能量最大的麦克风
        - 避免串音干扰，提高音频清晰度
        - 使用交叉淡入淡出避免切换爆音
        
        Args:
            audio_paths: 音频文件路径列表（如 ["192.wav", "193.wav", "194.wav"]）
            textgrids: 解析后的TextGrid数据
            output_path: 可选，保存混音文件的路径
        
        Returns:
            merged_audio: 智能混音后的音频
            merged_annotations: 合并的标注信息
            selection: 每个窗口选择的麦克风索引（用于调试/分析）
        
        示例:
            >>> processor = AudioProcessor()
            >>> audio_paths = ["mic1.wav", "mic2.wav", "mic3.wav"]
            >>> textgrids = [tg1, tg2, tg3]
            >>> merged, annotations, selection = processor.merge_speaker_audios_intelligent(
            ...     audio_paths, textgrids, output_path="mixed.wav"
            ... )
        """
        print(f"智能混音: 处理 {len(audio_paths)} 个麦克风...")
        
        # 执行智能混音
        merged_audio, duration = self.intelligent_mix_from_files(audio_paths, output_path)
        
        # 获取选择结果（用于调试）
        mic_audios = [self.load_audio(path)[0] for path in audio_paths]
        _, selection, _ = self.intelligent_mix(mic_audios)
        
        # 打印选择时间线
        timeline = self.get_selection_timeline(selection)
        print(f"\n麦克风切换时间线 (共 {len(timeline)} 段):")
        for start, end, mic in timeline[:10]:  # 只显示前10段
            print(f"  {start:.1f}s - {end:.1f}s: 麦克风 {mic+1}")
        if len(timeline) > 10:
            print(f"  ... 还有 {len(timeline)-10} 段")
        
        # 合并标注信息
        merged_annotations = self._merge_annotations(textgrids)
        
        return merged_audio, merged_annotations, selection
    
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


# =============================================================================
# 便捷函数
# =============================================================================

def create_processor(
    sample_rate: int = 16000,
    window_ms: float = 100.0,
    hop_ms: float = 50.0,
    crossfade_ms: float = 20.0,
    smoothing: int = 3
) -> AudioProcessor:
    """
    创建带自定义混音配置的音频处理器
    
    Args:
        sample_rate: 采样率
        window_ms: 分析窗口大小（毫秒）
        hop_ms: 窗口跳跃大小（毫秒）
        crossfade_ms: 交叉淡入淡出时长（毫秒）
        smoothing: 平滑窗口大小
    
    Returns:
        配置好的AudioProcessor实例
    """
    config = MixerConfig(
        window_size_ms=window_ms,
        hop_size_ms=hop_ms,
        crossfade_ms=crossfade_ms,
        smoothing_window=smoothing
    )
    return AudioProcessor(sample_rate=sample_rate, mixer_config=config)