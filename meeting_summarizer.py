"""
meeting_summarizer.py - 会议总结器（使用智能混音）
=================================================

修改说明：
- 新增方案C：使用智能混音后的单一音频进行处理
- 智能混音策略：在每个时间窗口选择能量最大的麦克风，避免串音
- 混音后的音频更清晰，适合喂给Qwen2.5-Omni
"""

import os
from typing import List, Dict, Optional
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import soundfile as sf
import numpy as np

from audio_processor import AudioProcessor, MixerConfig
from textgrid_parser import TextGridParser
from config import *


class MeetingSummarizer:
    def __init__(self, use_intelligent_mixing: bool = True):
        """
        初始化模型和处理器
        
        Args:
            use_intelligent_mixing: 是否使用智能混音（默认True）
        """
        print("Loading Qwen2.5-Omni model...")
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype="auto", 
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)

        # 配置智能混音器
        mixer_config = MixerConfig(
            window_size_ms=100.0,    # 100ms窗口
            hop_size_ms=50.0,        # 50ms跳跃
            crossfade_ms=20.0,       # 20ms淡入淡出
            smoothing_window=3,      # 平滑窗口
            energy_threshold=0.001   # 能量阈值
        )
        self.audio_processor = AudioProcessor(mixer_config=mixer_config)
        self.textgrid_parser = TextGridParser()
        self.use_intelligent_mixing = use_intelligent_mixing

    def process_meeting(self, meeting_folder: str, 
                       processing_mode: str = "intelligent_mixed") -> Dict:
        """
        处理整场会议
        
        Args:
            meeting_folder: 包含音频和TextGrid文件的文件夹
            processing_mode: 处理模式
                - "intelligent_mixed": 【推荐】智能混音后处理单一音频
                - "separately": 分别处理每个说话人
                - "simple_merged": 简单平均混音（原方案）
        
        Returns:
            会议总结字典
        """
        # 1. 收集文件
        speakers = ['192', '193', '194']
        audio_files = []
        textgrid_files = []

        for speaker in speakers:
            audio_files.append(os.path.join(
                meeting_folder, f'M005-F8N-{speaker}.wav'))
            textgrid_files.append(os.path.join(
                meeting_folder, f'M005-F8N-{speaker}.TextGrid'))

        # 2. 解析TextGrid获取结构化信息
        print("Parsing TextGrid files...")
        textgrids = [self.textgrid_parser.parse(tg) for tg in textgrid_files]

        # 3. 根据模式处理音频
        print(f"Processing audio files (mode: {processing_mode})...")
        
        if processing_mode == "intelligent_mixed":
            # 方案C: 【推荐】智能混音后处理
            print("智能混音模式_tag")
            return self._process_intelligent_mixed(audio_files, textgrids, speakers)
        elif processing_mode == "separately":
            # 方案A: 分别处理每个说话人
            print("分别处理每个说话人_tag")
            return self._process_separately(audio_files, textgrids, speakers)
        else:
            # 方案B: 简单合并
            return self._process_merged(audio_files, textgrids)

    def _process_intelligent_mixed(self, audio_files: List[str],
                                   textgrids: List[Dict],
                                   speakers: List[str]) -> Dict:
        """
        【新方案】使用智能混音处理会议录音
        
        流程：
        1. 对多个麦克风录音进行智能混音
        2. 将混音后的单一音频分段
        3. 逐段喂给Qwen2.5-Omni处理
        4. 合并生成最终总结
        """
        print("\n" + "="*60)
        print("智能混音模式")
        print("="*60)
        
        # 1. 执行智能混音
        print("\n[步骤1] 执行智能混音...")
        mixed_audio_path = os.path.join(
            os.path.dirname(audio_files[0]), 
            "intelligent_mixed.wav"
        )
        
        mixed_audio, duration = self.audio_processor.intelligent_mix_from_files(
            audio_files, 
            output_path=mixed_audio_path
        )
        
        print(f"  ✓ 混音完成: {mixed_audio_path}")
        print(f"  ✓ 总时长: {duration:.2f} 秒")
        
        # 获取混音选择信息（用于分析哪个麦克风在什么时候被选中）
        mic_audios = [self.audio_processor.load_audio(path)[0] for path in audio_files]
        _, selection, energies = self.audio_processor.intelligent_mix(mic_audios)
        
        # 打印混音统计
        print("\n  麦克风使用统计:")
        for i, speaker in enumerate(speakers):
            usage = np.sum(selection == i) / len(selection) * 100
            print(f"    Speaker {speaker}: {usage:.1f}%")
        
        # 2. 合并所有TextGrid标注
        merged_annotations = self.audio_processor._merge_annotations(textgrids)
        full_transcript = ' '.join([ann['text'] for ann in merged_annotations])
        
        # 3. 分段处理混音后的音频
        print(f"\n[步骤2] 分段处理音频...")
        all_segments = []
        
        if duration > SEGMENT_DURATION:
            # 音频较长，需要分段
            segments = self.audio_processor.segment_audio(mixed_audio, SEGMENT_DURATION)
            
            for j, segment in enumerate(segments):
                print(f"  处理段 {j+1}/{len(segments)}...")
                
                # 保存临时分段文件
                temp_audio_path = f"/tmp/mixed_segment_{j}.wav"
                sf.write(temp_audio_path, segment, self.audio_processor.sample_rate)
                
                # 计算时间范围
                start_time = j * SEGMENT_DURATION
                end_time = min((j + 1) * SEGMENT_DURATION, duration)
                
                # 获取该时间段的文本
                segment_text = self._get_text_for_timerange_from_annotations(
                    merged_annotations, start_time, end_time
                )
                
                # 获取该时间段的说话人统计
                speaker_stats = self._get_speaker_stats_for_timerange(
                    selection, speakers, start_time, end_time
                )
                
                # 生成段总结
                segment_summary = self._summarize_mixed_segment(
                    temp_audio_path,
                    segment_text,
                    speakers,
                    speaker_stats,
                    start_time,
                    end_time
                )
                
                all_segments.append({
                    'segment': j,
                    'start': start_time,
                    'end': end_time,
                    'speaker_stats': speaker_stats,
                    'summary': segment_summary
                })
        else:
            # 音频较短，直接处理
            speaker_stats = {f"Speaker {s}": np.sum(selection == i) / len(selection) * 100 
                           for i, s in enumerate(speakers)}
            
            segment_summary = self._summarize_mixed_segment(
                mixed_audio_path,
                full_transcript[:2000],
                speakers,
                speaker_stats,
                0,
                duration
            )
            
            all_segments.append({
                'segment': 0,
                'start': 0,
                'end': duration,
                'speaker_stats': speaker_stats,
                'summary': segment_summary
            })
        
        # 4. 生成最终总结
        print(f"\n[步骤3] 生成最终总结...")
        meeting_summary = self._generate_final_summary(all_segments)
        
        return {
            'mode': 'intelligent_mixed',
            'mixed_audio_path': mixed_audio_path,
            'duration': duration,
            'segments': all_segments,
            'final_summary': meeting_summary
        }
    
    def _summarize_mixed_segment(self, audio_path: str,
                                  text_context: str,
                                  speakers: List[str],
                                  speaker_stats: Dict[str, float],
                                  start_time: float,
                                  end_time: float) -> str:
        """
        为智能混音后的音频段生成总结
        """
        # 构建说话人统计信息
        stats_text = ", ".join([f"{k}: {v:.1f}%" for k, v in speaker_stats.items()])
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": f"""
This is a meeting recording segment from {start_time:.1f}s to {end_time:.1f}s.

Participants: {', '.join([f'Speaker {s}' for s in speakers])}
Speaking time distribution: {stats_text}

Reference transcript (may contain multiple speakers):
{text_context[:1000]}...

Please analyze this audio segment and provide:
1. Main topics discussed in this segment
2. Key points and opinions expressed
3. Speaker identification (when you can distinguish different voices)
4. Emotional tone and dynamics
5. Any questions, concerns, or decisions mentioned

Provide a structured summary.
"""}
                ]
            }
        ]

        # 处理输入
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=True)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        # 生成总结
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                use_audio_in_video=True,
                return_audio=False,
                max_new_tokens=512,
                do_sample=False
            )

        response = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response
    
    def _get_text_for_timerange_from_annotations(
        self, 
        annotations: List[Dict],
        start_time: float,
        end_time: float
    ) -> str:
        """从合并的标注中获取指定时间范围的文本"""
        texts = []
        for ann in annotations:
            if ann['start'] >= start_time and ann['end'] <= end_time:
                texts.append(f"[{ann['speaker']}] {ann['text']}")
        return ' '.join(texts)
    
    def _get_speaker_stats_for_timerange(
        self,
        selection: np.ndarray,
        speakers: List[str],
        start_time: float,
        end_time: float
    ) -> Dict[str, float]:
        """获取指定时间范围内各说话人的占比"""
        hop_ms = self.audio_processor.mixer_config.hop_size_ms
        sr = self.audio_processor.sample_rate
        
        # 计算窗口索引范围
        start_idx = int(start_time * 1000 / hop_ms)
        end_idx = int(end_time * 1000 / hop_ms)
        
        # 确保索引在有效范围内
        start_idx = max(0, min(start_idx, len(selection) - 1))
        end_idx = max(0, min(end_idx, len(selection)))
        
        # 统计该范围内的选择
        segment_selection = selection[start_idx:end_idx]
        
        stats = {}
        for i, speaker in enumerate(speakers):
            if len(segment_selection) > 0:
                usage = np.sum(segment_selection == i) / len(segment_selection) * 100
            else:
                usage = 0
            stats[f"Speaker {speaker}"] = usage
        
        return stats

    # =========================================================================
    # 原有方法（保持向后兼容）
    # =========================================================================
    
    def _process_separately(self, audio_files: List[str],
                            textgrids: List[Dict],
                            speakers: List[str]) -> Dict:
        """
        分别处理每个说话人的音频（原方案A）
        """
        all_segments = []

        for i, (audio_file, speaker) in enumerate(zip(audio_files, speakers)):
            print(f"Processing Speaker {speaker}...")

            audio, duration = self.audio_processor.load_audio(audio_file)

            if duration > SEGMENT_DURATION:
                segments = self.audio_processor.segment_audio(
                    audio, SEGMENT_DURATION)

                for j, segment in enumerate(segments):
                    print(f"  Processing segment {j+1}/{len(segments)}...")

                    temp_audio_path = f"/tmp/segment_{speaker}_{j}.wav"
                    sf.write(temp_audio_path, segment,
                             self.audio_processor.sample_rate)

                    start_time = j * SEGMENT_DURATION
                    end_time = min((j + 1) * SEGMENT_DURATION, duration)
                    segment_text = self._get_text_for_timerange(
                        textgrids[i], start_time, end_time
                    )

                    segment_summary = self._summarize_segment(
                        temp_audio_path,
                        segment_text,
                        speaker,
                        start_time,
                        end_time
                    )

                    all_segments.append({
                        'speaker': speaker,
                        'segment': j,
                        'start': start_time,
                        'end': end_time,
                        'summary': segment_summary
                    })
            else:
                segment_summary = self._summarize_segment(
                    audio_file,
                    self.textgrid_parser.extract_transcript(textgrids[i]),
                    speaker,
                    0,
                    duration
                )
                all_segments.append({
                    'speaker': speaker,
                    'summary': segment_summary
                })

        meeting_summary = self._generate_final_summary(all_segments)

        return {
            'mode': 'separately',
            'segments': all_segments,
            'final_summary': meeting_summary
        }

    def _summarize_segment(self, audio_path: str,
                           text_context: str,
                           speaker: str,
                           start_time: float,
                           end_time: float) -> str:
        """
        为音频段生成总结（原方法）
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": f"""
                    This is Speaker {speaker}'s audio segment from {start_time:.1f}s to {end_time:.1f}s.
                    
                    Reference transcript: {text_context[:500]}...
                    
                    Please analyze:
                    1. Main topics discussed
                    2. Key points made
                    3. Emotional tone
                    4. Questions or concerns raised
                    
                    Provide a concise summary.
                    """}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=True)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                use_audio_in_video=True,
                return_audio=False,
                max_new_tokens=512,
                do_sample=False
            )

        response = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response

    def _generate_final_summary(self, all_segments: List[Dict]) -> str:
        """
        基于所有段生成最终会议总结
        """
        segment_texts = []
        for seg in all_segments:
            start_time = seg.get('start', 0)
            end_time = seg.get('end', 0)
            speaker_info = seg.get('speaker', 'Mixed')
            
            # 添加说话人统计信息（如果有）
            stats_info = ""
            if 'speaker_stats' in seg:
                stats = seg['speaker_stats']
                stats_info = f"\n(Speaking distribution: {', '.join([f'{k}: {v:.0f}%' for k, v in stats.items()])})"
            
            segment_texts.append(
                f"**Segment [{start_time:.1f}s - {end_time:.1f}s] - {speaker_info}:**{stats_info}\n"
                f"{seg['summary']}"
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SEGMENT_MERGE_PROMPT.format(
                        segment_summaries='\n\n'.join(segment_texts)
                    )}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1200,
                do_sample=False,
                return_audio=False
            )

        final_summary = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return final_summary

    def _get_text_for_timerange(self, textgrid: Dict,
                                start_time: float,
                                end_time: float) -> str:
        """获取指定时间范围内的文本"""
        texts = []
        for interval in textgrid['intervals']:
            if interval['xmin'] >= start_time and interval['xmax'] <= end_time:
                texts.append(interval['text'])
        return ' '.join(texts)

    def save_results(self, results: Dict, output_path: str):
        """保存结果"""
        import json

        # 保存JSON格式
        with open(output_path, 'w', encoding='utf-8') as f:
            # 转换numpy类型为Python原生类型
            def convert_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(i) for i in obj]
                return obj
            
            json.dump(convert_types(results), f, ensure_ascii=False, indent=2)

        # 保存Markdown格式
        md_path = output_path.replace('.json', '.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Meeting Summary\n\n")
            
            # 添加处理模式信息
            if 'mode' in results:
                f.write(f"**Processing Mode:** {results['mode']}\n\n")
            
            # 添加混音音频路径（如果有）
            if 'mixed_audio_path' in results:
                f.write(f"**Mixed Audio:** {results['mixed_audio_path']}\n\n")
            
            f.write(results['final_summary'])

            if SAVE_INTERMEDIATE:
                f.write("\n\n## Segment Details\n\n")
                for seg in results['segments']:
                    speaker = seg.get('speaker', 'Mixed')
                    f.write(f"### Segment {seg.get('segment', 0)} - {speaker}\n")
                    f.write(f"Time: {seg.get('start', 0):.1f}s - {seg.get('end', 0):.1f}s\n\n")
                    
                    # 显示说话人统计（如果有）
                    if 'speaker_stats' in seg:
                        stats = seg['speaker_stats']
                        f.write(f"Speaking distribution: {', '.join([f'{k}: {v:.0f}%' for k, v in stats.items()])}\n\n")
                    
                    f.write(seg['summary'])
                    f.write("\n\n")


def main():
    """主函数"""
    meeting_folder = "../eval-F8N/M005/M005-F8N"
    output_path = "meeting_summary.json"

    # 使用智能混音模式
    summarizer = MeetingSummarizer(use_intelligent_mixing=True)
    
    # 可以选择不同的处理模式:
    # - "intelligent_mixed": 【推荐】智能混音后处理
    # - "separately": 分别处理每个说话人
    results = summarizer.process_meeting(
        meeting_folder, 
        processing_mode="intelligent_mixed"  # 使用智能混音
    )
    
    summarizer.save_results(results, output_path)

    print(f"\n{'='*60}")
    print(f"Summary saved to {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()