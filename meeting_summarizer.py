import os
from typing import List, Dict
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import soundfile as sf
import numpy as np

from audio_processor import AudioProcessor
from textgrid_parser import TextGridParser
from config import *

class MeetingSummarizer:
    def __init__(self):
        """初始化模型和处理器"""
        print("Loading Qwen3-Omni model...")
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2" if USE_FLASH_ATTN else "eager"
        )
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
        
        self.audio_processor = AudioProcessor()
        self.textgrid_parser = TextGridParser()
        
    def process_meeting(self, meeting_folder: str) -> Dict:
        """
        处理整场会议
        Args:
            meeting_folder: 包含音频和TextGrid文件的文件夹
        Returns:
            会议总结字典
        """
        # 1. 收集文件
        speakers = ['192', '193', '194']
        audio_files = []
        textgrid_files = []
        
        for speaker in speakers:
            audio_files.append(os.path.join(meeting_folder, f'M005-F8N-{speaker}.wav'))
            textgrid_files.append(os.path.join(meeting_folder, f'M005-F8N-{speaker}.TextGrid'))
        
        # 2. 解析TextGrid获取结构化信息
        print("Parsing TextGrid files...")
        textgrids = [self.textgrid_parser.parse(tg) for tg in textgrid_files]
        
        # 3. 处理音频
        print("Processing audio files...")
        
        # 方案A: 分别处理每个说话人，然后合并总结
        if True:  # 可配置
            return self._process_separately(audio_files, textgrids, speakers)
        
        # 方案B: 合并音频后统一处理
        else:
            return self._process_merged(audio_files, textgrids)
    
    def _process_separately(self, audio_files: List[str], 
                           textgrids: List[Dict], 
                           speakers: List[str]) -> Dict:
        """
        分别处理每个说话人的音频
        """
        all_segments = []
        
        for i, (audio_file, speaker) in enumerate(zip(audio_files, speakers)):
            print(f"Processing Speaker {speaker}...")
            
            # 加载音频
            audio, duration = self.audio_processor.load_audio(audio_file)
            
            # 分段处理
            if duration > SEGMENT_DURATION:
                segments = self.audio_processor.segment_audio(audio, SEGMENT_DURATION)
                
                for j, segment in enumerate(segments):
                    print(f"  Processing segment {j+1}/{len(segments)}...")
                    
                    # 保存临时音频文件
                    temp_audio_path = f"/tmp/segment_{speaker}_{j}.wav"
                    sf.write(temp_audio_path, segment, self.audio_processor.sample_rate)
                    
                    # 获取该段对应的文本
                    start_time = j * SEGMENT_DURATION
                    end_time = min((j + 1) * SEGMENT_DURATION, duration)
                    segment_text = self._get_text_for_timerange(
                        textgrids[i], start_time, end_time
                    )
                    
                    # 生成段总结
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
                # 音频较短，直接处理
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
        
        # 4. 生成综合会议总结
        meeting_summary = self._generate_final_summary(all_segments)
        
        return {
            'segments': all_segments,
            'final_summary': meeting_summary
        }
    
    def _summarize_segment(self, audio_path: str, 
                          text_context: str,
                          speaker: str,
                          start_time: float,
                          end_time: float) -> str:
        """
        为音频段生成总结
        """
        # 构建消息
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

        # Set whether to use audio in video
        USE_AUDIO_IN_VIDEO = True
        
        # 处理输入
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = self.processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        
        # 生成总结 - 禁用音频输出，只要文本
        self.model.disable_talker()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                return_audio=False
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
        # 组织段总结
        segment_texts = []
        for seg in all_segments:
            segment_texts.append(
                f"[Speaker {seg['speaker']}, "
                f"{seg.get('start', 0):.1f}s-{seg.get('end', 0):.1f}s]: "
                f"{seg['summary']}"
            )
        
        # 构建最终总结的prompt
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
        
        # 生成最终总结
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        
        # 确保禁用音频输出
        self.model.disable_talker()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
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
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存Markdown格式
        md_path = output_path.replace('.json', '.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Meeting Summary\n\n")
            f.write(results['final_summary'])
            
            if SAVE_INTERMEDIATE:
                f.write("\n\n## Segment Details\n\n")
                for seg in results['segments']:
                    f.write(f"### Speaker {seg['speaker']} - Segment {seg.get('segment', 0)}\n")
                    f.write(f"Time: {seg.get('start', 0):.1f}s - {seg.get('end', 0):.1f}s\n\n")
                    f.write(seg['summary'])
                    f.write("\n\n")

# 主函数
def main():
    meeting_folder = "../eval-F8N/M005/M005-F8N"
    output_path = "meeting_summary.json"
    
    summarizer = MeetingSummarizer()
    results = summarizer.process_meeting(meeting_folder)
    summarizer.save_results(results, output_path)
    
    print(f"Summary saved to {output_path}")

if __name__ == "__main__":
    main()