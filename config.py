# 模型配置
MODEL_PATH = "../Qwen3-Omni/Qwen3-Omni-30B-A3B-Instruct"
USE_FLASH_ATTN = False
DEVICE = "cuda"

# 音频处理配置
SEGMENT_DURATION = 300  # 5分钟一段
MAX_AUDIO_LENGTH = 1800  # 30分钟
SAMPLE_RATE = 16000

# 输出配置
OUTPUT_FORMAT = "markdown"  # 或 "json"
SAVE_INTERMEDIATE = True  # 保存中间结果

# Prompt模板
MEETING_SUMMARY_PROMPT = """
You are analyzing a meeting recording with 3 participants discussing graduate school choices.

Audio Information:
- Duration: {duration} seconds
- Participants: Speaker 192, Speaker 193, Speaker 194
- Topic: Graduate school entrance examination discussion

Please provide a comprehensive meeting summary including:

1. **Meeting Overview**
   - Main topic and objectives
   - Key discussion themes
   
2. **Speaker Contributions**
   - Speaker 192: Main points and opinions
   - Speaker 193: Main points and opinions  
   - Speaker 194: Main points and opinions
   
3. **Key Discussion Points**
   - Important topics covered
   - Consensus reached
   - Disagreements or debates
   
4. **Emotional Dynamics**
   - Overall meeting tone
   - Individual speaker emotions
   - Interaction patterns
   
5. **Conclusions and Action Items**
   - Key takeaways
   - Decisions made
   - Next steps discussed

Please structure the summary clearly and distinguish between different speakers' contributions.
"""

SEGMENT_MERGE_PROMPT = """
Merge the following segment summaries into a coherent overall summary:

{segment_summaries}

Maintain speaker distinctions and chronological flow.
"""