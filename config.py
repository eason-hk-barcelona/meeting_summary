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
Based on the following segment summaries with their time ranges, create a detailed meeting summary with time-stamped sentences.

Segment summaries:
{segment_summaries}

TASK: Create a concise summary (200-300 words) followed by sentence-to-time mapping.

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:

## COMPREHENSIVE SUMMARY

[Write your 200-300 word summary here covering main topics, key points from each speaker, decisions, and emotional dynamics]

## TIME-STAMPED SENTENCE MAPPING

Sentence 1: "First sentence from your summary above." -> [0s, 300s]
Sentence 2: "Second sentence from your summary above." -> [300s, 600s]
Sentence 3: "Third sentence from your summary above." -> [600s, 900s]
[Continue for ALL sentences in your summary]

CRITICAL REQUIREMENTS:
1. Write the complete summary first (200-300 words)
2. Then map EVERY sentence to time ranges based on the segment data
3. Use exact sentence text from your summary
4. Map time ranges logically to the segment information provided
5. Use format: Sentence X: "exact text" -> [XXXs, XXXs]

The segments span from 0s to approximately 1810s across three speakers (192, 193, 194).
"""
