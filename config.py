# 模型配置
MODEL_PATH = "Qwen/Qwen2.5-Omni-7B"
USE_FLASH_ATTN_QWEN25 = True  # Qwen2.5-Omni 启用 flash attention
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
Based on the following segment summaries, create a meeting summary.

Segment summaries:
{segment_summaries}

TASK: Write ONE cohesive paragraph (200-300 words) summarizing the entire meeting, then map each sentence to time ranges.

CRITICAL RULES:
1. Write as ONE CONTINUOUS PARAGRAPH - DO NOT separate by speaker
2. DO NOT write "Speaker 192 discusses..." or "Speaker 193 says..." - just describe what was discussed
3. Keep it between 200-300 words MAXIMUM
4. Cover: main topics, key decisions, emotional dynamics, conclusions
5. Then map EVERY sentence from your summary to time ranges

FORMAT EXACTLY LIKE THIS:

## COMPREHENSIVE SUMMARY

The speakers discuss the intense pressure and competitive nature of the job market, particularly in China, which is driving many graduates to pursue postgraduate degrees. A primary motivation for one speaker is to avoid the job market and prepare for a career civil servant exam, leading her to choose a Master's in Social Work, which she believes offers a clearer path to a government job. The conversation highlights the increasing difficulty of entering professions like teaching, where a master's degree is now a standard requirement, and the oversaturation of fields like IT, where even skilled graduates struggle to find work. The speakers express anxiety and frustration about the competitive job market, with one noting that 11 million graduates compete for a limited number of positions. They debate the value of a master's degree, acknowledging it can be a necessary threshold but questioning if it guarantees employment. The discussion also touches on the immense dedication required for success, with anecdotes of peers who studied rigorously for years, contrasted with those who use the exam as an excuse to avoid work.

## TIME-STAMPED SENTENCE MAPPING

Sentence 1: "The speakers discuss the intense pressure and competitive nature of the job market, particularly in China, which is driving many graduates to pursue postgraduate degrees." -> [300.0s, 600.0s]
Sentence 2: "A primary motivation for one speaker is to avoid the job market and prepare for a career civil servant exam, leading her to choose a Master's in Social Work, which she believes offers a clearer path to a government job." -> [0.0s, 300.0s]
Sentence 3: "The conversation highlights the increasing difficulty of entering professions like teaching, where a master's degree is now a standard requirement, and the oversaturation of fields like IT, where even skilled graduates struggle to find work." -> [1200.0s, 1500.0s]

[Continue mapping ALL sentences from your summary above]
"""

# Qwen2.5-Omni 专用配置
QWEN25_SYSTEM_PROMPT = """You are a helpful assistant capable of understanding audio, visual, and text inputs, 
and generating comprehensive meeting summaries. You excel at providing structured summaries with source attribution."""
