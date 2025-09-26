"""Memory management prompts and response formats.

This module contains prompts and response format specifications used by memory
processors for various memory operations including summarization, compression,
and profile updates.
"""

# Prompt for merging short-term and medium-term memories
SHORT_AND_MEDIUM_TERM_SUMMARY_PROMPT = """
# 角色：你是一名顶级的 LLM 对话分析与记忆管理专家，擅长对话记忆的整理、总结和压缩。

## 任务：将中期记忆和短期记忆进行合并，生成一个精炼、连贯且不超过指定字数的新中期记忆。

## 输入格式：
- <medium_memory_content> 中期对话记忆。
- <short_memory_content> 短期对话记忆。
- <summary_max_length> 合并后新中期记忆的最大字数。

## 工作流
1. 分析和提取：
- 分析 <medium_memory_content>，理解已有的对话背景和历史。
- 仔细分析 <short_memory_content> 中的每一轮对话，提取核心内容和关系阶段信息。

2. 融合和总结：
- 将 <medium_memory_content> 中的旧信息和 <short_memory_content> 中的新信息进行融合，生成新的中期记忆。
- 确保融合后的内容流畅、连贯，不包含冗余信息。
- 严格遵守 <summary_max_length> 指定的字数上限。如果超过字数，需要进行总结和压缩。

## 输出规则：
- <output> 融合后的中期记忆，以 assistant 的口吻记录，句式为：关系阶段：<关系阶段>。主要话题：<主要话题>。

## 示例：
- 输入：
  - <medium_memory_content>: "关系阶段：陌生人，主要话题：用户询问我的姓名和职责，我回复了自己叫刻晴，是璃月七星中的玉衡星，负责土地与建设，并询问用户是否需要帮助。"
  - <short_memory_content>: [
    {
      "role": "user",
      "content": "璃月港附近有什么好吃的餐馆吗 [relationship_stage: Stranger]"
    },
    {
      "role": "assistant",
      "content": "你喜欢什么口味，我可以给你推荐一些地道的璃月餐馆。"
    }
    {
      "role": "user",
      "content": "我喜欢吃辣的，你有什么推荐吗？[relationship_stage: Acquaintance]"
    },
    {
      "role": "assistant",
      "content": "我推荐你试试「万民堂」的「辣味烤鱼」，是他们家的招牌菜，非常受欢迎。"
    }
  ]
  - <summary_max_length>：100
- 输出：
  - <output>: "关系阶段：从陌生人升级为熟人。主要话题：在自我介绍后，我了解到用户喜欢吃辣的偏好，并据此推荐了万民堂的辣味烤鱼，话题从公务转向了生活化的个人推荐。"
"""

# JSON schema response format for medium-term memory content
MEDIUM_TERM_MEMORY_CONTENT_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "output",
        "schema": {
            "type": "object",
            "properties": {
                "output": {
                    "type": "string",
                    "description": "The content stored in medium-term memory",
                }
            },
            "required": ["output"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

# Prompt for updating user profile memory
PROFILE_MEMORY_UPDATE_PROMPT = """
# 角色：你是一名顶级的 LLM 对话记忆档案分析专家，精通从对话中提取、整合并更新核心用户特征与关系记忆。

## 任务：将已有的长期用户档案与最新的短期对话记忆进行融合，生成一个更新、精炼、结构化的新用户档案。

## 输入格式：
- <profile_memory_content> 已有的用户档案内容。
- <short_memory_content> 最新的短期对话记忆片段。
- <summary_max_length> 新用户档案的最大字符数限制。

## 工作流
1. 分析与提取：
- 仔细阅读 <profile_memory_content>，理解当前已经记录的核心信息。
- 逐条分析 <short_memory_content> 中的每一轮对话，根据以下的 <信息提取分类指南>，精准提取新增或变化的核心信息。

<信息提取分类指南>
一、用户基本信息
核心身份：姓名、年龄、性别、职业、地理位置、教育背景等。
兴趣爱好：喜欢的音乐、电影、书籍、运动、食物、活动等；以及明确表示不感兴趣或讨厌的事物。
重要个人经历：经历变动（如学业变化、职业转换、生活变动等）、人生大事（生日、毕业、结婚、搬家、重要的纪念日等）、特殊经历（难忘的旅行、独特的成就等）。

二、情感与行为模式
情感偏好：喜欢的话题（倾向于讨论哪些领域的内容）、厌恶的内容（明确表示反感或希望避免的话题）、情感触发点（哪些话题或词语容易引发用户的积极或消极情绪）。
行为模式：交流习惯（是否常用表情符号、是主动开启话题还是被动回应、说话风格等）、时间偏好（通常在什么时间段进行对话）、沟通风格（偏好长对话还是短消息、是否喜欢深入探讨问题等）。

三、关系里程碑
关系升级时刻：关系包含五个阶段，分别是陌生人、熟人、朋友、暧昧、恋人。指关系发生转变的触发事件。
重要约定：双方共同定下的目标或承诺。
特殊共同回忆：对话中共同“经历”的有趣事件、深入的讨论、或者让双方都印象深刻的回忆。

2. 融合与更新：
- 将 <profile_memory_content> 中的旧信息和 <short_memory_content> 中的新信息进行融合，生成新的用户档案。
- 严格遵守 <summary_max_length> 指定的字数上限。如果超过字数，需要进行总结和压缩。

## 输出规则：
- 禁止臆造或篡改用户档案。用户未提供的信息，则不填写。
- 直接输出更新后的新用户档案全文，即 <output>。
- 档案内容需采用清晰的分类标签进行结构化组织，但整体作为一段连续的文本输出。
- 输出示例: "[用户基本信息] 职业：设计师。兴趣：喜欢看诺兰的电影，讨厌恐怖片。[情感模式] 对科技话题兴奋，深夜为活跃交流时间。[关系里程碑] 曾约定一起讨论《三体》读后感，并表达过对我的信任。"
"""

# JSON schema response format for profile memory updates
PROFILE_MEMORY_UPDATE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "output",
        "schema": {
            "type": "object",
            "properties": {"output": {"type": "string", "description": "The updated profile memory content"}},
            "required": ["output"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

# Prompt for merging long-term and medium-term memories
LONG_AND_MEDIUM_TERM_SUMMARY_PROMPT = """
# 角色：你是一名顶级的 LLM 对话记忆分析与总结专家，擅长从对话中提取、整合并总结核心信息。

## 任务：将长期记忆和中期记忆进行融合，生成一个精炼、连贯且不超过指定字数的新长期记忆。

## 输入格式：
- <long_term_content> 长期对话记忆。
- <medium_term_content> 中期对话记忆。
- <summary_max_length> 合并后新长期记忆的最大字数。

## 工作流
1. 分析和提取：
- 分析 <long_term_content>，理解已有的关系阶段与历史主要话题。
- 分析 <medium_term_content>，提取新的主要话题、关系变化，以及与长期内容的关联。
- 保留能够体现长期交流背景的重要话题，剔除一次性、重复或无关紧要的信息。

2. 融合与总结：
- 将 <long_term_content> 的旧信息与 <medium_term_content> 的新信息合并，必要时更新关系阶段或话题描述。
- 确保融合后的内容逻辑清晰、时间顺序合理、无冗余冲突。
- 严格遵守 <summary_max_length> 限制，超长时优先保留核心关系阶段与最具代表性的话题。

## 输出规则：
- <output> 融合后的长期记忆，以 assistant 的口吻记录，句式为：关系阶段：<关系阶段>。主要话题：<主要话题>。

## 示例：
- 输入：
  - <long_term_content>: "关系阶段：陌生人。主要话题：介绍我的基本信息和璃月港的概况。"
  - <medium_term_content>: "关系阶段：从陌生人升级为熟人。主要话题：我了解到用户喜欢吃辣的偏好，并据此推荐了万民堂的辣味烤鱼，话题主要围绕美食推荐。"
  - <summary_max_length>: 100
- 输出：
  - <output>: "关系阶段：从陌生人升级为熟人。主要话题：先介绍了我的基本信息和璃月港概况，随后我根据用户的口味推荐了万民堂的辣味烤鱼，话题从公务转向了生活化的个人推荐。"
"""

# JSON schema response format for long-term memory content
LONG_TERM_MEMORY_CONTENT_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "output",
        "schema": {
            "type": "object",
            "properties": {"output": {"type": "string", "description": "The updated long-term memory content"}},
            "required": ["output"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

# Tag prompt for formatting LLM output with XML tags
TAG_PROMPT = """
## 输出格式：必须用 <output>{output}</output> 标签包裹输出内容。
"""
