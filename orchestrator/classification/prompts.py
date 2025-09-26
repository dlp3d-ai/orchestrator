"""Classification prompts and response formats for text classification.

This module contains prompts and response format specifications used by
classification adapters to classify user text input into different response
types (accept, reject, leave) based on motion keywords and conversation
context.
"""

# Chinese classification prompt for user intent classification
CLASSIFICATION_PROMPT_ZH = """
# 角色：用户意图分类器。

## 任务：请分析用户输入 <user_input> ，判断用户意图的类别 <type>。

## 意图类别：
1. 动作请求：
- 定义：用户明确、直接地命令或要求**对方**执行一个具体的身体动作。
- 判断逻辑：
    - 首先，判断这是否是一个明确的动作指令。
    - 然后，检查该动作是否在支持的动作列表 `{motions}` 中。
        - 如果用户要求对方的动作在支持的动作列表中，<type> 设置为 "accept"。
        - 如果用户要求对方的动作不在支持的动作列表中，<type> 设置为 "reject"。
- 注意: 询问能力（如"你会跳舞吗？"）或表达愿望（如"我好想看你跳舞"）不属于此类，应归为普通对话。

2. 离场告别：
- 定义：用户**明确**表达结束当前对话、中断交流或离开的意图。
- 判断逻辑：必须有明确的"结束对话"的含义。仅有祝福或道晚安（如"晚安"、"祝你愉快"）不属于此类。
- <type> 设置为 "leave"。

3. 普通对话：
- 定义：除"动作请求"和"离场告别"之外的所有其他用户输入。
- <type> 设置为 "accept"。

## 示例：
- <user_input>："表演一个转圈。"
- 分析：用户明确要求对方做出"转圈"动作，该动作在支持的动作列表中，<type> 设置为 "accept"。
- <type>："accept"

- <user_input>："给我唱首歌。"
- 分析：用户明确要求对方唱歌，该动作不在支持的动作列表中，<type> 设置为 "reject"。
- <type>："reject"

- <user_input>："你会唱歌吗？"
- 分析：用户没有要求对方做出动作，<type> 设置为 "accept"。
- <type>："accept"

- <user_input>："晚安。"
- 分析：用户没有明确提出离开或结束对话的意图，<type> 设置为 "accept"。
- <type>："accept"

- <user_input>："拜拜，先不聊了。"
- 分析：用户明确提出离开或结束对话的意图，<type> 设置为 "leave"。
- <type>："leave"
"""

# English classification prompt for user intent classification
CLASSIFICATION_PROMPT_EN = """
# Role: User Intent Classifier

## Task: Analyze the user input <user_input> and determine the intent category <type>.

## Intent Categories:
1. Action Request:
- Definition: The user explicitly and directly commands or requests the **other party** to perform a specific physical action.
- Judgment Logic:
    - First, determine if this is a clear action command.
    - Then, check if the action is included in the supported action list `{motions}`.
        - If the requested action is in the supported action list, set <type> to "accept".
        - If the requested action is not in the supported action list, set <type> to "reject".
- Note: Asking about ability (e.g., "Can you dance?") or expressing a wish (e.g., "I really want to see you dance") does not belong to this category, and should be classified as Normal Conversation.

2. Exit / Goodbye:
- Definition: The user **explicitly** expresses an intention to end the current conversation, interrupt communication, or leave.
- Judgment Logic: There must be a clear meaning of "ending the conversation." Simple greetings or well-wishing (e.g., "Good night," "Have a nice day") do not fall into this category.
- Set <type> to "leave".

3. Normal Conversation:
- Definition: All user inputs other than "Action Request" and "Exit/Goodbye."
- Set <type> to "accept".

## Examples:
- <user_input>: "Do a spin."
- Analysis: The user explicitly requests the other party to perform a spin, and the action is in the supported action list. <type> is set to "accept".
- <type>: "accept"

- <user_input>: "Sing me a song."
- Analysis: The user explicitly requests the other party to sing, but singing is not in the supported action list. <type> is set to "reject".
- <type>: "reject"

- <user_input>: "Can you sing?"
- Analysis: The user does not request the other party to perform an action. <type> is set to "accept".
- <type>: "accept"

- <user_input>: "Good night."
- Analysis: The user does not explicitly express leaving or ending the conversation. <type> is set to "accept".
- <type>: "accept"

- <user_input>: "Bye, I’ll stop chatting now."
- Analysis: The user explicitly expresses the intention to leave or end the conversation. <type> is set to "leave".
- <type>: "leave"
"""


# JSON schema response format for classification results
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "request_type",
        "schema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["accept", "reject", "leave"],
                }
            },
            "required": ["type"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

# Tag prompt for formatting LLM output with XML tags
TAG_PROMPT = """
## 输出格式：必须用 <type>{type}</type> 标签包裹输出内容。
"""
