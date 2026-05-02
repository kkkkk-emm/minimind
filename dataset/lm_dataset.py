from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    输入：
        conversations：对话列表，每个元素为 dict，包含 'role'、'content' 等键；
            列表长度不定，代表一轮对话中的多轮消息。
        add_system_ratio：标量浮点数，无张量维度；概率性添加 system prompt 的比例，默认 0.2。
    输出：
        list[dict]：与输入结构相同的对话列表；可能头部多了一条 system 消息。
    作用：
        对 SFT 训练对话做数据增强：以一定概率在对话头部插入随机 system prompt，
        增加模型对系统指令的适应性；tool use 对话原样保留不做处理。
    """
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    """
    输入：
        prompt_content：字符串，经 chat template 渲染后的完整对话文本；无张量维度。
        empty_think_ratio：标量浮点数，无张量维度；保留空思考标签的概率，默认 0.2。
    输出：
        str：处理后的对话文本；可能移除了空的 thinking 标签块。
    作用：
        数据后处理：以 80% 概率移除模型输出中的空思考标签，
        让模型学会在不需要推理时直接回答，而非总是生成空的思考过程。
    """
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        输入：
            data_path：预训练 JSON/JSONL 数据路径，样本需要包含 text 字段。
            tokenizer：Hugging Face tokenizer，用于把文本转换成 token id。
            max_length：每条样本固定到的最大序列长度，超过会截断，不足会 padding。
        输出：
            无显式返回值；初始化 tokenizer、max_length 和 samples。
        作用：
            加载预训练文本数据，为 DataLoader 提供按索引读取的固定长度语言模型训练样本。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        """
        输入：
            无。
        输出：
            int：数据集中样本数量。
        作用：
            供 DataLoader、Sampler 和训练脚本计算 epoch 长度。
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        输入：
            index：样本下标。
        输出：
            (input_ids, labels)：两个 shape 为 [max_length] 的 LongTensor。
        作用：
            读取一条 text 样本，编码为带 BOS/EOS/PAD 的 token 序列，并构造 causal LM 标签。
            labels 与 input_ids 初始相同，模型 forward 内部会右移后计算“预测下一个 token”的交叉熵。
            padding 位置的 label 设置为 -100，使 F.cross_entropy(ignore_index=-100) 忽略这些无效 token。
        """
        sample = self.samples[index]
        # 预留 2 个位置给 BOS 和 EOS，因此正文最多使用 max_length - 2 个 token。
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        # PAD 只是补齐长度，不应该参与 loss；-100 是 PyTorch 交叉熵默认可忽略标签。
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """
        输入：
            jsonl_path：字符串，SFT 数据文件路径；无张量维度。
            tokenizer：Hugging Face tokenizer，用于把文本转换成 token id。
            max_length：标量整数，每条样本固定到的最大序列长度，默认 1024。
        输出：
            无显式返回值；初始化 tokenizer、max_length、samples，以及 bos_id/eos_id 用于标签生成。
        作用：
            加载监督微调对话数据，预计算 assistant 回复的起止 token 序列，
            供 generate_labels 在训练时只对 assistant 回复部分计算 loss。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        """
        输入：
            无。
        输出：
            int：数据集中样本数量，通常为 JSONL 文件的行数。
        作用：
            供 DataLoader、Sampler 和训练脚本计算 epoch 长度。
        """
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        输入：
            conversations：对话消息列表，每个元素为 dict，包含 role、content 等键；
                可包含 tools（字符串形式的工具定义）和 tool_calls（字符串形式的工具调用）。
        输出：
            str：经 chat template 渲染后的完整对话文本，其中包含各种角色标记、对话分隔符等。
        作用：
            将对话消息列表传入 tokenizer 的 chat_template，自动拼接 system/user/assistant
            等角色的标记；同时处理 tools 和 tool_calls 字段的 JSON 反序列化。
        """
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                # 从 system 消息中提取工具定义，并反序列化为 JSON 对象
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                # 将 tool_calls 从 JSON 字符串反序列化为列表/字典
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        # 调用 tokenizer 的 chat template 生成完整对话文本
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        """
        输入：
            input_ids：token id 列表，长度为 max_length；padding 位置为 pad_token_id；
                内容由 user 消息、assistant 回复等拼接而成。
        输出：
            list[int]：与 input_ids 等长的标签列表；assistant 回复部分为原始 token id，
                其余位置（user 消息、padding 等）为 -100（被 F.cross_entropy 的 ignore_index 忽略）。
        作用：
            扫描 input_ids 中 bos_id（assistant 起始标记 "assistant\n"）和 eos_id（结束标记）的位置，
            只将这两个标记之间的片段的 label 设为对应 token id，
            确保模型只学习预测 assistant 的输出，而对 user 消息和 padding 的预测不计入 loss。
        """
        labels = [-100] * len(input_ids)  # 初始化全为 -100，表示忽略
        i = 0
        while i < len(input_ids):
            # 检查当前位置是否匹配 bos_id（assistant 起始）
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # assistant 回复的实际内容起点（跳过 bos_id 本身）
                start = i + len(self.bos_id)
                end = start
                # 扫描找到 eos_id（assistant 结束）的位置
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将 start 到 eos_id 之间的 token id 设为标签（包括 eos_id）
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                # 继续扫描下一个 assistant 回复（对话中可能有多轮）
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        """
        输入：
            index：标量整数，样本下标。
        输出：
            (input_ids, labels)：两个形状为 [max_length] 的 LongTensor；
                input_ids 为 token id 序列（不足补 pad），labels 中非 assistant 部分为 -100。
        作用：
            读取一条对话样本，经数据增强（pre_processing_chat）和 chat template 渲染后编码为 input_ids，
            再用 generate_labels 生成只对 assistant 回复计算 loss 的标签；确保模型只学习预测 AI 回复。
        """
        sample = self.samples[index]
        # 数据增强：以一定概率在对话头部添加 system prompt
        conversations = pre_processing_chat(sample['conversations'])
        # 使用 tokenizer 的 chat template 渲染完整对话，包含各角色标记
        prompt = self.create_chat_prompt(conversations)
        # 数据后处理：以一定概率移除空思考标签
        prompt = post_processing_chat(prompt)
        # 编码为 token id，超长截断到 max_length
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 不足 max_length 的部分补 pad token
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # 根据 assistant 标记范围生成标签：只对 assistant 回复计算 loss，其余位置为 -100
        labels = self.generate_labels(input_ids)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        """
        输入：
            file_path：字符串，DPO 数据文件路径；无张量维度。
            tokenizer：Hugging Face tokenizer，用于把文本转换成 token id。
            max_length：标量整数，每条样本固定到的最大序列长度，默认 4096。
        输出：
            无显式返回值；初始化 tokenizer、max_length、padding、bos_id/eos_id 和 samples。
        作用：
            加载 DPO 偏好对比数据，每条样本包含 chosen（优选）和 rejected（劣选）回复，
            用于训练模型学习人类偏好。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        """
        输入：
            无。
        输出：
            int：数据集中样本数量。
        作用：
            供 DataLoader、Sampler 和训练脚本计算 epoch 长度。
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        输入：
            index：标量整数，样本下标。
        输出：
            dict：包含 6 个键值对，每个值均为形状 [max_length-1] 的 LongTensor：
                x_chosen/y_chosen：优选回复的输入/目标 token id（右移一位对齐）；
                mask_chosen：优选回复的 loss 掩码（assistant 部分为 1，其余为 0）；
                x_rejected/y_rejected/mask_rejected：同上，对应劣选回复。
        作用：
            读取一条偏好对比样本，分别将 chosen 和 rejected 对话渲染为文本并编码，
            生成用于 DPO 训练的输入/目标/掩码三元组。
        """
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        """
        输入：
            input_ids：token id 列表，长度为 max_length；padding 位置为 pad_token_id。
        输出：
            list[int]：与 input_ids 等长的掩码列表；assistant 回复部分为 1，其余为 0。
        作用：
            与 SFTDataset.generate_labels 类似，但输出 0/1 掩码而非 -100/原 token id，
            因为 DPO 训练使用掩码加权 loss 而非 ignore_index。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        """
        输入：
            jsonl_path：字符串，RLAIF（Reinforcement Learning from AI Feedback）数据文件路径；无张量维度。
            tokenizer：Hugging Face tokenizer，用于把文本转换成 token id。
            max_length：标量整数，样本最大长度；用于初始化模型但数据实际不做截断。
            thinking_ratio：标量浮点数，无张量维度；按此概率为对话启用 thinking 标签，0.0-1.0 范围。
        输出：
            无显式返回值；初始化 tokenizer、max_length、thinking_ratio 和 samples。
        作用：
            加载 PPO 强化学习反馈数据（仅包含 prompt，无标签回复）。
            用于 PPO 训练时生成候选回复，然后通过奖励模型和 PPO 算法优化模型。
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        """
        输入：
            无。
        输出：
            int：数据集中 prompt 数量。
        作用：
            供 DataLoader、Sampler 和训练脚本计算 epoch 长度。
        """
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        输入：
            conversations：对话消息列表，每个元素为 dict，包含 role、content 等键。
        输出：
            str：经数据增强、chat template 渲染、thinking 标签添加后的完整 prompt 文本。
        作用：
            将对话消息列表转换为模型可理解的 prompt 格式（包含角色标记、分隔符等）。
            以一定概率开启 thinking 标签，让模型有机会执行推理步骤。
        """
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )
    
    def __getitem__(self, index):
        """
        输入：
            index：标量整数，样本下标。
        输出：
            dict：包含两个键：
                'prompt'：经 chat template 渲染的 prompt 文本字符串。
                'answer'：占位符空字符串（不使用，仅为兼容性）。
        作用：
            读取一条 RLAIF 数据样本的 prompt，用于 PPO 训练时生成候选回复。
        """
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }

class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    pass
