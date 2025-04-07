import os
import pandas as pd
import json
import openai
import time

# 配置参数（请替换实际API Key）
openai.api_key = "sk-agsvmeamgonduteaplipmmiyhriniwkbhogfcymilqxcvoxz"
openai.api_base = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

# 文件路径
input_path = "宋1_simplified.xlsx"
output_path = "result.jsonl"
log_path = "error.log"

# 读取数据（仅保留必要字段）
df = pd.read_excel(input_path, usecols=["name", "given_name1", "given_name2"])

# 判断是否已有部分结果，若有则跳过已处理行（中断后续接）
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        processed_lines = f.readlines()
    processed_count = len(processed_lines)
    print(f"检测到已处理 {processed_count} 条记录，将从第 {processed_count + 1} 行继续处理。")
    df = df.iloc[processed_count:]
else:
    processed_count = 0

df = df.reset_index(drop=True)  # 重置索引

# 修改系统提示：批量处理，统一使用1p、2p表示词性，要求输出为JSON数组
system_prompt = (
    "作为古汉语姓名分析专家，请对以下名字的给定的两个字进行词性标注。词性标注需要参考1.古汉语语法知识；2.在名字的语境中的具体含义和语法功能。"
    "每个名字包含两部分输入：字1和字2。请按以下规则处理："
    "1.词性标注选项：n-名词/v-动词/a-形容词/d-副词"
    "2.输出严格符合JSON格式，要求所有键和值均使用双引号"
    "对于每个名字，请输出一个JSON对象，键为\"1p\"和\"2p\"；"
    "若名字中只有一个字，输出示例：苏轼 → {\"1p\": \"n\", \"2p\": \"\"}。"
    "最终输出应为一个JSON数组，其中每个元素对应一个名字的词性标注，且顺序与输入顺序一致。"
    "注意：保持JSON结构完整，不添加任何额外解释。"
)

batch_size = 10  # 每次请求处理10条记录
total = len(df)
batch_count = (total + batch_size - 1) // batch_size  # 计算批次数

# 打开输出和日志文件（追加模式，方便多次运行）
with open(output_path, "a", encoding="utf-8") as out_file, open(log_path, "a", encoding="utf-8") as log_file:
    for batch_index in range(batch_count):
        start_idx = batch_index * batch_size
        end_idx = min(start_idx + batch_size, total)
        batch_df = df.iloc[start_idx:end_idx]

        # 构造批量输入：每个名字的输入格式为"字1:{...}\n字2:{...}"
        # 每个记录之间用一个空行分隔，保持顺序与DataFrame一致
        user_input_lines = []
        for _, row in batch_df.iterrows():
            char1 = str(row['given_name1'])
            char2 = str(row['given_name2']) if pd.notnull(row['given_name2']) else ""
            user_input_lines.append(f"字1:{char1}\n字2:{char2}")
        user_input = "\n\n".join(user_input_lines)

        try:
            # 控制请求速率：每秒10次请求，即每次请求间隔0.1秒
            time.sleep(0.1)

            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0,
                max_tokens=200  # 适当增加以确保返回完整的JSON数组
            )

            content = response.choices[0].message.content.strip()
            try:
                # 解析返回的JSON数组，每个元素对应一行记录的词性标注
                results = json.loads(content)
                if not isinstance(results, list):
                    raise ValueError("返回结果不是JSON数组")
                # 若返回的结果条数与实际输入不符，记录日志
                if len(results) != (end_idx - start_idx):
                    log_file.write(
                        f"批次 {batch_index+1} (行 {start_idx+processed_count+1}-{end_idx+processed_count}) 返回结果数 {len(results)} 与输入数 {(end_idx - start_idx)} 不符，返回内容: {content}\n"
                    )
                    # 对数量不符的情况，补全空结果
                    results = results + [{"1p": "", "2p": ""}] * ((end_idx - start_idx) - len(results))
            except Exception as json_err:
                log_file.write(
                    f"批次 {batch_index+1} (行 {start_idx+processed_count+1}-{end_idx+processed_count}) JSON解析失败，返回内容: {content}\n"
                )
                results = [{"1p": "", "2p": ""} for _ in range(end_idx - start_idx)]
        except Exception as e:
            log_file.write(
                f"批次 {batch_index+1} (行 {start_idx+processed_count+1}-{end_idx+processed_count}) 请求失败：{str(e)}\n"
            )
            results = [{"1p": "", "2p": ""} for _ in range(end_idx - start_idx)]

        # 构造输出记录，并写入文件
        for i, (_, row) in enumerate(batch_df.iterrows()):
            record = {
                "n": row['name'],  # 原name字段
                "1": str(row['given_name1']),
                "2": str(row['given_name2']) if pd.notnull(row['given_name2']) else "",
                "1p": results[i].get("1p", ""),
                "2p": results[i].get("2p", "")
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        out_file.flush()

        # 每处理100条记录，在终端汇报一次进度
        if (start_idx + batch_size) % 100 == 0 or (batch_index == batch_count - 1):
            processed_total = start_idx + batch_size + processed_count
            print(f"已处理 {min(processed_total, processed_count + total)} 条记录。")