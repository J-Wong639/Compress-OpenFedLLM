import pandas as pd

# 加载原始文件，注意分隔符为分号
file_path = "test.csv"
df = pd.read_csv(file_path, sep=';')

# 构造 instruction 和 response 字段
df['instruction'] = "Which newsgroup category does the following message belong to?\nMessage: " + df['text']
df['response'] = df['label_name']

# 仅保留 instruction 和 response 两列
formatted_df = df[['instruction', 'response']]

# 保存为新的 CSV 文件
# output_path = "20ng_instruction_response.csv"
# formatted_df.to_csv(output_path, index=False)




# 读取 CSV 文件
# csv_file = "20ng_instruction_response.csv"
# df = pd.read_csv(csv_file)

# 保存为 JSONL 格式
jsonl_file = "test.jsonl"
formatted_df.to_json(jsonl_file, orient='records', lines=True, force_ascii=False)
