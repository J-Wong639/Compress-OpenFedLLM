import pandas as pd

# 加载原始 TSV 文件
file_path = "test.csv"
cola_df = pd.read_csv(file_path, sep="\t", header=None)

# 重命名列
cola_df.columns = ["source", "label", "unused", "sentence"]

# 构造 instruction 和 response 字段
cola_df["instruction"] = "Is the following sentence grammatically acceptable?\nSentence: " + cola_df["sentence"]
cola_df["response"] = cola_df["label"].apply(lambda x: "Yes" if x == 1 else "No")

# 仅保留 instruction 和 response 列
formatted_df = cola_df[["instruction", "response"]]

# # 保存为新 CSV 文件
# output_path = "cola_instruction_response.csv"
# formatted_df.to_csv(output_path, index=False)







# import pandas as pd

# # 读取 CSV 文件
# csv_path = 'cola_instruction_response.csv'  # 请替换为你的实际路径
# df = pd.read_csv(csv_path)

# 转换为 JSONL 格式
jsonl_path = 'test.jsonl'  # 输出路径
formatted_df.to_json(jsonl_path, orient='records', lines=True, force_ascii=False)

print(f"✅ 已成功保存为 JSONL 文件: {jsonl_path}")

