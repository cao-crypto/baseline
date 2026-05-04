import os
import json
import torch
import time
from openai import OpenAI

# -------------------------- 阿里云百炼API配置（需用户修改）--------------------------
API_KEY = "sk-025d76005d994866984569b539541894" # 替换为你的阿里云百炼API Key
# 地域选择：北京地域用这个base_url；新加坡地域替换为 "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "text-embedding-v4"  # 阿里云百炼嵌入模型（官网实例指定模型）
BATCH_SIZE = 10  # 批量大小（遵循之前API限制：≤10，避免400错误）
MAX_RETRIES = 3  # 每批失败后的最大重试次数
RETRY_DELAY = 1  # 重试间隔（秒），避免触发API频率限制

# -------------------------- 初始化阿里云百炼客户端 --------------------------
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


# -------------------------- 分批生成嵌入核心函数 --------------------------
def batch_generate_embeddings(texts):
    """
    基于阿里云百炼API，分批生成文本嵌入
    texts: 待生成嵌入的文本列表（从JSON读取）
    return: 所有文本的嵌入向量列表（顺序与texts一致）
    """
    all_embeddings = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE  # 总批次数（向上取整）

    print(f"开始生成嵌入：共 {len(texts)} 条文本，分 {total_batches} 批处理（每批≤{BATCH_SIZE}条）")

    for batch_idx in range(total_batches):
        # 切片获取当前批次文本（最后一批可能不足10条）
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(texts))
        batch_texts = texts[start_idx:end_idx]
        current_batch = batch_idx + 1

        print(f"\n处理批次 {current_batch}/{total_batches}（文本范围：{start_idx + 1}-{end_idx}）")

        # 文本合法性校验（避免空文本/超长文本导致API报错）
        valid_batch_texts = []
        for idx, text in enumerate(batch_texts):
            # 过滤空文本/纯空格文本
            if not text or text.strip() == "":
                print(f"警告：批次{current_batch}第{idx + 1}条文本为空，已跳过")
                continue
            # 阿里云text-embedding-v4单条文本上限约8192字符，超长截断
            if len(text) > 8192:
                print(f"警告：批次{current_batch}第{idx + 1}条文本超长（{len(text)}字符），已截断为8192字符")
                text = text[:8192]
            valid_batch_texts.append(text)

        # 若当前批次无有效文本，直接跳过
        if not valid_batch_texts:
            print(f"批次 {current_batch} 无有效文本，跳过")
            continue

        # 调用阿里云API（带重试机制）
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                # 调用阿里云百炼嵌入API
                response = client.embeddings.create(
                    model=MODEL_NAME,
                    input=valid_batch_texts
                )

                # 提取当前批次的嵌入向量（保持与有效文本顺序一致）
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                print(f"批次 {current_batch} 成功：{len(valid_batch_texts)} 条文本生成嵌入")
                time.sleep(RETRY_DELAY)  # 轻微延迟，避免频率限制
                break

            except Exception as e:
                retry_count += 1
                error_msg = str(e)[:100]  # 截取部分错误信息（避免过长）
                print(f"批次 {current_batch} 第{retry_count}次重试失败：{error_msg}...")
                if retry_count >= MAX_RETRIES:
                    raise Exception(f"批次 {current_batch} 重试{MAX_RETRIES}次仍失败，请检查API配置/网络/文本") from e

    # 验证嵌入数量与有效文本数量一致（避免遗漏）
    assert len(all_embeddings) == len([t for t in texts if t and t.strip() != ""]), \
        f"最终嵌入数量（{len(all_embeddings)}）与有效文本数量不匹配"

    print(f"\n所有批次处理完成！共生成 {len(all_embeddings)} 条嵌入向量")
    return all_embeddings


# -------------------------- 主函数（读取JSON→生成嵌入→保存.pth）--------------------------
def main():
    # 1. 配置文件路径（需用户根据实际情况修改）
    input_json_path = "gpt_prompts/s3dis-simple.json"  # 如：gpt_prompts/s3dis_10_gpt-3.5-turbo.json
    output_pth_path = "gpt_prompts/s3dis-aliyun.pth"  # 如：gpt_prompts/s3dis_10_gpt-3.5-turbo.pth

    # 2. 读取JSON文件（按原项目JSON结构：键为类别名，值为文本描述列表）
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"输入JSON文件不存在：{input_json_path}")

    with open(input_json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # 3. 整理所有文本（将JSON中所有类别下的文本合并为一个列表）
    texts = []
    for category, desc_list in json_data.items():
        if isinstance(desc_list, list):  # 确保是列表格式
            texts.extend(desc_list)
            print(f"读取类别「{category}」：{len(desc_list)} 条文本")

    print(f"\nJSON文件读取完成：共 {len(texts)} 条文本")

    # 4. 分批生成嵌入
    embeddings = batch_generate_embeddings(texts)

    # 5. 转换为PyTorch Tensor并保存为.pth文件
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    torch.save(embeddings_tensor, output_pth_path)

    print(f"\n嵌入向量已保存为 .pth 文件：{output_pth_path}")
    print(f"Tensor形状：{embeddings_tensor.shape}（条数 × 嵌入维度）")


# -------------------------- 执行入口 --------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n程序执行失败：{str(e)}")