# LLM 模型配置节点 (`LLMLoaderNode`)

**原名**：`ModelScopeAPILoaderNode` → 魔搭 API 配置  
**新名**：`LLMLoaderNode` → LLM 模型配置  
（相关实现位于 `qwen_clip_node.py`）

---

## 输入接口

| 参数      | 类型     | 说明                             |
|-----------|----------|----------------------------------|
| `platform`| 下拉选择 | 选择 API 平台：`ModelScope`、`OnethingAI`、`Custom` |
| `api_key` | STRING   | API 密钥                         |
| `model`   | STRING   | 模型名称（可选，留空则使用平台默认模型） |
| `base_url`| STRING   | API 地址（可选，留空则使用平台默认地址） |

> **默认行为**  
> - 下拉选择 `platform` 的默认值为 `ModelScope`。  
> - 当 `model` 或 `base_url` 留空时，节点会根据所选平台自动填入下表中的默认值。

---

## 平台与默认配置

`platform` 选项由代码 `["ModelScope", "OnethingAI", "Custom"]` 定义（见 `qwen_clip_node.py:186-188`）。

| 平台       | Base URL                                           | 默认模型                         | 需要 `job_type` |
|------------|----------------------------------------------------|----------------------------------|:---------------:|
| ModelScope | `https://api-inference.modelscope.cn/v1`           | `Qwen/Qwen3-VL-8B-Instruct`      | ❌              |
| OnethingAI | `https://api-model.onethingai.com/v2/generation`   | `doubao-seed-1-6-flash-250615`   | ✅              |
| Custom     | 用户自定义                                         | 用户自定义                       | ❌              |

> **自定义平台** 要求用户显式提供 `base_url` 与 `model`，不会应用任何默认值。

---

## OnethingAI 平台 API 兼容

1. **自动添加 `job_type`**  
   发送请求时，自动在 JSON 体中注入 `"job_type": "chat/completions"` 字段。

2. **路径拼接规则**  
   - OnethingAI **直接使用**配置的 `base_url`，**不会**在其后追加 `/chat/completions`。  
   - 其他平台（包括 ModelScope 和 Custom）会在 `base_url` 后**自动拼接** `/chat/completions`。

---

## 使用示例

### OnethingAI 平台

```text
[LLM 模型配置] → api_config → [反推提示词 (LLM)]
  platform: OnethingAI
  api_key: your-api-key
  model: doubao-seed-1-6-flash-250615  (或留空使用默认)
