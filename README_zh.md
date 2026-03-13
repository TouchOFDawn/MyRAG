# 混合智能体系统 (Hybrid Agent with GraphRAG)

[English](README.md) | [中文](README_zh.md)

---

## 概述

一个结合先进 RAG 能力与推测性工具执行的混合智能体系统。系统使用本地小模型进行智能路由和分支预测，通过并行推测执行工具调用实现低延迟响应。

## 核心特性

### 智能体系统
- **推测性工具执行**：本地小模型预测并预执行高延迟工具（网页抓取、天气 API），与主 LLM 推理并行运行
- **智能查询路由**：本地 Qwen2.5-0.5B 将查询路由到最优检索策略（向量/图谱/混合）
- **多工具集成**：RAG 检索、网页抓取和外部 API 之间的无缝协调

### RAG 能力
- **多源数据加载**：支持 PDF、文本文件和网页内容
- **知识图谱构建**：使用 LLM 自动提取实体和关系
- **混合检索**：在向量搜索和图遍历之间智能路由
- **双数据库架构**：ChromaDB 存储向量 + Neo4j 存储知识图谱

### 基础设施
- **多模型支持**：兼容 OpenAI、DeepSeek、DashScope、Ollama 和 Claude
- **灵活嵌入**：支持 HuggingFace、OpenAI 和本地嵌入模型
- **智能缓存**：增量索引避免重复处理
- **交互式对话**：实时问答，上下文感知生成

## 系统架构

```
                    ┌─────────────────────────────────┐
                    │      数据摄入管道                │
                    └─────────────────────────────────┘
                                  ↓
    数据源 → 分块 → 图谱提取 → 双重存储
                                  ↓
                    ┌─────────────────────────────────┐
                    │      智能体执行层                │
                    └─────────────────────────────────┘
                                  ↓
    用户查询 → 小模型预测器 (Qwen2.5-0.5B)
                    ↓                    ↓
              查询路由器            工具预测器
                    ↓                    ↓
          向量/图谱/混合          异步工具执行
               检索              (网页/天气/等)
                    ↓                    ↓
                    ┌─────────────────────────────────┐
                    │   主 LLM (并行)                  │
                    └─────────────────────────────────┘
                                  ↓
                            最终响应
```

**核心创新：**
- **推测执行**：主 LLM 处理查询时，小模型预测所需工具并并行执行，降低总延迟
- **双重路由**：分别使用小模型进行检索策略路由和工具预测

## 环境要求

- Python 3.13+
- Neo4j 数据库（本地或云端）
- 推荐使用 GPU 运行本地模型（可选）

## 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/TouchOFDawn/MyRAG.git
cd MyRAG
```

2. **安装依赖**
```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e .
```

3. **配置 Neo4j**
   - 从 [neo4j.com](https://neo4j.com/download/) 下载
   - 启动 Neo4j 并记录凭据
   - 默认地址：`bolt://localhost:7687`

4. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 填入你的 API 密钥和数据库凭据
```

## 配置说明

编辑 `.env` 文件：

```bash
# 选择 LLM 提供商（openai/deepseek/dashscope/ollama/claude）
# 设置对应的 API 密钥

# Neo4j 凭据
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=你的密码

# 在 src/graphrag/config.py 中选择模型
model_type = "claude"  # 或 "openai", "deepseek" 等
embed_type = "huggingface"  # 或 "openai", "ollama"
```

## 使用方法

1. **准备数据**
```bash
mkdir -p data/input
# 将 PDF/文本文件放入 data/input/ 目录
```

2. **运行系统**
```bash
python main.py
```

3. **交互模式**
```
User Query: 文档的主要主题是什么？
Model Answer: [基于你的数据的上下文感知响应]

User Query: exit
```

## 项目结构

```
MyRAG/
├── src/graphrag/
│   ├── config.py              # 配置管理
│   ├── data/                  # 数据加载模块
│   │   ├── loader.py
│   │   ├── directory.py
│   │   └── loaders/           # PDF、文本、网页加载器
│   ├── db/                    # 数据库管理器
│   │   ├── neo4j_manager.py
│   │   └── vector_store.py
│   ├── graph/
│   │   └── builder.py         # 知识图谱提取
│   ├── retrieval/             # 检索策略
│   │   ├── router.py          # 查询路由
│   │   ├── vector_retriever.py
│   │   └── graph_retriever.py
│   ├── generation/
│   │   └── generator.py       # 响应生成
│   ├── splitters/
│   │   └── markdown.py        # 语义分块
│   └── utils/
│       └── state.py           # 索引缓存
├── main.py                    # 入口文件
├── pyproject.toml             # 依赖配置
└── .env.example               # 环境变量模板
```

## 工作原理

1. **数据摄入**：加载文档并转换为 markdown
2. **语义分块**：将文本分割为有意义的片段
3. **图谱提取**：LLM 提取实体和关系
4. **双重索引**：
   - 文本块 → ChromaDB（向量搜索）
   - 实体/关系 → Neo4j（图遍历）
5. **查询处理**：
   - 路由器分析查询类型
   - 选择向量/图谱/混合检索
   - 检索相关上下文
6. **生成**：LLM 使用检索到的上下文生成答案

## 支持的模型

**LLM 提供商：**
- OpenAI (GPT-4, GPT-3.5)
- DeepSeek
- 阿里云 DashScope (通义千问)
- Ollama（本地模型）
- Anthropic Claude

**嵌入模型：**
- HuggingFace Transformers（本地）
- OpenAI Embeddings
- Ollama Embeddings

## 高级特性

- **多模态支持**：使用视觉模型处理图像
- **增量索引**：智能缓存避免重复处理未更改的数据
- **灵活路由**：本地小模型实现快速查询分类
- **工具集成**：天气 API 和网页抓取功能

## 故障排除

**Neo4j 连接错误：**
- 确保 Neo4j 正在运行：`neo4j status`
- 检查 `.env` 中的凭据

**API 密钥问题：**
- 验证 `.env` 中的 API 密钥设置正确
- 检查 API 提供商状态

**内存问题：**
- 减小 `main.py` 中的 `chunk_size`
- 使用更小的嵌入模型

## 贡献

欢迎贡献！请提交 issue 或 pull request。

## 许可证

MIT License
