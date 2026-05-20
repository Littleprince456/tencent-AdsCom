# PCVRHyFormer 模型架构文档

## 目录

- [整体架构](#整体架构)
- [输入层](#输入层)
- [处理层](#处理层)
- [输出层](#输出层)
- [组件详细说明](#组件详细说明)
- [模型训练与推理](#模型训练与推理)

---

## 整体架构

PCVRHyFormer 是一个混合 Transformer 模型，用于点击后转化率预测。该模型通过以下关键组件协同工作：

```
输入 (ModelInput)
    |
    v
┌─────────────────────────────────────────────┐
│  1. NS Token 构建 (非序列特征)              │
│  - GroupNSTokenizer 或 RankMixerNSTokenizer │
└─────────────────────────────────────────────┘
    |
    v
┌─────────────────────────────────────────────┐
│  2. 序列特征嵌入 (Seq Tokens)               │
│  - 多域序列嵌入 + 时间桶                   │
└─────────────────────────────────────────────┘
    |
    v
┌─────────────────────────────────────────────┐
│  3. 多序列查询生成                         │
│  - MultiSeqQueryGenerator                  │
└─────────────────────────────────────────────┘
    |
    v
┌─────────────────────────────────────────────┐
│  4. HyFormer 块堆叠                        │
│  - MultiSeqHyFormerBlock (多次)            │
│    • 序列演化 (独立编码)                   │
│    • 查询解码 (交叉注意力)                 │
│    • Token 融合                            │
│    • RankMixer 交互增强                    │
└─────────────────────────────────────────────┘
    |
    v
┌─────────────────────────────────────────────┐
│  5. 输出投影与分类器                       │
│  - 多查询融合 + MLP                        │
└─────────────────────────────────────────────┘
    |
    v
输出: logits (B, action_num)
```

---

## 输入层

### ModelInput

输入数据通过 `ModelInput` 结构体组织，包含以下字段：

```python
class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor       # 用户整数特征 (B, user_int_dim)
    item_int_feats: torch.Tensor       # 物品整数特征 (B, item_int_dim)
    user_dense_feats: torch.Tensor     # 用户稠密特征 (B, user_dense_dim)
    item_dense_feats: torch.Tensor     # 物品稠密特征 (B, item_dense_dim)
    seq_data: dict                     # 序列数据 {domain: tensor (B, S, L)}
    seq_lens: dict                     # 序列长度 {domain: tensor (B)}
    seq_time_buckets: dict             # 时间桶 {domain: tensor (B, L)}
```

**字段说明:**
- `user_int_feats` / `item_int_feats`: 按特征规格拼接的离散特征
- `user_dense_feats` / `item_dense_feats`: 连续特征（可选）
- `seq_data`: 每个域的序列特征，形状为 (batch, num_features, seq_len)
- `seq_lens`: 每个样本的有效序列长度
- `seq_time_buckets`: 每个位置的时间间隔编码

---

## 处理层

### 1. NS Token 构建

NS（Non-Sequence，非序列）特征通过两种可选方式 token 化：

#### 方案 A: GroupNSTokenizer

**功能:**
- 按语义分组特征
- 每个组通过独立投影生成一个 NS Token

**实现流程:**
```
特征分组 → Embedding Lookup → 拼接 → 线性投影 → NS Token
```

**关键代码位置:** [model.py#L988-L1068](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L988-L1068)

**参数:**
- `feature_specs`: 特征规格列表 [(vocab_size, offset, length), ...]
- `groups`: 特征索引分组列表
- `emb_dim`: 嵌入维度
- `d_model`: 模型维度
- `emb_skip_threshold`: 跳过的高基数字典阈值

#### 方案 B: RankMixerNSTokenizer（默认）

**功能:**
- 所有特征 Embedding 拼接成一个长向量
- 均匀分割成指定数量的片段
- 每个片段独立投影为 NS Token

**优势:** 可以自由设置 NS Token 数量，与分组数量解耦

**实现流程:**
```
全部特征 Embedding → 拼接 → 填充至可分割 → 均匀分块 → 独立投影 → NS Tokens
```

**关键代码位置:** [model.py#L1070-L1189](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L1070-L1189)

**参数:**
- `num_ns_tokens`: 输出的 NS Token 数量

#### 稠密特征处理

如果提供了用户/物品稠密特征：
- 通过线性层投影到 `d_model`
- 使用 SiLU 激活
- 添加 LayerNorm 归一化
- 作为独立 NS Token 加入

### 2. 序列特征嵌入

**功能:** 将每个域的序列特征编码为序列 Tokens

**实现流程:**
```
序列数据 (B, num_fids, L)
    |
    v
逐个 fid Embedding Lookup
    |
    v
拼接所有 fid 嵌入 → (B, L, num_fids*emb_dim)
    |
    v
线性投影 + GELU + LayerNorm → (B, L, D)
    |
    v
+ 时间桶 Embedding (可选)
    |
    v
序列 Token (B, L, D)
```

**关键代码位置:** [model.py#L1544-L1574](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L1544-L1574)

**参数:**
- `seq_id_threshold`: ID 特征阈值（超过该值的特征会额外 Dropout）
- `emb_skip_threshold`: 跳过的高基数字典阈值
- `num_time_buckets`: 时间桶数量（0 表示不使用）

**特殊处理:**
- ID 特征在训练时会额外应用 Dropout（`dropout_rate * 2`）
- 跳过的特征填充零向量
- 时间桶使用 padding_idx=0（零输入产生零向量）

### 3. 多序列查询生成

**功能:** 为每个序列独立生成查询 Tokens

**实现流程:**
```
NS Tokens (B, num_ns, D)
    |
    v
展平 → (B, num_ns*D)

序列 Token_i (B, L_i, D)
    |
    v
Mean Pooling → (B, D)

GlobalInfo_i = Concat(NS_flat, Seq_pooled_i)
    |
    v
LayerNorm
    |
    v
通过 num_queries 个独立 FFN 生成查询
    |
    v
Q Tokens_i (B, Nq, D)
```

**关键代码位置:** [model.py#L415-L495](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L415-L495)

**参数:**
- `num_queries`: 每个序列的查询数量

### 4. HyFormer 块堆叠

**功能:** 通过多个块迭代处理 Q Tokens、NS Tokens 和序列 Tokens

**块内流程:**

```
┌─────────────────────────────────────────────────┐
│  1. 序列演化 (独立)                              │
│  - 每个序列通过序列编码器编码                    │
│  - 可选: RoPE 位置编码                          │
└─────────────────────────────────────────────────┘
                    |
                    v
┌─────────────────────────────────────────────────┐
│  2. 查询解码 (独立)                              │
│  - Q Tokens_i 通过交叉注意力关注编码后的序列    │
└─────────────────────────────────────────────────┘
                    |
                    v
┌─────────────────────────────────────────────────┐
│  3. Token 融合                                  │
│  - 所有解码后的 Q Tokens + NS Tokens 拼接       │
│  - 形状: (B, Nq*S + Nns, D)                     │
└─────────────────────────────────────────────────┘
                    |
                    v
┌─────────────────────────────────────────────────┐
│  4. Query Boosting (RankMixer)                  │
│  - Token 混合 (参数自由重连)                    │
│  - 共享 FFN (每个 Token 独立)                   │
│  - 残差连接 + Post-LN                           │
└─────────────────────────────────────────────────┘
                    |
                    v
┌─────────────────────────────────────────────────┐
│  5. 分割返回                                    │
│  - 拆分回 per-sequence Q Tokens + NS Tokens     │
└─────────────────────────────────────────────────┘
```

**关键代码位置:** [model.py#L850-L981](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L850-L981)

**序列编码器选项:**

1. **SwiGLUEncoder (无注意力)**
   - 前馈网络 + 残差连接
   - 高效，适合短序列

2. **TransformerEncoder (自注意力)**
   - 标准 Transformer Encoder
   - 支持 RoPE 位置编码

3. **LongerEncoder (自适应长度)**
   - 如果序列长度 > top_k: 交叉注意力，取最后 top_k 个位置作为 Q
   - 如果序列长度 ≤ top_k: 自注意力
   - 支持因果掩码

**RankMixer 模式:**
- `full`: 完整模式（Token 混合 + FFN）
- `ffn_only`: 仅 FFN
- `none`: 恒等映射

### 位置编码: RoPE (Rotary Position Embedding)

**功能:** 通过旋转操作注入位置信息

**实现:**
- 预计算 cos/sin 值并缓存
- 对 Q 和 K 独立应用旋转
- 不修改 V

**关键代码位置:** [model.py#L26-L93](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L26-L93)

---

## 输出层

### 输出投影与分类器

**流程:**

```
处理后的 Q Tokens 列表 [ (B, Nq, D) for each sequence ]
    |
    v
拼接 → (B, Nq*S, D)
    |
    v
展平 → (B, Nq*S*D)
    |
    v
输出投影 MLP → (B, D)
    |
    v
分类器 MLP → (B, action_num)
```

**输出投影:**
- 线性层: `(Nq*S*D) → D`
- LayerNorm 归一化

**分类器:**
- 线性层: `D → D`
- LayerNorm
- SiLU 激活
- Dropout
- 线性层: `D → action_num`

---

## 组件详细说明

### 基础组件

#### 1. SwiGLU

**功能:** SwiGLU 激活函数的前馈网络实现

**架构:**
```
x → Linear → 分为 x1, x2 → x1 * SiLU(x2) → Linear → 输出
```

**关键代码位置:** [model.py#L100-L114](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L100-L114)

#### 2. RoPEMultiHeadAttention

**功能:** 支持 RoPE 的多头注意力

**特点:**
- 手动实现 Q/K/V 投影
- 支持 Q 侧单独的 RoPE（用于 LongerEncoder 交叉注意力）
- 使用 SDPA 高效计算
- 带门控输出

**关键代码位置:** [model.py#L117-L241](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L117-L241)

#### 3. CrossAttention

**功能:** 交叉注意力模块，Q 来自全局 Tokens，K/V 来自序列 Tokens

**特点:**
- RoPE 仅应用于 K/V 侧
- Pre-LN 结构

**关键代码位置:** [model.py#L244-L312](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L244-L312)

#### 4. RankMixerBlock

**功能:** Query Boosting，通过 Token 混合增强表达能力

**Token 混合流程:**
```
Q (B, T, D)
    |
    v
Split → (B, T, T, d_sub)    d_sub = D/T
    |
    v
Transpose (swap token & subspace axes)
    |
    v
Flatten → (B, T, D)
```

**约束:**
- `full` 模式下 `d_model` 必须被 `T = Nq*S + Nns` 整除

**关键代码位置:** [model.py#L315-L412](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L315-L412)

### 序列编码器

#### 1. SwiGLUEncoder

**架构:**
```
x → LN → SwiGLU → Dropout → +x → 输出
```

**关键代码位置:** [model.py#L502-L542](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L502-L542)

#### 2. TransformerEncoder

**架构:**
```
x → LN1 → Self-Attention (w/ RoPE) → +x
  → LN2 → FFN → Dropout → +x → 输出
```

**关键代码位置:** [model.py#L544-L615](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L544-L615)

#### 3. LongerEncoder

**架构 (L > top_k):**
```
序列 x → 取最后 top_k 位置 → Q
       → 全部位置 → K/V
       → Cross Attention (Q, K, V) + residual
       → FFN → 输出 (B, top_k, D)
```

**架构 (L ≤ top_k):**
```
序列 x → Self-Attention (可选因果掩码) + residual
       → FFN → 输出 (B, L, D)
```

**关键代码位置:** [model.py#L616-L809](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L616-L809)

#### 4. create_sequence_encoder (工厂函数)

**功能:** 根据 `encoder_type` 创建对应编码器

**关键代码位置:** [model.py#L811-L843](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L811-L843)

---

## 模型训练与推理

### 前向传播 (forward)

**流程:**
1. 构建 NS Tokens
2. 嵌入各序列域
3. 生成 Q Tokens
4. 运行块堆叠（训练时应用 Dropout）
5. 分类器输出 logits

**关键代码位置:** [model.py#L1634-L1675](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L1634-L1675)

### 推理 (predict)

**流程:**
- 复用 `forward` 逻辑
- `apply_dropout=False`
- 返回 `(logits, embeddings)`

**关键代码位置:** [model.py#L1677-L1714](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L1677-L1714)

### 参数优化器分组

模型支持将参数分为两组优化：

1. **稀疏参数 (Sparse Parameters)**
   - 所有 Embedding 表权重
   - 使用 Adagrad 优化器

2. **稠密参数 (Dense Parameters)**
   - 其他所有参数
   - 使用 AdamW 优化器

**关键方法:**
- `get_sparse_params()`: [model.py#L1531-L1537](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L1531-L1537)
- `get_dense_params()`: [model.py#L1539-L1542](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L1539-L1542)

### 高基数字典重新初始化

支持只重新初始化高基数字典 Embedding，保留低基数字典和时间特征权重。

**关键方法:** `reinit_high_cardinality_params()`

**关键代码位置:** [model.py#L1470-L1529](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L1470-L1529)

### 初始化策略

所有 Embedding 表使用 Xavier 初始化，padding 索引（0）初始化为零向量。

**关键方法:** `_init_params()`

**关键代码位置:** [model.py#L1454-L1468](file:///Users/bytedance/repo/tencent-AdsCom/baseline/model.py#L1454-L1468)

---

## 超参数总结

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 64 | 模型维度 |
| `emb_dim` | 64 | 嵌入维度 |
| `num_queries` | 1 | 每个序列查询数量 |
| `num_hyformer_blocks` | 2 | HyFormer 块数量 |
| `num_heads` | 4 | 注意力头数 |
| `seq_encoder_type` | 'transformer' | 序列编码器类型 |
| `hidden_mult` | 4 | FFN 扩展倍数 |
| `dropout_rate` | 0.01 | Dropout 率 |
| `seq_top_k` | 50 | LongerEncoder top_k |
| `seq_causal` | False | 是否使用因果掩码 |
| `action_num` | 1 | 输出动作数 |
| `num_time_buckets` | 65 | 时间桶数量 |
| `rank_mixer_mode` | 'full' | RankMixer 模式 |
| `use_rope` | False | 是否使用 RoPE |
| `emb_skip_threshold` | 0 | 跳过高基数字典阈值 |
| `seq_id_threshold` | 10000 | ID 特征阈值 |
| `ns_tokenizer_type` | 'rankmixer' | NS Tokenizer 类型 |

---

## 数据流形状变化示例

假设:
- Batch size = 32
- 序列域数 S = 4
- 每个序列查询数 Nq = 2
- 模型维度 D = 128
- NS Token 数 Nns = 8

| 阶段 | 形状 |
|------|------|
| 输入 user_int_feats | (32, user_int_dim) |
| 输入 seq_data[domain] | (32, num_fids, L) |
| 用户 NS Tokens | (32, num_user_ns, 128) |
| NS Tokens (总) | (32, 8, 128) |
| 序列 Token[domain] | (32, L, 128) |
| Q Tokens 列表元素 | (32, 2, 128) |
| 融合后的 Token | (32, 2*4+8=16, 128) |
| 最终所有 Q | (32, 8, 128) |
| 输出投影后 | (32, 128) |
| 分类器输出 logits | (32, action_num) |
