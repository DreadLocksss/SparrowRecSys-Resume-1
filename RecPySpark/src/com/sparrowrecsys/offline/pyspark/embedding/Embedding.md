# Embedding.py 中获得 item / user Embedding 的三种方法

本文根据 [Embedding.py](/c:/Users/32116/Desktop/ut/algo/resume/1/SparrowRecSys-Resume-1/RecPySpark/src/com/sparrowrecsys/offline/pyspark/embedding/Embedding.py) 的实现，归纳出 3 种获得 `item embedding` / `user embedding` 的方法。

## 1. 基于用户观影序列的 Item2Vec item embedding

对应函数：

- `processItemSequence`
- `trainItem2vec`

核心思路：

把每个用户按时间排序后的观影序列当作一句“句子”，把电影 `movieId` 当作“词”，再用 `Word2Vec` 训练电影向量。因此，经常在同一用户行为序列中共同出现、上下文相似的电影，会得到相近的 embedding。

实现流程：

1. 读取评分数据 `ratings.csv`。
2. 过滤掉评分小于 `3.5` 的行为，只保留偏正向反馈。
3. 按 `userId` 聚合，并按 `timestamp` 对每个用户的电影观看记录排序。
4. 得到形如 `['1', '296', '32', ...]` 的用户观影序列。
5. 用这些序列训练 `Word2Vec`：
   - 向量维度：`embLength`
   - 窗口大小：`5`
   - 迭代次数：`10`
6. 训练完成后，直接从 `model.getVectors()` 中取出每个 `movieId` 的 embedding。

输出结果：

- 得到的是 `item embedding`
- 文件中输出到 `item2vecEmb.csv`

特点：

- 利用了用户真实行为序列。
- 更适合学习“同一上下文中一起出现”的电影相似性。

## 2. 基于转移图 + 随机游走的 graph item embedding

对应函数：

- `generate_pair`
- `generateTransitionMatrix`
- `oneRandomWalk`
- `randomWalk`
- `graphEmb`

核心思路：

先把用户观影序列转成“物品转移图”，再在图上做随机游走，生成大量新的 item 序列，最后仍然使用 `Word2Vec` 训练 embedding。这个方法本质上是把图结构信息转成序列信息，再学习 item 向量。

实现流程：

1. 对每条用户观影序列生成相邻电影对，例如：
   - 输入：`['858', '50', '593', '457']`
   - 输出：`[('858', '50'), ('50', '593'), ('593', '457')]`
2. 统计相邻电影对出现次数，构建转移计数矩阵。
3. 对每个起点电影做归一化，得到转移概率：
   - `P(next_item | current_item)`
4. 同时计算起始节点分布 `itemDistribution`。
5. 根据上述概率分布进行多次随机游走：
   - 样本条数：`20000`
   - 每条长度：`10`
6. 得到新的“游走序列”后，转成 RDD。
7. 对这些随机游走序列再次调用 `trainItem2vec`，训练出图 embedding。

输出结果：

- 得到的是另一套 `item embedding`
- 文件中输出到 `itemGraphEmb.csv`

特点：

- 不仅使用了原始共现关系，还显式利用了 item 之间的转移结构。
- 更适合建模“从一个物品跳到另一个物品”的图关系。

## 3. 基于 item embedding 聚合得到 user embedding

对应函数：

- `generateUserEmb`

核心思路：

用户向量不是直接训练出来的，而是先拿到电影向量，再把用户看过电影的 embedding 按用户聚合，得到用户 embedding。也就是说，这是一种“由 item embedding 反推 user embedding”的方式。

实现流程：

1. 读取评分数据。
2. 从训练好的 item2vec 模型中取出每个 `movieId` 的 embedding。
3. 将评分数据与电影向量表按 `movieId` 做 `join`。
4. 对每个 `userId` 聚合其对应的电影 embedding。
5. 聚合方式是逐维相加：
   - `user_emb = item_emb_1 + item_emb_2 + ... + item_emb_n`
6. 将得到的用户向量写入文件。

输出结果：

- 得到的是 `user embedding`
- 文件中输出到 `userEmb.csv`

特点：

- 实现简单，依赖已有 item embedding。
- 当前代码里使用的是“向量求和”，不是平均池化，也不是带权重的注意力聚合。

## 总结

`Embedding.py` 中一共体现了 3 条获得 embedding 的路径：

1. `Item2Vec`：直接用用户观影序列训练 `item embedding`。
2. `Graph Embedding`：先构建 item 转移图并随机游走，再训练 `item embedding`。
3. `User Embedding Aggregation`：基于已有 `item embedding`，按用户历史行为逐维求和，生成 `user embedding`。

如果从结果类型来看：

- 前两种方法产出的是 `item embedding`
- 第三种方法产出的是 `user embedding`
