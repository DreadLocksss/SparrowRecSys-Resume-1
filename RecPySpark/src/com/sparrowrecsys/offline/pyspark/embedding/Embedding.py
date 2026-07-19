import os
from pyspark import SparkConf  # 导入 SparkConf，用于配置 Spark 应用参数
from pyspark.sql import SparkSession  # 导入 SparkSession，作为 Spark SQL/DataFrame 入口
from pyspark.sql.functions import *  # 导入 SQL 函数（udf/collect_list/array_join 等）
from pyspark.sql.types import *  # 导入 SQL 类型（StringType/ArrayType/StructType 等）
from pyspark.ml.feature import BucketedRandomProjectionLSH  # 导入 LSH 近似近邻模型
from pyspark.mllib.feature import Word2Vec  # 导入 mllib 版本 Word2Vec，用于 item2vec 训练
from pyspark.ml.linalg import Vectors  # 导入向量工具，构造 dense 向量
import random
from collections import defaultdict
import numpy as np
from pyspark.sql import functions as F


class UdfFunction:
    @staticmethod

    # 定义按时间排序电影序列的方法
    def sortF(movie_list, timestamp_list):
        """
        sort by time and return the corresponding movie sequence
        eg:
            input: movie_list:[1,2,3]
                   timestamp_list:[1112486027,1212546032,1012486033]
            return [3,1,2]
        """
        pairs = []
        for m, t in zip(movie_list, timestamp_list):
            pairs.append((m, t))
        # sort by time
        pairs = sorted(pairs, key=lambda x: x[1])
        return [x[0] for x in pairs]


# 从原始评分数据生成用户观影序列
def processItemSequence(spark, rawSampleDataPath):
    # 读取带表头的 rating data 为 DataFrame。
    ratingSamples = spark.read.format("csv").option("header", "true").load(rawSampleDataPath)

    # 调试用：查看前 5 行和打印 schema
    # ratingSamples.show(5)  
    # ratingSamples.printSchema()

    sortUdf = udf(UdfFunction.sortF, ArrayType(StringType()))

    # 把同一个用户的 movieId 全部收集成一个列表
    # 把同一个用户的 timestamp 也全部收集成一个列表
    # 把这两个列表传给 sortUdf，按时间排序，得到该用户按观看顺序排列的电影序列
    # 把聚合结果命名为 movieIds，并把列表转成空格分隔的字符串，命名为 movieIdStr
    # 不加括号时，另一种写法是用反斜杠
    userSeq = (
        ratingSamples
        .where(F.col("rating") >= 3.5)  # 过滤评分小于 3.5 的行为，仅保留偏正反馈
        .groupBy("userId")  # 按用户分组
        .agg(sortUdf(F.collect_list("movieId"), F.collect_list("timestamp")).alias('movieIds'))
        .withColumn("movieIdStr", array_join(F.col("movieIds"), " "))
    )

    # 调试用：展示用户序列
    # userSeq.select("userId", "movieIdStr").show(10, truncate = False)  

    # 仅保留 movieIdStr 列，转成 RDD 后 map 切分字符串为列表
    # 转成 RDD 后，每条记录是一个 Row 对象，x[0] 取出 movieIdStr 列的字符串，再 split 成列表
    # 因为只 select 了一列，索引从 0 开始，唯一那列就是 0 号位
    return userSeq.select('movieIdStr').rdd.map(lambda x: x[0].split(' '))


# 使用 LSH 对电影 embedding 做近似近邻查询演示
def embeddingLSH(spark, movieEmbMap):
    # 初始化列表，保存 (movieId, denseVector)
    movieEmbSeq = []
    # 组装成 Spark 向量并保存
    for key, embedding_list in movieEmbMap.items():
        embedding_list = [np.float64(embedding) for embedding in embedding_list]
        movieEmbSeq.append((key, Vectors.dense(embedding_list)))
    # 创建 DataFrame 并命名列
    movieEmbDF = spark.createDataFrame(movieEmbSeq).toDF("movieId", "emb")
    # 初始化 LSH：输入 emb，输出桶 ID，桶长度 0.1，哈希表数量 3
    bucketProjectionLSH = BucketedRandomProjectionLSH(inputCol="emb", 
                                                      outputCol="bucketId", 
                                                      bucketLength=0.1,
                                                      numHashTables=3)
    # 在 movie embedding 上拟合 LSH 模型
    bucketModel = bucketProjectionLSH.fit(movieEmbDF)
    # 对每个 embedding 计算其桶分配结果
    embBucketResult = bucketModel.transform(movieEmbDF)

    # 调试用：展示桶映射结果
    print("movieId, emb, bucketId schema:")
    embBucketResult.printSchema()
    print("movieId, emb, bucketId data result:")
    embBucketResult.show(10, truncate=False)
    print("Approximately searching for 5 nearest neighbors of the sample embedding:")
    sampleEmb = Vectors.dense(0.795, 0.583, 1.120, 0.850, 0.174, -0.839, -0.0633, 0.249, 0.673, -0.237)
    bucketModel.approxNearestNeighbors(movieEmbDF, sampleEmb, 5).show(truncate=False)


def trainItem2vec(spark, samples, embLength, embOutputPath, saveToRedis, redisKeyPrefix):
      # 设置 Word2Vec 维度/窗口/迭代次数
    word2vec = (
        Word2Vec()
        .setVectorSize(embLength)
        .setWindowSize(5)
        .setNumIterations(10)
    )

    model = word2vec.fit(samples)

    # 以下是调试用的相似度查询示例，训练好模型后可以查看某个电影的相似电影列表
    # 查找电影 "158" 的前 20 个相似项
    synonyms = model.findSynonyms("158", 20)
    for synonym, cosineSimilarity in synonyms:
        print(synonym, cosineSimilarity)

    # 保存文件
    embOutputDir = os.path.dirname(embOutputPath)
    if not os.path.exists(embOutputDir):
        os.makedirs(embOutputDir)
    # 以写模式打开 embedding 输出文件。
    with open(embOutputPath, 'w') as f:
        for movie_id in model.getVectors():
            vectors = " ".join([str(emb) for emb in model.getVectors()[movie_id]])
            f.write(movie_id + ":" + vectors + "\n")  # 按 "movieId:vec..." 格式写入一行。

    # 调用 LSH 演示函数，查看向量近邻效果
    embeddingLSH(spark, model.getVectors())
    return model


def generate_pair(x):  # 将一条序列转换为相邻二元组序列。
    # eg:  # 注释：下面是输入输出示例。
    # watch sequence:['858', '50', '593', '457']  # 输入示例序列。
    # return:[['858', '50'],['50', '593'],['593', '457']]  # 输出示例相邻对。
    pairSeq = []  # 初始化相邻对列表。
    previousItem = ''  # 初始化前一个元素为空字符串。
    for item in x:  # 依次遍历序列中的每个 item。
        if not previousItem:  # 如果当前是第一个元素（尚无 previousItem）。
            previousItem = item  # 将当前元素记录为 previousItem。
        else:  # 如果已经有 previousItem。
            pairSeq.append((previousItem, item))  # 追加 (前一个, 当前) 的相邻对。
            previousItem = item  # 更新 previousItem 为当前元素。
    return pairSeq  # 返回整条序列的相邻对列表。

# 统计转移次数并构建转移概率矩阵
def generateTransitionMatrix(samples):
    pairSamples = samples.flatMap(lambda x: generate_pair(x))  # 对每条序列做相邻拆分并打平。
    pairCountMap = pairSamples.countByValue()  # 统计每个相邻对出现次数。
    pairTotalCount = 0  # 初始化相邻对总次数计数器。
    transitionCountMatrix = defaultdict(dict)  # 初始化转移计数矩阵：key1 -> {key2: cnt}。
    itemCountMap = defaultdict(int)  # 初始化每个起点 item 的出边总计数。
    for key, cnt in pairCountMap.items():  # 遍历每个相邻对及其次数。
        key1, key2 = key  # 拆分相邻对为起点与终点。
        transitionCountMatrix[key1][key2] = cnt  # 写入 key1 到 key2 的转移计数。
        itemCountMap[key1] += cnt  # 累加 key1 的总出边计数。
        pairTotalCount += cnt  # 累加全局相邻对总次数。
    transitionMatrix = defaultdict(dict)  # 初始化转移概率矩阵。
    itemDistribution = defaultdict(dict)  # 初始化首节点采样分布。
    for key1, transitionMap in transitionCountMatrix.items():  # 遍历每个起点的转移 map。
        for key2, cnt in transitionMap.items():  # 遍历每个终点及其计数。
            transitionMatrix[key1][key2] = transitionCountMatrix[key1][key2] / itemCountMap[key1]  # 归一化得到条件概率 P(key2|key1)。
    for itemid, cnt in itemCountMap.items():  # 遍历每个起点 item 的总计数。
        itemDistribution[itemid] = cnt / pairTotalCount  # 归一化得到起始 item 的采样概率。
    return transitionMatrix, itemDistribution  # 返回转移概率矩阵和起点分布。


def oneRandomWalk(transitionMatrix, itemDistribution, sampleLength):  # 执行一次随机游走，生成一条样本序列。
    sample = []  # 初始化单次游走输出序列。
    # pick the first element  # 注释：先按 itemDistribution 采样起始节点。
    randomDouble = random.random()  # 生成 [0,1) 的随机数用于抽样。
    firstItem = ""  # 初始化首个元素为空。
    accumulateProb = 0.0  # 初始化累计概率。
    for item, prob in itemDistribution.items():  # 遍历起始分布。
        accumulateProb += prob  # 累加概率质量。
        if accumulateProb >= randomDouble:  # 累计概率超过随机阈值时命中当前 item。
            firstItem = item  # 设置当前 item 为首元素。
            break  # 命中后结束循环。
    sample.append(firstItem)  # 将首元素加入样本序列。
    curElement = firstItem  # 当前游走节点初始化为首元素。
    i = 1  # 初始化已生成长度计数为 1。
    while i < sampleLength:  # 当序列长度未达到目标长度时继续游走。
        if (curElement not in itemDistribution) or (curElement not in transitionMatrix):  # 如果当前节点无有效分布或无转移边。
            break  # 无法继续游走则提前终止。
        probDistribution = transitionMatrix[curElement]  # 取当前节点的下一跳概率分布。
        randomDouble = random.random()  # 生成随机数用于下一跳抽样。
        accumulateProb = 0.0  # 重置累计概率。
        for item, prob in probDistribution.items():  # 遍历候选下一跳节点分布。
            accumulateProb += prob  # 累加候选概率。
            if accumulateProb >= randomDouble:  # 命中抽样阈值。
                curElement = item  # 将当前节点更新为命中的下一跳节点。
                break  # 结束本轮下一跳抽样。
        sample.append(curElement)  # 将当前节点追加到序列。
        i += 1  # 增加序列长度计数。
    return sample  # 返回一次随机游走得到的序列。


def randomWalk(transitionMatrix, itemDistribution, sampleCount, sampleLength):  # 多次执行随机游走以生成训练样本。
    samples = []  # 初始化所有游走样本列表。
    for i in range(sampleCount):  # 按 sampleCount 指定次数重复采样。
        samples.append(oneRandomWalk(transitionMatrix, itemDistribution, sampleLength))  # 每次生成一条游走序列并追加。
    return samples  # 返回所有随机游走序列。

# 基于转移图随机游走训练图 embedding
def graphEmb(samples, spark, embLength, embOutputFilename, saveToRedis, redisKeyPrefix):
    # 从原序列生成转移概率矩阵与起点分布
    transitionMatrix, itemDistribution = generateTransitionMatrix(samples)

    # 设置随机游走样本条数和每条样本长度
    sampleCount = 20000
    sampleLength = 10

    # 生成随机游走样本序列
    newSamples = randomWalk(transitionMatrix, itemDistribution, sampleCount, sampleLength)
    # 将 Python list 并行化为 Spark RDD
    rddSamples = spark.sparkContext.parallelize(newSamples)

    # 在游走序列上训练并输出图 embedding
    trainItem2vec(spark, rddSamples, embLength, embOutputFilename, saveToRedis, redisKeyPrefix)

# 由电影向量聚合得到用户向量
def generateUserEmb(spark, rawSampleDataPath, model, embLength, embOutputPath, saveToRedis, redisKeyPrefix):
    ratingSamples = spark.read.format("csv").option("header", "true").load(rawSampleDataPath)
    
    # 初始化列表，保存 (movieId, embedding list)
    Vectors_list = []
    # 遍历已训练模型中的电影向量。
    for key, value in model.getVectors().items():
        Vectors_list.append((key, list(value)))
    fields = [
        StructField('movieId', StringType(), False),
        StructField('emb', ArrayType(FloatType()), False)
    ]
    schema = StructType(fields)
    # 创建电影向量 DataFrame
    Vectors_df = spark.createDataFrame(Vectors_list, schema=schema)

    # 按 movieId 内连接评分数据与电影向量
    ratingSamples = ratingSamples.join(Vectors_df, on='movieId', how='inner')
    # 开始构建用户向量聚合计算链
    result = (
        ratingSamples.select('userId', 'emb') # 只留 userId 和 emb 列
        .rdd.map(lambda x: (x[0], x[1]))
        .reduceByKey(lambda a, b: [a[i] + b[i] for i in range(len(a))])  # 按 userId 聚合，对向量逐维相加
        .collect()  # 将聚合结果拉回 Driver 端，便于本地写文件
    )

    # 保存 UserEmbedding 到文件
    with open(embOutputPath, 'w') as f:
        for row in result:
            vectors = " ".join([str(emb) for emb in row[1]])
            f.write(row[0] + ":" + vectors + "\n")


if __name__ == '__main__':
    # 初始化 Spark 配置：应用名 ctrModel，本地模式 local
    conf = SparkConf().setAppName('ctrModel').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # 数据集读取路径
    spark_base = "file:///C:/Users/32116/Desktop/ut/algo/resume/1/SparrowRecSys-Resume-1/src/main/resources"
    rawSampleDataPath = spark_base + "/webroot/sampledata/ratings.csv"

    # 保存路径
    local_base = r"C:\Users\32116\Desktop\ut\algo\resume\1\SparrowRecSys-Resume-1\src\main\resources"
    model_item2vec_path = local_base + r"\webroot\modeldata2\item2vecEmb.csv"
    model_graph_path = local_base + r"\webroot\modeldata2\itemGraphEmb.csv"
    model_user_path = local_base + r"\webroot\modeldata2\userEmb.csv"

    # 特征维度
    embLength = 10

    # 生成训练用用户观影序列样本
    samples = processItemSequence(spark, rawSampleDataPath)
    # 训练 item2vec 电影 embedding，并输出到文件
    model = trainItem2vec(spark, samples, embLength,
                          embOutputPath=model_item2vec_path, 
                          saveToRedis=False, redisKeyPrefix="i2vEmb")
    # 基于转移图随机游走训练图 embedding
    graphEmb(samples, spark, embLength, 
             embOutputFilename=model_graph_path,
             saveToRedis=False, redisKeyPrefix="graphEmb")
    # 用户向量生成：由电影向量聚合得到用户向量
    generateUserEmb(spark, rawSampleDataPath, model, embLength,
                    embOutputPath=model_user_path, 
                    saveToRedis=False, redisKeyPrefix="uEmb")
