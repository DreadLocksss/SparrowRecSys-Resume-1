from pyspark import SparkConf  # 导入 Spark 配置类
from pyspark.ml import Pipeline  # 导入 ML 流水线工具
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, QuantileDiscretizer, MinMaxScaler  # 导入特征转换器
from pyspark.ml.linalg import VectorUDT, Vectors  # 导入向量类型与向量构造器
from pyspark.sql import SparkSession  # 导入 SparkSession 入口
from pyspark.sql.functions import *  # 导入常用 SQL 函数
from pyspark.sql.types import *  # 导入 SQL 数据类型
from pyspark.sql import functions as F  # 将 SQL 函数起别名为 F


def oneHotEncoderExample(movieSamples):  # 定义电影 ID 的 one-hot 编码示例
    samplesWithIdNumber = movieSamples.withColumn("movieIdNumber", F.col("movieId").cast(IntegerType()))  # 将 movieId 转成整数
    encoder = OneHotEncoderEstimator(inputCols=["movieIdNumber"], outputCols=['movieIdVector'], dropLast=False)  # 构建 one-hot 编码器
    oneHotEncoderSamples = encoder.fit(samplesWithIdNumber).transform(samplesWithIdNumber)  # 拟合编码器并转换数据
    oneHotEncoderSamples.printSchema()  # 打印转换后表结构
    oneHotEncoderSamples.show(10)  # 展示前 10 行


def array2vec(genreIndexes, indexSize):  # 将类型索引列表转为稀疏向量
    genreIndexes.sort()  # 为满足稀疏向量要求先排序索引
    fill_list = [1.0 for _ in range(len(genreIndexes))]  # 构造非零值列表
    return Vectors.sparse(indexSize, genreIndexes, fill_list)  # 返回稀疏 multi-hot 向量


def multiHotEncoderExample(movieSamples):  # 定义电影类型 multi-hot 编码示例
    samplesWithGenre = movieSamples.select("movieId", "title", explode(  # 拆分类型字符串并按类型展开成多行
        split(F.col("genres"), "\\|").cast(ArrayType(StringType()))).alias('genre'))  # 每行保留一个 genre
    genreIndexer = StringIndexer(inputCol="genre", outputCol="genreIndex")  # 创建字符串到索引的转换器
    StringIndexerModel = genreIndexer.fit(samplesWithGenre)  # 在类型数据上拟合索引器
    genreIndexSamples = StringIndexerModel.transform(samplesWithGenre).withColumn("genreIndexInt",  # 应用索引器并新增整型索引列
                                                                                  F.col("genreIndex").cast(IntegerType()))  # 将双精度索引转为整数
    indexSize = genreIndexSamples.agg(max(F.col("genreIndexInt"))).head()[0] + 1  # 由最大索引计算向量维度
    processedSamples = genreIndexSamples.groupBy('movieId').agg(  # 按电影聚合类型索引
        F.collect_list('genreIndexInt').alias('genreIndexes')).withColumn("indexSize", F.lit(indexSize))  # 收集所有类型索引并补充维度
    finalSample = processedSamples.withColumn("vector",  # 新增 multi-hot 向量列
                                              udf(array2vec, VectorUDT())(F.col("genreIndexes"), F.col("indexSize")))  # 用 UDF 构造稀疏向量
    finalSample.printSchema()  # 打印输出表结构
    finalSample.show(10)  # 展示前 10 行


def ratingFeatures(ratingSamples):  # 从评分数据构建数值特征
    ratingSamples.printSchema()  # 打印原始评分表结构
    ratingSamples.show()  # 展示评分样例数据
    # 计算电影平均评分与评分次数
    movieFeatures = ratingSamples.groupBy('movieId').agg(F.count(F.lit(1)).alias('ratingCount'),  # 按电影聚合并统计评分次数
                                                         F.avg("rating").alias("avgRating"),  # 计算平均评分
                                                         F.variance('rating').alias('ratingVar')) \
        .withColumn('avgRatingVec', udf(lambda x: Vectors.dense(x), VectorUDT())('avgRating'))  # 将平均评分包装为稠密向量
    movieFeatures.show(10)  # 展示前 10 行工程化特征
    # 分桶处理
    ratingCountDiscretizer = QuantileDiscretizer(numBuckets=100, inputCol="ratingCount", outputCol="ratingCountBucket")  # 对评分次数做分位数分桶
    # 归一化处理
    ratingScaler = MinMaxScaler(inputCol="avgRatingVec", outputCol="scaleAvgRating")  # 将平均评分缩放到 0 到 1
    pipelineStage = [ratingCountDiscretizer, ratingScaler]  # 定义特征流水线阶段
    featurePipeline = Pipeline(stages=pipelineStage)  # 创建流水线对象
    movieProcessedFeatures = featurePipeline.fit(movieFeatures).transform(movieFeatures)  # 拟合并转换特征
    movieProcessedFeatures.show(10)  # 展示处理后的特征


if __name__ == '__main__':  # 本地执行入口
    conf = SparkConf().setAppName('featureEngineering').setMaster('local')  # 配置 Spark 应用名和本地模式
    spark = SparkSession.builder.config(conf=conf).getOrCreate()  # 创建 Spark 会话
    file_path = 'file:///Users/zhewang/Workspace/SparrowRecSys/src/main/resources'  # 基础资源路径
    movieResourcesPath = file_path + "/webroot/sampledata/movies.csv"  # 生成电影数据路径
    movieSamples = spark.read.format('csv').option('header', 'true').load(movieResourcesPath)  # 读取电影样本数据
    print("Raw Movie Samples:")  # 打印原始电影样本标题
    movieSamples.show(10)  # 展示前 10 条电影数据
    movieSamples.printSchema()  # 打印电影数据结构
    print("OneHotEncoder Example:")  # 打印 one-hot 示例标题
    oneHotEncoderExample(movieSamples)  # 运行 one-hot 编码示例
    print("MultiHotEncoder Example:")  # 打印 multi-hot 示例标题
    multiHotEncoderExample(movieSamples)  # 运行 multi-hot 编码示例
    print("Numerical features Example:")  # 打印数值特征示例标题
    ratingsResourcesPath = file_path + "/webroot/sampledata/ratings.csv"  # 生成评分数据路径
    ratingSamples = spark.read.format('csv').option('header', 'true').load(ratingsResourcesPath)  # 读取评分样本数据
    ratingFeatures(ratingSamples)  # 运行数值特征工程
