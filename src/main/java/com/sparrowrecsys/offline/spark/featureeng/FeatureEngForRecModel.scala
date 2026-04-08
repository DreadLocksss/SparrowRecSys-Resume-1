package com.sparrowrecsys.offline.spark.featureeng

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.sql.functions.{format_number, _}
import org.apache.spark.sql.types.{DecimalType, FloatType, IntegerType, LongType}
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}
import redis.clients.jedis.Jedis

import scala.collection.immutable.ListMap
import scala.collection.{JavaConversions, mutable}
import scala.util.Try

object FeatureEngForRecModel {

  val NUMBER_PRECISION = 2
  val redisEndpoint = sys.props.get("sparrow.redis.host")
    .orElse(sys.env.get("SPARROW_REDIS_HOST"))
    .getOrElse("localhost")
  val redisPort = sys.props.get("sparrow.redis.port")
    .orElse(sys.env.get("SPARROW_REDIS_PORT"))
    .flatMap(port => Try(port.toInt).toOption)
    .getOrElse(6379)
  val redisTtlSeconds = sys.props.get("sparrow.redis.ttl.seconds")
    .orElse(sys.env.get("SPARROW_REDIS_TTL_SECONDS"))
    .flatMap(ttl => Try(ttl.toInt).toOption)
    .getOrElse(60 * 60 * 24 * 30)

  private def getAsString(sample: Row, columnName: String): String = {
    Option(sample.getAs[Any](columnName)).map(_.toString).getOrElse("")
  }

  /**
   * 把原始评分样本转成二分类训练标签，同时打印一些数据分布用于检查
   * @param ratingSamples 来自于 rating.csv 的原始评分样本
   * @return
   */
  def addSampleLabel(ratingSamples:DataFrame): DataFrame ={
    ratingSamples.show(10, truncate = false)
    ratingSamples.printSchema()
    val sampleCount = ratingSamples.count()
    ratingSamples.groupBy(col("rating")).count().orderBy(col("rating"))
      .withColumn("percentage", col("count")/sampleCount).show(100,truncate = false)

    ratingSamples.withColumn("label", when(col("rating") >= 3.5, 1).otherwise(0))
  }

  /**
   * 添加电影特征
   * @param movieSamples 电影元数据
   * @param ratingSamples 评分元数据
   * @return 电影特征
   */
  def addMovieFeatures(movieSamples:DataFrame, ratingSamples:DataFrame): DataFrame ={

    //add movie basic features
    val samplesWithMovies1 = ratingSamples.join(movieSamples, Seq("movieId"), "left")
    //add release year
    val extractReleaseYearUdf = udf({(title: String) => {
      if (null == title || title.trim.length < 6) {
        1990 // default value
      }
      else {
        val yearString = title.trim.substring(title.length - 5, title.length - 1)
        yearString.toInt
      }
    }})

    //add title
    val extractTitleUdf = udf({(title: String) => {title.trim.substring(0, title.trim.length - 6).trim}})

    val samplesWithMovies2 = samplesWithMovies1.withColumn("releaseYear", extractReleaseYearUdf(col("title")))
      .withColumn("title", extractTitleUdf(col("title")))
      .drop("title")  //title is useless currently

    //split genres
    val samplesWithMovies3 = samplesWithMovies2.withColumn("movieGenre1",split(col("genres"),"\\|").getItem(0))
      .withColumn("movieGenre2",split(col("genres"),"\\|").getItem(1))
      .withColumn("movieGenre3",split(col("genres"),"\\|").getItem(2))

    //add rating features
    val movieRatingFeatures = samplesWithMovies3.groupBy(col("movieId"))
      .agg(count(lit(1)).as("movieRatingCount"),
        format_number(avg(col("rating")), NUMBER_PRECISION).as("movieAvgRating"),
        stddev(col("rating")).as("movieRatingStddev"))
    .na.fill(0).withColumn("movieRatingStddev",format_number(col("movieRatingStddev"), NUMBER_PRECISION))


    //join movie rating features
    val samplesWithMovies4 = samplesWithMovies3.join(movieRatingFeatures, Seq("movieId"), "left")
    samplesWithMovies4.printSchema()
    samplesWithMovies4.show(10, truncate = false)

    samplesWithMovies4
  }

  /**
   * 统计每个用户看过的每一种类型的电影的数量
   */
  val extractGenres: UserDefinedFunction = udf { (genreArray: Seq[String]) => {
    val genreMap = mutable.Map[String, Int]()
    genreArray.foreach((element:String) => {
      val genres = element.split("\\|")
      genres.foreach((oneGenre:String) => {
        genreMap(oneGenre) = genreMap.getOrElse[Int](oneGenre, 0)  + 1
      })
    })

    // 按照第二个元素（统计的某个类型对应的电影数）的值进行降序排序，得到一个新的 ListMap
    val sortedGenres = ListMap(genreMap.toSeq.sortWith(_._2 > _._2):_*)
    sortedGenres.keys.toSeq
  }}

  /**
   * 添加用户特征，注意不引入未来信息，只取之前 100 条
   * @param ratingSamples 电影特征
   * @return 用户特征
   */
  def addUserFeatures(ratingSamples:DataFrame): DataFrame ={
    val samplesWithUserFeatures = ratingSamples
      .withColumn("userPositiveHistory", collect_list(when(col("label") === 1, col("movieId")).otherwise(lit(null)))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userPositiveHistory", reverse(col("userPositiveHistory")))
      .withColumn("userRatedMovie1",col("userPositiveHistory").getItem(0))
      .withColumn("userRatedMovie2",col("userPositiveHistory").getItem(1))
      .withColumn("userRatedMovie3",col("userPositiveHistory").getItem(2))
      .withColumn("userRatedMovie4",col("userPositiveHistory").getItem(3))
      .withColumn("userRatedMovie5",col("userPositiveHistory").getItem(4))
      .withColumn("userRatingCount", count(lit(1))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userAvgReleaseYear", avg(col("releaseYear"))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)).cast(IntegerType))
      .withColumn("userReleaseYearStddev", stddev(col("releaseYear"))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userAvgRating", format_number(avg(col("rating"))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)), NUMBER_PRECISION))
      .withColumn("userRatingStddev", stddev(col("rating"))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userGenres", extractGenres(collect_list(when(col("label") === 1, col("genres")).otherwise(lit(null)))
        .over(Window.partitionBy("userId")
          .orderBy(col("timestamp")).rowsBetween(-100, -1))))
      .na.fill(0)
      .withColumn("userRatingStddev",format_number(col("userRatingStddev"), NUMBER_PRECISION))
      .withColumn("userReleaseYearStddev",format_number(col("userReleaseYearStddev"), NUMBER_PRECISION))
      .withColumn("userGenre1",col("userGenres").getItem(0))
      .withColumn("userGenre2",col("userGenres").getItem(1))
      .withColumn("userGenre3",col("userGenres").getItem(2))
      .withColumn("userGenre4",col("userGenres").getItem(3))
      .withColumn("userGenre5",col("userGenres").getItem(4))
      .drop("genres", "userGenres", "userPositiveHistory")
      .filter(col("userRatingCount") > 1)

    samplesWithUserFeatures.printSchema()
    samplesWithUserFeatures.show(100, truncate = false)

    samplesWithUserFeatures
  }

  def extractAndSaveMovieFeaturesToRedis(samples:DataFrame): DataFrame = {
    val movieLatestSamples = samples.withColumn("movieRowNum", row_number()
      .over(Window.partitionBy("movieId")
        .orderBy(col("timestamp").desc)))
      .filter(col("movieRowNum") === 1)
      .select("movieId","releaseYear", "movieGenre1","movieGenre2","movieGenre3","movieRatingCount",
        "movieAvgRating", "movieRatingStddev")
      .na.fill("")

    movieLatestSamples.printSchema()
    movieLatestSamples.show(100, truncate = false)

    val movieFeaturePrefix = "mf:"

    val redisClient = new Jedis(redisEndpoint, redisPort)
    val sampleArray = movieLatestSamples.collect()
    println("total movie size:" + sampleArray.length)
    var insertedMovieNumber = 0
    val movieCount = sampleArray.length
    for (sample <- sampleArray){
      val movieKey = movieFeaturePrefix + getAsString(sample, "movieId")
      val valueMap = mutable.Map[String, String]()
      valueMap("movieGenre1") = getAsString(sample, "movieGenre1")
      valueMap("movieGenre2") = getAsString(sample, "movieGenre2")
      valueMap("movieGenre3") = getAsString(sample, "movieGenre3")
      valueMap("movieRatingCount") = getAsString(sample, "movieRatingCount")
      valueMap("releaseYear") = getAsString(sample, "releaseYear")
      valueMap("movieAvgRating") = getAsString(sample, "movieAvgRating")
      valueMap("movieRatingStddev") = getAsString(sample, "movieRatingStddev")

      // Use HMSET for Redis 3.x compatibility.
      redisClient.hmset(movieKey, JavaConversions.mapAsJavaMap(valueMap))
      redisClient.expire(movieKey, redisTtlSeconds)
      insertedMovieNumber += 1
      if (insertedMovieNumber % 100 ==0){
        println(insertedMovieNumber + "/" + movieCount + "...")
      }
    }

    redisClient.close()
    movieLatestSamples
  }

  def splitAndSaveTrainingTestSamples(samples:DataFrame, savePath:String)={
    //generate a smaller sample set for demo
    val smallSamples = samples.sample(0.1)

    //split training and test set by 8:2
    val Array(training, test) = smallSamples.randomSplit(Array(0.8, 0.2))

    val sampleResourcesPath = this.getClass.getResource(savePath)
    training.repartition(1).write.option("header", "true").mode(SaveMode.Overwrite)
      .csv(sampleResourcesPath+"/trainingSamples")
    test.repartition(1).write.option("header", "true").mode(SaveMode.Overwrite)
      .csv(sampleResourcesPath+"/testSamples")
  }

  def splitAndSaveTrainingTestSamplesByTimeStamp(samples:DataFrame, savePath:String)={
    //generate a smaller sample set for demo
    val smallSamples = samples.sample(0.1).withColumn("timestampLong", col("timestamp").cast(LongType))

    val quantile = smallSamples.stat.approxQuantile("timestampLong", Array(0.8), 0.05)

    // 就是取下标 0 的元素，也就是 80% 分位对应的时间戳
    // 这行等价写法是 quantile(0)
    val splitTimestamp = quantile.apply(0)

    val training = smallSamples.where(col("timestampLong") <= splitTimestamp).drop("timestampLong")
    val test = smallSamples.where(col("timestampLong") > splitTimestamp).drop("timestampLong")

    val sampleResourcesPath = this.getClass.getResource(savePath)
    training.repartition(1).write.option("header", "true").mode(SaveMode.Overwrite)
      .csv(sampleResourcesPath+"/trainingSamples")
    test.repartition(1).write.option("header", "true").mode(SaveMode.Overwrite)
      .csv(sampleResourcesPath+"/testSamples")
  }

  private def loadSamplesWithUserFeaturesFromTrainingAndTest(sparkSession: SparkSession,
                                                              savePath: String = "/webroot/sampledata"): DataFrame = {
    val sampleResourcesPath = this.getClass.getResource(savePath)
    require(sampleResourcesPath != null, s"sample path not found: $savePath")
    val basePath = sampleResourcesPath.getPath
    val trainingPath = basePath + "/trainingSamples"
    val testPath = basePath + "/testSamples"

    sparkSession.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(trainingPath, testPath)
      .withColumn("userRatingCount", col("userRatingCount").cast(LongType))
      .withColumn("userAvgReleaseYear", col("userAvgReleaseYear").cast(IntegerType))
      .withColumn("movieRatingCount", col("movieRatingCount").cast(LongType))
      .withColumn("releaseYear", col("releaseYear").cast(IntegerType))
  }

  def extractAndSaveUserFeaturesToRedisFromTrainingAndTest(sparkSession: SparkSession,
                                                            savePath: String = "/webroot/sampledata"): DataFrame = {
    val samplesWithUserFeatures = loadSamplesWithUserFeaturesFromTrainingAndTest(sparkSession, savePath)
    extractAndSaveUserFeaturesToRedis(samplesWithUserFeatures)
  }

  def extractAndSaveMovieFeaturesToRedisFromTrainingAndTest(sparkSession: SparkSession,
                                                             savePath: String = "/webroot/sampledata"): DataFrame = {
    val samplesWithUserFeatures = loadSamplesWithUserFeaturesFromTrainingAndTest(sparkSession, savePath)
    extractAndSaveMovieFeaturesToRedis(samplesWithUserFeatures)
  }


  def extractAndSaveUserFeaturesToRedis(samples:DataFrame): DataFrame = {
    val userLatestSamples = samples.withColumn("userRowNum", row_number()
      .over(Window.partitionBy("userId")
        .orderBy(col("timestamp").desc)))
      .filter(col("userRowNum") === 1)
      .select("userId","userRatedMovie1", "userRatedMovie2","userRatedMovie3","userRatedMovie4","userRatedMovie5",
        "userRatingCount", "userAvgReleaseYear", "userReleaseYearStddev", "userAvgRating", "userRatingStddev",
        "userGenre1", "userGenre2","userGenre3","userGenre4","userGenre5")
      .na.fill("")

    userLatestSamples.printSchema()
    userLatestSamples.show(100, truncate = false)

    val userFeaturePrefix = "uf:"

    val redisClient = new Jedis(redisEndpoint, redisPort)
    val sampleArray = userLatestSamples.collect()
    println("total user size:" + sampleArray.length)
    var insertedUserNumber = 0
    val userCount = sampleArray.length
    for (sample <- sampleArray){
      val userKey = userFeaturePrefix + getAsString(sample, "userId")
      val valueMap = mutable.Map[String, String]()
      valueMap("userRatedMovie1") = getAsString(sample, "userRatedMovie1")
      valueMap("userRatedMovie2") = getAsString(sample, "userRatedMovie2")
      valueMap("userRatedMovie3") = getAsString(sample, "userRatedMovie3")
      valueMap("userRatedMovie4") = getAsString(sample, "userRatedMovie4")
      valueMap("userRatedMovie5") = getAsString(sample, "userRatedMovie5")
      valueMap("userGenre1") = getAsString(sample, "userGenre1")
      valueMap("userGenre2") = getAsString(sample, "userGenre2")
      valueMap("userGenre3") = getAsString(sample, "userGenre3")
      valueMap("userGenre4") = getAsString(sample, "userGenre4")
      valueMap("userGenre5") = getAsString(sample, "userGenre5")
      valueMap("userRatingCount") = getAsString(sample, "userRatingCount")
      valueMap("userAvgReleaseYear") = getAsString(sample, "userAvgReleaseYear")
      valueMap("userReleaseYearStddev") = getAsString(sample, "userReleaseYearStddev")
      valueMap("userAvgRating") = getAsString(sample, "userAvgRating")
      valueMap("userRatingStddev") = getAsString(sample, "userRatingStddev")

      // Use HMSET for Redis 3.x compatibility.
      redisClient.hmset(userKey, JavaConversions.mapAsJavaMap(valueMap))
      redisClient.expire(userKey, redisTtlSeconds)
      insertedUserNumber += 1
      if (insertedUserNumber % 100 ==0){
        println(insertedUserNumber + "/" + userCount + "...")
      }
    }

    redisClient.close()
    userLatestSamples
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("featureEngineering")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()
    // val movieResourcesPath = this.getClass.getResource("/webroot/sampledata/movies.csv")
    // val movieSamples = spark.read.format("csv").option("header", "true").load(movieResourcesPath.getPath)

    // val ratingsResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
    // val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)

    // val ratingSamplesWithLabel = addSampleLabel(ratingSamples)
    // ratingSamplesWithLabel.show(10, truncate = false)

    // val samplesWithMovieFeatures = addMovieFeatures(movieSamples, ratingSamplesWithLabel)
    // val samplesWithUserFeatures = addUserFeatures(samplesWithMovieFeatures)


    // //save samples as csv format
    // splitAndSaveTrainingTestSamples(samplesWithUserFeatures, "/webroot/sampledata")

    // //save user features and item features to redis for online inference
    // extractAndSaveUserFeaturesToRedis(samplesWithUserFeatures)
    // extractAndSaveMovieFeaturesToRedis(samplesWithUserFeatures)
    extractAndSaveUserFeaturesToRedisFromTrainingAndTest(spark)
    extractAndSaveMovieFeaturesToRedisFromTrainingAndTest(spark)

    spark.close()
  }

}
