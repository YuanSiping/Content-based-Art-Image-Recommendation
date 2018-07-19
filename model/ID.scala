package LFW

import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by ysp on 16-10-30.
  */
object ID {
    def main(args: Array[String]) {
      val sc = new SparkContext(new SparkConf().setMaster("local").setAppName("ID"))
      val path = "/home/ysp/Art/*"
      val rdd = sc.wholeTextFiles(path)
      val files = rdd.map { case (fileName, content) => fileName.replace("file:", "") }
      var j = 0
      val fs = files.map{ case ( add ) =>
        j += 1
        (j.toString+"|"+add)
      }
      fs.foreach(println)
  }
}

