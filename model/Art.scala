import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import java.awt.image.BufferedImage
import java.awt.image
import javax.imageio.ImageIO
import java.io.File
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import breeze.linalg.DenseMatrix
import org.jblas.DoubleMatrix
import breeze.linalg.csvwrite
import org.apache.spark.mllib.recommendation.Rating

/**
  * Created by ysp on 2016-12-30.
  */
object Art {
    def main(args: Array[String]) {
        val sc = new SparkContext(new SparkConf().setMaster("local").setAppName("Art"))
        val item = sc.textFile("/home/ysp/Arts.item")
        val t = item.map(line => line.split("\\|").take(2)).map(array=>
            (array(0).toInt,array(1))).collectAsMap()
        val fs = item.map(line => line.split("\\|").take(2))

        val file = fs.map(array=>array(1))
        println(file.count)
        file.first()
        t(1)

        def loadImageFromFile(path: String): BufferedImage = {
            import javax.imageio.ImageIO
            import java.io.File
            ImageIO.read(new File(path))
        }
        def processImage(image: BufferedImage, width: Int, height: Int):
        BufferedImage = {
            val bwImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
            val g = bwImage.getGraphics()
            g.drawImage(image, 0, 0, width, height, null)
            g.dispose()
            bwImage
        }
        def getPixelsFromImage(image: BufferedImage): Array[Double] = {
            val width = image.getWidth
            val height = image.getHeight
            val pixels = Array.ofDim[Double](width * height)
            image.getData.getPixels(0, 0, width, height, pixels)
        }
        def extractPixels(path: String, width: Int, height: Int):
        Array[Double] = {
            val raw = loadImageFromFile(path)
            val processed = processImage(raw, width, height)
            getPixelsFromImage(processed)
        }

        file.first()

        val pixels = file.map(f => extractPixels(f, 50, 50))
        val vectors = pixels.map(p => Vectors.dense(p))
        vectors.setName("image-vectors")
        vectors.cache
        val scaler = new StandardScaler(withMean = true, withStd = false).fit(vectors)
        val scaledVectors = vectors.map(v => scaler.transform(v))
        val matrix = new RowMatrix(scaledVectors)
        val K = 10
        val pc = matrix.computePrincipalComponents(K)
        val projected = matrix.multiply(pc)
        println(projected.numRows, projected.numCols)
        println(projected.rows.take(10).mkString("\n"))

        var i = -1
        val ps = projected.rows.take(1539).map{ case ( x ) =>
            i = i+1
            val y= x.toArray
            val address = t(i)
            (i, y, address)
        }

        val itemFactor =projected.rows.take(1).map{case ( x ) =>
            val y= x.toArray
            (y)
        }
        val itemVector = new DoubleMatrix(itemFactor)

        def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double = {
            vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
        }
        val sims = ps.map{ case (id, factor, add) =>
            val factorVector = new DoubleMatrix(factor)
            val sim = cosineSimilarity(factorVector, itemVector)
            (id, sim, add)
        }

        println(sims.take(1).mkString("\n"))
        val sortedSims = sims.sortBy(-_._2)
        println()
        println("推荐：")
        //println(sortedSims.slice(2,40)mkString("\n"))
        sortedSims.foreach(println)
        val st = sortedSims.map{case(id,sim,add)=>
            sim
        }
        st.foreach(println)
        //val sa = sortedSims.map { case (id, sim, add) =>
          //  add.split("/").take(4).toString
        //}
        //sa.foreach(println)
    }
}

