import java.awt.image.BufferedImage
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.{SparkConf, SparkContext}
import org.jblas.DoubleMatrix

/**
  * Created by ysp on 2016-12-30.
  */
object SC {
    def main(args: Array[String]) {
        val sc = new SparkContext(new SparkConf().setMaster("local").setAppName("SC"))
        val path = "/home/ysp/Art/*"
        val rdd = sc.wholeTextFiles(path)
        val files = rdd.map { case (fileName, content) => fileName.replace("file:", "") }
        def loadImageFromFile(path: String): BufferedImage = {
            import java.io.File
            import javax.imageio.ImageIO
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
        val pixels = files.map(f => extractPixels(f, 50, 50))
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

        var i = 0
        val projecteds = projected.rows.take(1332).map{ case ( x ) =>
            i += 1
            (i,x)
        }
        val ps = projecteds.map{ case (i,x) =>
          val j= i.toInt
          val y= x.toArray
          (j,y)
        }

        var j = 0
        val fs = files.map{ case ( add ) =>
            j += 1
            (j.toString+"|"+add)
        }
        val t = fs.map(line => line.split("\\|").take(2)).map(array=>
            (array(0),array(1))).collectAsMap()

        val itemf = ps.map{ case (i,x) =>
            if(i==158) return x.toArray
        }

        val itemFactor =projected.rows.take(1).map{case ( x ) =>
            val y= x.toArray
            (y)
        }

        val itemVector = new DoubleMatrix(itemFactor)

        def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double = {
            vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
        }

        val sims = ps.map{ case (id, factor) =>
          val factorVector = new DoubleMatrix(factor)
          val sim = cosineSimilarity(factorVector, itemVector)
          val add = t(id.toString)
          (id, sim ,add)
        }
        println(sims.take(1).mkString("\n"))

        val sortedSims = sims.sortBy(-_._2)
        println(sortedSims.slice(1,11)mkString("\n"))
    }
}

