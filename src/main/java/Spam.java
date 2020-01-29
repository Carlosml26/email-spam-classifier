import org.apache.avro.ipc.trace.TestSpanTraceFormation;
import org.apache.spark.SparkConf;
import org.apache.spark.StopMapOutputTracker;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.SparkSession;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;



public class Spam {

    public static void main (String[] args){

        // start spark session
        SparkConf conf = new SparkConf().setAppName("spam").setMaster("local");

        SparkSession spark = SparkSession
                .builder()
                .config(conf)
                .getOrCreate();

        // load data as spark-datasets


        StructType schema = DataTypes.createStructType(new StructField[]{
                DataTypes.createStructField("label", DataTypes.DoubleType, true),
                DataTypes.createStructField("sentence", DataTypes.StringType, true)
        });


        JavaRDD<Row> spamTrainingRDD = spark.read().textFile("src/main/resources/spam_training.txt").javaRDD().map(row -> RowFactory.create(1.0,row));
        spamTrainingRDD.cache();

        JavaRDD<Row> noSpamTrainingRDD = spark.read().textFile("src/main/resources/nospam_training.txt").javaRDD().map(row -> RowFactory.create(0.0,row));
        spamTrainingRDD.cache();

        JavaRDD<Row> spamTestingRDD = spark.read().textFile("src/main/resources/spam_testing.txt").javaRDD().map(row -> RowFactory.create(1.0,row));
        spamTrainingRDD.cache();

        JavaRDD<Row> noSpamTestingRDD = spark.read().textFile("src/main/resources/nospam_testing.txt").javaRDD().map(row -> RowFactory.create(0.0,row));
        spamTrainingRDD.cache();

        ArrayList<Row> listTrainingLabeled = new ArrayList<>(spamTrainingRDD.collect());
        listTrainingLabeled.addAll(noSpamTrainingRDD.collect());

        ArrayList<Row> listTestingLabeled = new ArrayList<>(spamTestingRDD.collect());
        listTestingLabeled.addAll(noSpamTestingRDD.collect());

        Dataset<Row> dataFrameTraining = spark.createDataFrame(listTrainingLabeled,schema);

        Dataset<Row> dataFrameTesting = spark.createDataFrame(listTestingLabeled,schema);



        // implement: convert datasets to either rdds or dataframes (your choice) and build your pipeline
        Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");
        Dataset<Row> wordsDataTraining = tokenizer.transform(dataFrameTraining);
        Dataset<Row> wordsDataTesting = tokenizer.transform(dataFrameTesting);


        StopWordsRemover filtered = new StopWordsRemover().setInputCol("words").setOutputCol("wordsFiltered");
        wordsDataTraining = filtered.transform(wordsDataTraining);
        wordsDataTesting = filtered.transform(wordsDataTesting);


        HashingTF hashingTF = new HashingTF().setInputCol("wordsFiltered").setOutputCol("rawFeatures");

        Dataset<Row> featurizedDataTraining = hashingTF.transform(wordsDataTraining);
        Dataset<Row> featurizedDataTesting = hashingTF.transform(wordsDataTesting);

        // alternatively, CountVectorizer can also be used to get term frequency vectors
        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
        IDFModel idfModel = idf.fit(featurizedDataTraining);

        Dataset<Row> rescaledDataTraining = idfModel.transform(featurizedDataTraining);
        Dataset<Row> rescaledDataTesting = idfModel.transform(featurizedDataTesting);


        // create the trainer and set its parameters
        NaiveBayes nb = new NaiveBayes();

        // train the model
        NaiveBayesModel model = nb.fit(rescaledDataTraining);
        Dataset<Row> predictions = model.transform(rescaledDataTesting);
        predictions.show();

        // compute accuracy on the test set
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test set accuracy = " + accuracy);


    }
}
