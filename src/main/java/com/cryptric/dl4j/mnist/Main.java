package com.cryptric.dl4j.mnist;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.SamplingDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class Main {

    private static final Logger log = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) throws IOException {

        String dataSetPath = "data\\mnist.csv";
        double trainTestRatio = 0.9;
        int numSamples = 70 * 1000;

        int inputSize = 28 * 28;
        int numClasses = 10;

        int batchSize = 128;
        int numEpochs = 10;

        int seed = 3141;


        log.info("========== load dataset ==========");
        DataSet dataSet = null;
        try {
            dataSet = loadDataSet(dataSetPath, inputSize, numClasses).next(numSamples);
        } catch (InterruptedException e) {
            log.error("Could not load dataset");
            e.printStackTrace();
            System.exit(-1);
        }


        log.info("========== normalize dataset ==========");
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(dataSet);
        normalizer.transform(dataSet);


        log.info("========== split dataset ==========");
        SplitTestAndTrain dataSetSplit = dataSet.splitTestAndTrain(trainTestRatio);

        DataSet trainingData = dataSetSplit.getTrain();
        DataSet testingData = dataSetSplit.getTest();

        DataSetIterator trainingIterator = new SamplingDataSetIterator(trainingData, batchSize, trainingData.numExamples());
        DataSetIterator testingIterator = new SamplingDataSetIterator(testingData, batchSize, testingData.numExamples());


        log.info("========== configure model ==========");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaDelta())
                .list()
                .inputPreProcessor(0,new FeedForwardToCnnPreProcessor(28, 28, 1))
                .setInputType(InputType.feedForward(inputSize))
                .layer(new ConvolutionLayer.Builder(5, 5).nIn(1).stride(1, 1).nOut(20).activation(Activation.IDENTITY).build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())
                .layer(new ConvolutionLayer.Builder(5, 5).nIn(1).stride(1, 1).nOut(20).activation(Activation.IDENTITY).build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())
                .layer(new DenseLayer.Builder().nIn(50).nOut(50).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nIn(50).nOut(numClasses).build())
                .build();


        log.info("========== init model ==========");
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));


        log.info("========== train model ==========");
        model.fit(trainingIterator, numEpochs);


        log.info("========== evaluate model ==========");
        Evaluation evaluation = model.evaluate(testingIterator);
        log.info(evaluation.toString());

    }

    private static DataSetIterator loadDataSet(String path, int labelIndex, int numClasses) throws IOException, InterruptedException {
        RecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(new FileSplit(new File(path)));
        return new RecordReaderDataSetIterator(recordReader, 1, labelIndex, numClasses);
    }

}
