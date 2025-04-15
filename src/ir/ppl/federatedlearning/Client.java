package ir.ppl.federatedlearning;

import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Client {

    private NeuralNetwork model;
    private final List<MNISTDataset.DataPoint> localData;
    private final int batchSize;
    private final int localEpochs;
    private final double learningRate;

    public Client(NeuralNetwork globalModel, List<MNISTDataset.DataPoint> localData,
            int batchSize, int localEpochs, double learningRate) {
        this.model = globalModel.copy();
        this.localData = localData;
        this.batchSize = batchSize;
        this.localEpochs = localEpochs;
        this.learningRate = learningRate;
    }

    public void setModel(NeuralNetwork model) {
        this.model = model;
    }

    public NeuralNetwork getModel() {
        return model;
    }

    public void trainLocal() {
        Random rand = new Random();

        for (int epoch = 0; epoch < localEpochs; epoch++) {
            Collections.shuffle(localData, rand);

            for (int i = 0; i < localData.size(); i += batchSize) {
                int end = Math.min(i + batchSize, localData.size());
                List<MNISTDataset.DataPoint> batch = localData.subList(i, end);

                for (MNISTDataset.DataPoint data : batch) {
                    model.train(data.pixels, data.label);
                }
            }
        }
    }
}
