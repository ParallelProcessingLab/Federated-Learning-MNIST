package ir.ppl.federatedlearning;

import java.util.List;

public class Server {

    private NeuralNetwork globalModel;
    private final double learningRate;

    public Server(double learningRate) {
        this.learningRate = learningRate;
        this.globalModel = new NeuralNetwork(learningRate);
    }

    public NeuralNetwork getGlobalModel() {
        return globalModel;
    }

    public void federatedAveraging2(List<NeuralNetwork> clientModels) {
        if (clientModels.isEmpty()) {
            return;
        }

        globalModel.averageWeights(clientModels);
    }

    public void federatedAveraging(List<NeuralNetwork> clientModels) {
        if (clientModels.isEmpty()) {
            return;
        }

        // Reset global model weights
        globalModel.resetWeightsToZero();

        // Sum all client models
        for (NeuralNetwork model : clientModels) {
            globalModel.addModelWeights(model);
        }

        // Average the weights
        globalModel.divideWeights(clientModels.size());
    }

    public double evaluate(List<MNISTDataset.DataPoint> testData) {
        int correct = 0;
        int[] classCorrect = new int[10];
        int[] classTotal = new int[10];

        for (MNISTDataset.DataPoint data : testData) {
            double[] output = globalModel.predict(data.pixels);
            int predicted = argmax(output);

            classTotal[data.label]++;
            if (predicted == data.label) {
                correct++;
                classCorrect[data.label]++;
            }
        }

        System.out.println("\nClass-wise Accuracy:");
        for (int i = 0; i < 10; i++) {
            if (classTotal[i] > 0) {
                double acc = (double) classCorrect[i] / classTotal[i];
                System.out.printf("Class %d: %.2f%% (%d/%d)%n",
                        i, acc * 100, classCorrect[i], classTotal[i]);
            }
        }

        return (double) correct / testData.size();
    }

    public void validateModel(List<MNISTDataset.DataPoint> validationData) {
        int correct = 0;
        for (MNISTDataset.DataPoint data : validationData) {
            double[] output = globalModel.predict(data.pixels);
            if (argmax(output) == data.label) {
                correct++;
            }
        }
        System.out.println("Validation Accuracy: " + (double) correct / validationData.size());
    }

    private int argmax(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
