package ir.ppl.federatedlearning;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class NeuralNetwork {

    private final int inputSize = 784; // 28x28
    private final int hiddenSize = 128;
    private final int outputSize = 10;

    private double[][] W1;
    private double[] b1; 
    private double[][] W2;
    private double[] b2; 

    private final double learningRate;

    public NeuralNetwork(double learningRate) {
        this.learningRate = learningRate;
        initializeWeights();
    }

    private void initializeWeights() {
        Random rand = new Random();

        // Initialize W1 (input -> hidden)
        W1 = new double[hiddenSize][inputSize];
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                W1[i][j] = rand.nextGaussian() * 0.01;
            }
        }

        // Initialize b1 (hidden bias)
        b1 = new double[hiddenSize];

        // Initialize W2 (hidden -> output)
        W2 = new double[outputSize][hiddenSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W2[i][j] = rand.nextGaussian() * 0.01;
            }
        }

        // Initialize b2 (output bias)
        b2 = new double[outputSize];
    }

    public double[] predict(double[] input) {
        // Forward pass
        double[] hidden = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            double sum = b1[i];
            for (int j = 0; j < inputSize; j++) {
                sum += W1[i][j] * input[j];
            }
            hidden[i] = relu(sum);
        }

        double[] output = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            double sum = b2[i];
            for (int j = 0; j < hiddenSize; j++) {
                sum += W2[i][j] * hidden[j];
            }
            output[i] = softmax(sum, output, i);
        }

        return output;
    }

    public void train(double[] input, int label) {
        // Forward pass
        double[] hidden = new double[hiddenSize];
        double[] hiddenPreActivation = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            double sum = b1[i];
            for (int j = 0; j < inputSize; j++) {
                sum += W1[i][j] * input[j];
            }
            hiddenPreActivation[i] = sum;
            hidden[i] = relu(sum);
        }

        double[] output = new double[outputSize];
        double[] outputPreSoftmax = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            double sum = b2[i];
            for (int j = 0; j < hiddenSize; j++) {
                sum += W2[i][j] * hidden[j];
            }
            outputPreSoftmax[i] = sum;
        }

        // Softmax
        double softmaxSum = 0;
        for (double value : outputPreSoftmax) {
            softmaxSum += Math.exp(value);
        }
        for (int i = 0; i < outputSize; i++) {
            output[i] = Math.exp(outputPreSoftmax[i]) / softmaxSum;
        }

        // Backward pass
        double[] dOutput = output.clone();
        dOutput[label] -= 1;

        // Gradients for W2 and b2
        double[][] dW2 = new double[outputSize][hiddenSize];
        double[] db2 = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                dW2[i][j] = dOutput[i] * hidden[j];
            }
            db2[i] = dOutput[i];
        }

        // Gradient for hidden layer
        double[] dHidden = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            double sum = 0;
            for (int i = 0; i < outputSize; i++) {
                sum += W2[i][j] * dOutput[i];
            }
            dHidden[j] = sum * reluDerivative(hiddenPreActivation[j]);
        }

        // Gradients for W1 and b1
        double[][] dW1 = new double[hiddenSize][inputSize];
        double[] db1 = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                dW1[i][j] = dHidden[i] * input[j];
            }
            db1[i] = dHidden[i];
        }

        // L2 Regularization
        double lambda = 0.001;
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                dW2[i][j] += lambda * W2[i][j];
            }
        }
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                dW1[i][j] += lambda * W1[i][j];
            }
        }

        // Update weights with smaller learning rate
        double effectiveLearningRate = learningRate * 0.1; 
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W2[i][j] -= effectiveLearningRate * dW2[i][j];
            }
            b2[i] -= effectiveLearningRate * db2[i];
        }

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                W1[i][j] -= effectiveLearningRate * dW1[i][j];
            }
            b1[i] -= effectiveLearningRate * db1[i];
        }
    }

    public void resetWeightsToZero() {
        for (int i = 0; i < hiddenSize; i++) {
            Arrays.fill(W1[i], 0);
            b1[i] = 0;
        }
        for (int i = 0; i < outputSize; i++) {
            Arrays.fill(W2[i], 0);
            b2[i] = 0;
        }
    }

    public void addModelWeights(NeuralNetwork model) {
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                W1[i][j] += model.W1[i][j];
            }
            b1[i] += model.b1[i];
        }
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W2[i][j] += model.W2[i][j];
            }
            b2[i] += model.b2[i];
        }
    }

    public void divideWeights(int divisor) {
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                W1[i][j] /= divisor;
            }
            b1[i] /= divisor;
        }
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W2[i][j] /= divisor;
            }
            b2[i] /= divisor;
        }
    }

    public NeuralNetwork copy() {
        NeuralNetwork copy = new NeuralNetwork(this.learningRate);

        // Copy W1
        for (int i = 0; i < hiddenSize; i++) {
            System.arraycopy(this.W1[i], 0, copy.W1[i], 0, inputSize);
        }

        // Copy b1
        System.arraycopy(this.b1, 0, copy.b1, 0, hiddenSize);

        // Copy W2
        for (int i = 0; i < outputSize; i++) {
            System.arraycopy(this.W2[i], 0, copy.W2[i], 0, hiddenSize);
        }

        // Copy b2
        System.arraycopy(this.b2, 0, copy.b2, 0, outputSize);

        return copy;
    }

    public void averageWeights(List<NeuralNetwork> models) {
        if (models.isEmpty()) {
            return;
        }

        // Reset weights to zero
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                W1[i][j] = 0;
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            b1[i] = 0;
        }

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W2[i][j] = 0;
            }
        }

        for (int i = 0; i < outputSize; i++) {
            b2[i] = 0;
        }

        // Sum all weights
        for (NeuralNetwork model : models) {
            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < inputSize; j++) {
                    W1[i][j] += model.W1[i][j];
                }
            }

            for (int i = 0; i < hiddenSize; i++) {
                b1[i] += model.b1[i];
            }

            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    W2[i][j] += model.W2[i][j];
                }
            }

            for (int i = 0; i < outputSize; i++) {
                b2[i] += model.b2[i];
            }
        }

        // Divide by number of models to get average
        int numModels = models.size();
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                W1[i][j] /= numModels;
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            b1[i] /= numModels;
        }

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W2[i][j] /= numModels;
            }
        }

        for (int i = 0; i < outputSize; i++) {
            b2[i] /= numModels;
        }
    }

    private double relu(double x) {
        return Math.max(0, x);
    }

    private double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    private double softmax(double x, double[] output, int index) {
        double sum = 0;
        for (double value : output) {
            sum += Math.exp(value);
        }
        return Math.exp(x) / sum;
    }
}
