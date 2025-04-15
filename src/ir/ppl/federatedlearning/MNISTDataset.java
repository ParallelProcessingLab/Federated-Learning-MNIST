package ir.ppl.federatedlearning;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Collections;

public class MNISTDataset {

    public static class DataPoint {

        public double[] pixels;
        public int label;

        public DataPoint(double[] pixels, int label) {
            this.pixels = pixels;
            this.label = label;
        }
    }

    private List<DataPoint> trainData;
    private List<DataPoint> testData;

    public MNISTDataset(String trainPath, String testPath) throws IOException {
        trainData = new ArrayList<>();
        testData = new ArrayList<>();
        loadData(trainPath, testPath);
    }

    private void loadData(String trainPath, String testPath) throws IOException {
        loadCSV(trainPath, trainData);

        loadCSV(testPath, testData);
    }

    private void loadCSV(String filePath, List<DataPoint> targetList) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;
        boolean firstLine = true;

        while ((line = reader.readLine()) != null) {
            if (firstLine) {
                firstLine = false;
                continue; // Skip header
            }

            String[] values = line.split(",");
            int label = Integer.parseInt(values[0]);
            double[] pixels = new double[values.length - 1];

            for (int i = 1; i < values.length; i++) {
                pixels[i - 1] = Double.parseDouble(values[i]) / 255.0; // Normalize to [0,1]
            }

            targetList.add(new DataPoint(pixels, label));
        }

        reader.close();
    }

    public List<List<DataPoint>> splitData(int numClients, boolean iid) {
        List<List<DataPoint>> clientData = new ArrayList<>();

        if (iid) {
            for (int i = 0; i < numClients; i++) {
                clientData.add(new ArrayList<>());
            }

            Collections.shuffle(trainData);
            int samplesPerClient = trainData.size() / numClients;

            for (int i = 0; i < numClients; i++) {
                int start = i * samplesPerClient;
                int end = (i == numClients - 1) ? trainData.size() : (i + 1) * samplesPerClient;
                clientData.set(i, new ArrayList<>(trainData.subList(start, end)));
            }
        } else {
            
        }

        return clientData;
    }

    public List<DataPoint> getTestData() {
        return testData;
    }

    public List<DataPoint> getTrainData() {
        return trainData;
    }
}
