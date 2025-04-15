package ir.ppl.federatedlearning;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Main {

    public static void main(String[] args) throws IOException {
    	int numClients = 5;
        int numRounds = 50;
        int localEpochs = 3;
        int batchSize = 64;
        double learningRate = 0.1;

        MNISTDataset dataset = new MNISTDataset("data\\mnist_train.csv", "data\\mnist_test.csv");

        Server server = new Server(learningRate);
        List<Client> clients = new ArrayList<>();

       List<List<MNISTDataset.DataPoint>> clientData = dataset.splitData(numClients, true);

        for (int i = 0; i < numClients; i++) {
            clients.add(new Client(
                    server.getGlobalModel(),
                    clientData.get(i),
                    batchSize,
                    localEpochs,
                    learningRate
            ));
        }

        double bestAccuracy = 0;
        int noImprovementCount = 0;
        final int PATIENCE = 5;

        for (int round = 0; round < numRounds; round++) {
            System.out.println("\n=== Round " + (round + 1) + "/" + numRounds + " ===");

           List<Client> selectedClients = selectRandomClients(clients, 0.4);

            List<NeuralNetwork> clientModels = new ArrayList<>();
            for (Client client : selectedClients) {
                client.setModel(server.getGlobalModel().copy());
                client.trainLocal();
                clientModels.add(client.getModel());
            }

            server.federatedAveraging(clientModels);

            double testAccuracy = server.evaluate(dataset.getTestData());
            System.out.println("Test Accuracy: " + testAccuracy);

            // Early stopping
            if (testAccuracy > bestAccuracy) {
                bestAccuracy = testAccuracy;
                noImprovementCount = 0;
            } else {
                noImprovementCount++;
                if (noImprovementCount >= PATIENCE) {
                    System.out.println("Early stopping at round " + (round + 1));
                    break;
                }
            }
        }
    }

    private static List<Client> selectRandomClients(List<Client> clients, double fraction) {
        int numSelected = (int) (clients.size() * fraction);
        List<Client> selected = new ArrayList<>();
        List<Client> tempList = new ArrayList<>(clients);
        Collections.shuffle(tempList);

        return tempList.subList(0, numSelected);
    }
}
