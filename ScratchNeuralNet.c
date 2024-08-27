#include <stdlib.h>
#include <stdio.h>
#include <math.h>


// Neural Network on XOR

double sigmoid(double x) {return 1/ (1 + exp(-x));}
double derivativeSigmoid(double x) {return x * (1-x);}
double init_weights() {return ((double)rand()) / ((double)RAND_MAX);}

// Fisher-Yates shuffle
void shuffle(int *array, size_t n){
    if (n > 1){
        size_t i;
        for (i = 0; i < n-1; i++){
            size_t j = i + rand() / (RAND_MAX / (n-i) +1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4

int main(void) {
    const double lr = 0.1f;

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];
    
    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];

    double training_inputs[numTrainingSets][numInputs] = {{0.0f,0.0f},
                                                          {1.0f,0.0f},
                                                          {0.0f,1.0f},
                                                          {1.0f,1.0f}};
    double training_outputs[numTrainingSets][numOutputs] = {{0.0f},
                                                            {1.0f},
                                                            {1.0f},
                                                            {0.0f}};

    // First we can intialize with random weights
    for (int i = 0; i < numInputs; i++){
        for (int j = 0; j < numHiddenNodes; j++){
            hiddenWeights[i][j] = init_weights();
        }
    }

    for (int i = 0; i < numHiddenNodes; i++){
        for (int j = 0; j < numOutputs; j++){
            outputWeights[i][j] = init_weights();
        }
    }

    for (int i = 0; i < numOutputs; i++){
            outputLayerBias[i] = init_weights();
    }

    int trainingSetOrder[] = {0,1,2,3};

    int numberOfEpochs = 10000;

    for (int epoch = 0; epoch < numberOfEpochs; epoch++){
        shuffle(trainingSetOrder, numTrainingSets);

        for (int x = 0; x < numTrainingSets; x++){
            int i = trainingSetOrder[x];

            for(int j = 0; j< numHiddenNodes; j++){
                double activation = hiddenLayerBias[j];

                for(int k = 0; k < numInputs; k++){
                    activation += training_inputs[i][k] * hiddenWeights[k][j];
                }

                hiddenLayer[j] = sigmoid(activation);
            }

            for(int j = 0; j< numOutputs; j++){
                double activation = outputLayerBias[j];

                for(int k = 0; k < numHiddenNodes; k++){
                    activation += hiddenLayer[k] * outputWeights[k][j];
                }

                outputLayer[j] = sigmoid(activation);
            }

            printf("Input: %g %g   Output: %g   Predicted Output: %g  \n",
            training_inputs[i][0], training_inputs[i][1], 
            training_outputs[i][0], outputLayer[0]);

            
            
            double deltaOutput[numOutputs];

            for (int j = 0; j < numOutputs; j++){
                double error = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] = error *derivativeSigmoid(outputLayer[j]);
            }

            double deltaHiddenLayer[numHiddenNodes];
            for (int j = 0; j < numHiddenNodes; j++){
                double error = 0.0f;
                for (int k = 0; k < numOutputs; k++){
                    error += deltaOutput[k] * outputWeights[j][k];
                }

                deltaHiddenLayer[j] = error * derivativeSigmoid(hiddenLayer[j]);
            }

            for(int j = 0; j < numOutputs; j++){
                outputLayerBias[j] += deltaOutput[j] * lr;
                for(int k = 0; k < numHiddenNodes; k++){
                    outputWeights[k][j]+= hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }

             for(int j = 0; j < numHiddenNodes; j++){
                hiddenLayerBias[j] += deltaHiddenLayer[j] * lr;
                for(int k = 0; k < numInputs; k++){
                    hiddenWeights[k][j]+= training_inputs[i][k] * deltaHiddenLayer[j] * lr;
                }
            }

        }
    }

            fputs ("Final Hidden Weights\n [", stdout);
            for (int j = 0; j < numHiddenNodes; j++){
                fputs ("[ ", stdout);
                for(int k = 0; k < numInputs; k++){
                printf("%f ", hiddenWeights[k][j]);
                }
                fputs ("] ", stdout);
            }

            fputs ( "]\n Final Hidden Biases\n[ ", stdout);
            for (int j = 0; j < numHiddenNodes; j++){
                printf("%f ", hiddenLayerBias[j]);
            }


            fputs ("] Final Output Weights\n [", stdout);
            for (int j = 0; j < numOutputs; j++){
                fputs ("[ ", stdout);
                for(int k = 0; k < numHiddenNodes; k++){
                printf("%f ", outputWeights[k][j]);
                }
                fputs ("] ", stdout);
            }



            fputs ( "]\n Final Output Biases\n[ ", stdout);
            for (int j = 0; j < numOutputs; j++){
                printf("%f ", outputLayerBias[j]);
            }

            

        

    

    fputs("] ", stdout);

    return 0;


    


}