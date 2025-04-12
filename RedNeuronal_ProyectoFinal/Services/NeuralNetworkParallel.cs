using RedNeuronal_ProyectoFinal.Model;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RedNeuronal_ProyectoFinal.Services
{
   
        /// <summary>
        /// Implementación paralela de la red neuronal.
        /// Se utilizan Parallel.For y sincronización (lock) en la retropropagación para evitar condiciones de carrera.
        /// Arquitectura: (4, 4, 3)
        /// </summary>
        public class NeuralNetworkParallel
        {
            private Layer[] Layers;
            private double LearningRate;

            public NeuralNetworkParallel(int inputSize, int hiddenSize, int outputSize, double learningRate = 0.1)
            {
                Layers = new Layer[]
                {
                new Layer(hiddenSize, inputSize),
                new Layer(outputSize, hiddenSize)
                };
                LearningRate = learningRate;
            }

            /// <summary>
            /// Propagación hacia adelante.
            /// </summary>
            public double[] Forward(double[] inputs)
            {
                foreach (var layer in Layers)
                    inputs = layer.ComputeOutputs(inputs);
                return inputs;
            }

            /// <summary>
            /// Entrena la red utilizando procesamiento paralelo por muestra.
            /// Se usa lock en la retropropagación para evitar conflictos.
            /// </summary>
            public void Train(double[][] inputs, double[][] targets, int epochs)
            {
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    double[] errors = new double[inputs.Length];

                    Parallel.For(0, inputs.Length, i =>
                    {
                        double[] output = Forward(inputs[i]);
                        double sampleError = 0;
                        for (int j = 0; j < output.Length; j++)
                            sampleError += Math.Pow(targets[i][j] - output[j], 2);
                        errors[i] = sampleError;

                        // Retropropagación sincronizada.
                        lock (this)
                        {
                            Backpropagation(inputs[i], targets[i]);
                        }
                    });

                    double totalError = errors.Sum();
                    Console.WriteLine($"[Paralelo] Epoch {epoch + 1}, Error: {totalError / inputs.Length}");
                }
            }

            /// <summary>
            /// Retropropagación paralela: se ajustan los pesos utilizando Parallel.For en cada capa.
            /// </summary>
            private void Backpropagation(double[] inputs, double[] targets)
            {
                for (int i = Layers.Length - 1; i >= 0; i--)
                {
                    var layer = Layers[i];
                    double[] prevOutputs = (i == 0) ? inputs : Layers[i - 1].ComputeOutputs(inputs);

                    Parallel.For(0, layer.Neurons.Length, j =>
                    {
                        var neuron = layer.Neurons[j];
                        if (i == Layers.Length - 1)
                            neuron.Delta = (targets[j] - neuron.Output) * Neuron.SigmoidDerivative(neuron.Output);
                        else
                        {
                            double sum = 0;
                            foreach (var nextNeuron in Layers[i + 1].Neurons)
                                sum += nextNeuron.Weights[j] * nextNeuron.Delta;
                            neuron.Delta = sum * Neuron.SigmoidDerivative(neuron.Output);
                        }

                        for (int k = 0; k < neuron.Weights.Length; k++)
                            neuron.Weights[k] += LearningRate * neuron.Delta * prevOutputs[k];
                        neuron.Bias += LearningRate * neuron.Delta;
                    });
                }
            }
        }
    }

