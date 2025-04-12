using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using RedNeuronal_ProyectoFinal.Model;

namespace RedNeuronal_ProyectoFinal.Services
{
    /// <summary>
    /// Implementación secuencial de una red neuronal.
    /// Arquitectura: (input, hidden, output) – en este caso (4, 4, 3) para el dataset Iris.
    /// </summary>
    public class NeuralNetworkSequential
    {
        private Layer[] Layers;
        private double LearningRate;

        public NeuralNetworkSequential(int inputSize, int hiddenSize, int outputSize, double learningRate = 0.1)
        {
            Layers = new Layer[]
            {
                new Layer(hiddenSize, inputSize),  // Capa oculta
                new Layer(outputSize, hiddenSize)   // Capa de salida
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
        /// Entrena la red iterando sobre todas las muestras durante un número de épocas.
        /// </summary>
        public void Train(double[][] inputs, double[][] targets, int epochs)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalError = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    double[] output = Forward(inputs[i]);
                    // Error cuadrático medio
                    totalError += output.Zip(targets[i], (o, t) => Math.Pow(t - o, 2)).Sum();
                    Backpropagation(inputs[i], targets[i]);
                }
                Console.WriteLine($"[Secuencial] Epoch {epoch + 1}, Error: {totalError / inputs.Length}");
            }
        }

        /// <summary>
        /// Retropropagación: cálculo de deltas y actualización de pesos para cada capa.
        /// </summary>
        private void Backpropagation(double[] inputs, double[] targets)
        {
            for (int i = Layers.Length - 1; i >= 0; i--)
            {
                var layer = Layers[i];
                double[] prevOutputs = (i == 0) ? inputs : Layers[i - 1].ComputeOutputs(inputs);

                for (int j = 0; j < layer.Neurons.Length; j++)
                {
                    var neuron = layer.Neurons[j];
                    if (i == Layers.Length - 1)
                        neuron.Delta = (targets[j] - neuron.Output) * Neuron.SigmoidDerivative(neuron.Output);
                    else
                        neuron.Delta = Layers[i + 1].Neurons.Sum(n => n.Weights[j] * n.Delta) * Neuron.SigmoidDerivative(neuron.Output);

                    // Actualización de pesos y bias.
                    for (int k = 0; k < neuron.Weights.Length; k++)
                        neuron.Weights[k] += LearningRate * neuron.Delta * prevOutputs[k];
                    neuron.Bias += LearningRate * neuron.Delta;
                }
            }
        }
    }
}
