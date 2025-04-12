using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using RedNeuronal_ProyectoFinal.Services;

namespace RedNeuronal_ProyectoFinal.Utils
{
    /// <summary>
    /// Herramienta para comparar el rendimiento entre la implementación secuencial y la paralela.
    /// </summary>
    public static class PerformanceTester
    {
        /// <summary>
        /// Ejecuta el entrenamiento de ambas redes con los mismos datos y muestra los tiempos.
        /// </summary>
        public static void TestNetworks(double[][] inputs, double[][] targets, int epochs)
        {
            // Versión secuencial
            var sequentialNN = new NeuralNetworkSequential(4, 4, 3);
            var watchSeq = Stopwatch.StartNew();
            sequentialNN.Train(inputs, targets, epochs);
            watchSeq.Stop();
            Console.WriteLine($"[Secuencial] Tiempo total: {watchSeq.ElapsedMilliseconds} ms");

            // Versión paralela
            var parallelNN = new NeuralNetworkParallel(4, 4, 3);
            var watchPar = Stopwatch.StartNew();
            parallelNN.Train(inputs, targets, epochs);
            watchPar.Stop();
            Console.WriteLine($"[Paralelo] Tiempo total: {watchPar.ElapsedMilliseconds} ms");
        }
    }
}
