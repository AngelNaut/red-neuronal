using RedNeuronal_ProyectoFinal.Utils;
using static RedNeuronal_ProyectoFinal.Data.IrisDataProccessor;

namespace RedNeuronal_ProyectoFinal
{
    public class Program
    {
        static void Main()
        {
            // Ruta al archivo CSV (asegúrate de tener iris.csv en la ruta indicada)
            string filePath = "iris.csv";

            // 1. Cargar y normalizar datos
            var data = IrisDataProcessor.LoadData(filePath);
            IrisDataProcessor.NormalizeData(data);

            // 2. Dividir datos (80% entrenamiento, 20% test)
            var (trainingData, testData) = IrisDataProcessor.SplitData(data, 0.8);

            // 3. Preparar los arreglos de entrada y target para el entrenamiento
            // Convertimos cada muestra en un arreglo de características y obtenemos el vector one-hot para la especie.
            double[][] trainInputs = trainingData.Select(s => s.GetFeatures()).ToArray();
            double[][] trainTargets = trainingData.Select(s => IrisDataProcessor.GetOneHotVector(s.Species)).ToArray();

            // 4. Definir número de épocas
            int epochs = 2000;

            // 5. Ejecutar pruebas de rendimiento (sec vs. paralelo)
            PerformanceTester.TestNetworks(trainInputs, trainTargets, epochs);

            // 6. (Opcional) Se podría evaluar en los datos de test para medir la precisión del modelo.
            Console.WriteLine("Entrenamiento completado. Presiona cualquier tecla para finalizar.");
            Console.ReadKey();
        }
    }
}
