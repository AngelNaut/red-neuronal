using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RedNeuronal_ProyectoFinal.Data
{
    /// <summary>
 /// Representa una muestra del dataset Iris.
 /// Incluye las 4 características y la especie.
 /// </summary>
    public class IrisSample
    {
        public double SepalLength { get; set; }
        public double SepalWidth { get; set; }
        public double PetalLength { get; set; }
        public double PetalWidth { get; set; }
        public string Species { get; set; }

        /// <summary>
        /// Retorna las características en un arreglo de double.
        /// </summary>
        public double[] GetFeatures() => new double[] { SepalLength, SepalWidth, PetalLength, PetalWidth };
    }
}
