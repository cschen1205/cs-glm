using System;
using System.Collections.Generic;
using System.IO;

namespace GlmSharp.FT
{
    class Program
    {
        static void Main(string[] args)
        {
            TestContraception();
        }

        public static Dictionary<string, List<double>> GetData()
        {
            string[] headers = null;
            Dictionary<string, List<double>> data = new Dictionary<string, List<double>>();
            using (StreamReader reader = new StreamReader("contraception.csv"))
            {
                string line;
                bool firstLine = true;
                while ((line = reader.ReadLine()) != null)
                {
                    if (firstLine)
                    {
                        firstLine = false;
                        headers = line.Split(new char[] { ',' });
                        for (int i = 0; i < headers.Length; ++i)
                        {
                            headers[i] = headers[i].Replace("\"", "");
                            data[headers[i]] = new List<double>();
                        }
                        continue;
                    }

                    string[] content = line.Split(new char[] { ',' });
                    for (int i = 0; i < content.Length; ++i)
                    {
                        content[i] = content[i].Replace("\"", "");
                        content[i] = content[i].Replace("+", "");
                        string header = headers[i];
                        double value;
                        if (content[i] == "Y")
                        {
                            value = 1;
                        }
                        else if (content[i] == "N")
                        {
                            value = 0;
                        }
                        else
                        {
                            double.TryParse(content[i], out value);
                        }
                        data[header].Add(value);

                    }

                }
            }

            return data;
        }
        
        public static void TestContraception()
        {
            Console.WriteLine("Get Data");
            Dictionary<string, List<double>> data = GetData();
            int n = 8;
            int m = data[""].Count;
            double[,] A = new double[m, n];
            double[] b = new double[m];

            for (int i = 0; i < m; ++i)
            {
                A[i, 0] = 1;
                A[i, 1] = data["age"][i];
                A[i, 2] = System.Math.Pow(data["age"][i], 2);
                A[i, 3] = data["urban"][i];

                // livch = 0 is not included as it is the base
                A[i, 4] = data["livch"][i] == 1 ? 1 : 0;
                A[i, 5] = data["livch"][i] == 2 ? 1 : 0;
                A[i, 6] = data["livch"][i] == 3 ? 1 : 0;

                b[i] = data["use"][i];
            }
            Console.WriteLine("Running Irls Qr Newton");
            GlmIrls solver = new GlmIrls(GlmDistributionFamily.Binomial, A, b);
            double[] x = solver.Solve();

            Console.WriteLine("(Intercept): {0}", x[0]);
            Console.WriteLine("age: {0}", x[1]);
            Console.WriteLine("I(age^2): {0}", x[2]);
            Console.WriteLine("urbanY: {0}", x[3]);
            Console.WriteLine("livch1: {0}", x[4]);
            Console.WriteLine("livch2: {0}", x[5]);
            Console.WriteLine("livch3: {0}", x[6]);


        }
    }
}
