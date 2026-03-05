using Microsoft.ML;
using SentimentAnalysis.DataStructures;

namespace SentimentAnalysis
{
    internal static class Program
    {
        //Declaring function for model paths
        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory?.FullName ?? throw new Exception("Assembly directory not found.");

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
        //Input Path Declaration
        private static readonly string baseDataPath = @"../../../Data";
        private static readonly string trainRelativePath = $"{baseDataPath}/train.tsv";
        private static readonly string trainPath = GetAbsolutePath(trainRelativePath);
        private static readonly string testRelativePath = $"{baseDataPath}/test.tsv";
        private static readonly string testPath = GetAbsolutePath(testRelativePath);

        //Model Path Declaration
        private static readonly string baseModelPath = @"../../../Data";
        private static readonly string modelRelativePath = $"{baseModelPath}/model.zip";
        private static readonly string modelPath = GetAbsolutePath(modelRelativePath);

        //Some Beautiful Animation
        static T RunWithSpinner<T>(string text, Func<T> work)
        {
            char[] seq = new[] { '.', '/', '-', '\\' };
            int i = 0;

            var t = Task.Run(work);

            while (!t.IsCompleted)
            {
                Console.Write($"\r{text} {seq[i++ % seq.Length]}");
                Thread.Sleep(120);
            }

            Console.Write("\r" + new string(' ', text.Length + 3) + "\r"); // clear line
            return t.GetAwaiter().GetResult();
        }


        public static void Main(string[] args)
        {
            Console.ForegroundColor = ConsoleColor.White;
            // Creating status variable for while loop.
            string status = "undefined";

            // Creating MLContext.
            var mlContext = new MLContext(seed: 1);

            // Dataset existence check (train/test are required)
            if (!File.Exists(trainPath) || !File.Exists(testPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("Dataset files not found. Please check README.");
                Console.ForegroundColor = ConsoleColor.White;
                return;
            }

            //Data loading configuration.
            IDataView trainingData = mlContext.Data.LoadFromTextFile<FeedBackTrainingData>(trainPath, hasHeader: true);
            IDataView testData = mlContext.Data.LoadFromTextFile<FeedBackTrainingData>(testPath, hasHeader: true);

            //Data process configuration with pipeline data transformations.
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(FeedBackTrainingData.FeedBackText));

            //Training algorithm and model builder.
            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
            var pipeline = dataProcessPipeline.Append(trainer);

            //Training the model fitting to the DataSet and saving trained model
            ITransformer trainedmodel;

            if (File.Exists(modelPath))
            {
                Console.WriteLine("Pre-trained model found.\n");
                try
                {
                    trainedmodel = RunWithSpinner("Loading existing model", () => mlContext.Model.Load(modelPath, out _));
                }
                catch
                {
                    Console.WriteLine("Model load failed. Re-training...\n");
                    trainedmodel = RunWithSpinner("Training model", () => pipeline.Fit(trainingData));
                    mlContext.Model.Save(trainedmodel, trainingData.Schema, modelPath);
                }
            }
            else
            {
                Console.WriteLine("Pre-trained model not found.\n");
                trainedmodel = RunWithSpinner("Training model", () => pipeline.Fit(trainingData));
                mlContext.Model.Save(trainedmodel, trainingData.Schema, modelPath);
            }

            //Evaluating the model and showing accuracy
            var predictions = RunWithSpinner("Running predictions on test set", () => trainedmodel.Transform(testData));
            var metrics = RunWithSpinner("Evaluating metrics", () =>
                mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score"));
            double acc = metrics.Accuracy;
            acc *= 1e2;

            Console.Clear();
            Console.WriteLine("===================================== Evaluating Model Accuracy =====================================\n");
            Console.WriteLine("    ACCURACY = " + metrics.Accuracy.ToString("F4"));
            Console.WriteLine("    The model achieved " + acc.ToString("F2") + "% accuracy on the test set.");
            Console.WriteLine("\n=====================================================================================================");

            //Creating prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<FeedBackTrainingData, FeedBackPrediction>(trainedmodel);

            string? strcont = "Y";
            while (strcont != "n" && strcont != "N")
            {
                double negative;
                double positive;
                //Getting input string
                Console.Write("\n    Enter a feedback string: ");
                string? feedbackstring = Console.ReadLine();

                if (string.IsNullOrEmpty(feedbackstring))
                {
                    Console.WriteLine("\n    Please enter an input.");
                    continue;
                }
                FeedBackTrainingData feedbackinput = new FeedBackTrainingData();
                feedbackinput.FeedBackText = feedbackstring;

                //Prediction Result
                var prediction = predictionEngine.Predict(feedbackinput);

                if (!prediction.IsGood)
                {
                    status = "NEGATIVE";
                }
                else if (prediction.IsGood)
                {
                    status = "POSITIVE";
                }
                Console.Write("\n    Predicted: ");

                //Setting some variables to see % probability
                negative = 1 - (Convert.ToDouble(prediction.Probability));
                string negshorted = negative.ToString("0.##");
                double negchanged = Convert.ToDouble(negshorted);
                negchanged *= 1e2;
                positive = (Convert.ToDouble(prediction.Probability));
                string posshorted = positive.ToString("0.##");
                double poschanged = Convert.ToDouble(posshorted);
                poschanged *= 1e2;

                if (!prediction.IsGood)
                {
                    Console.Write("\"" + feedbackstring + "\" is a ");
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Write(status);
                    Console.ForegroundColor = ConsoleColor.White;
                    Console.WriteLine(" string." + "\n    Probability of being negative: " + negchanged.ToString("F2") + "%");
                    Console.WriteLine("\n                    Negative:    " + Convert.ToDouble(negshorted).ToString("F2") + "/1");
                    Console.WriteLine("    Probabilities: -----------------------");
                    Console.WriteLine("                    Positive:    " + Convert.ToDouble(posshorted).ToString("F2") + "/1");
                }
                else if (prediction.IsGood)
                {
                    Console.Write("\"" + feedbackstring + "\" is a ");
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.Write(status);
                    Console.ForegroundColor = ConsoleColor.White;
                    Console.WriteLine(" string." + "\n    Probability of being positive: " + poschanged.ToString("F2") + "%");
                    Console.WriteLine("\n                    Positive:    " + Convert.ToDouble(posshorted).ToString("F2") + "/1");
                    Console.WriteLine("    Probabilities: -----------------------");
                    Console.WriteLine("                    Negative:    " + Convert.ToDouble(negshorted).ToString("F2") + "/1");
                }
                Console.WriteLine("\n======================= Hit n/N to stop process. Hit another key to continue. =======================\n");
                Console.Write("    ==> ");
                strcont = Console.ReadLine();

                if (strcont is null)
                {
                    Console.WriteLine("Input stream closed.");
                    break;
                }
                Console.WriteLine("\n=====================================================================================================");
            }
        }
    }
}