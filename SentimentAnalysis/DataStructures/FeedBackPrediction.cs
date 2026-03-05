using Microsoft.ML.Data;

namespace SentimentAnalysis.DataStructures
{
    public class FeedBackPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsGood { get; set; }

        public float Probability { get; set; }

    }
}
