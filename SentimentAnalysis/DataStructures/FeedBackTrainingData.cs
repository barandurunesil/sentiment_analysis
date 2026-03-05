using Microsoft.ML.Data;

namespace SentimentAnalysis.DataStructures
{
    public class FeedBackTrainingData
    {

        [LoadColumn(0), ColumnName("Label")]
        public bool IsGood { get; set; }

        [LoadColumn(1)]
        public string FeedBackText { get; set; }

    }
}
