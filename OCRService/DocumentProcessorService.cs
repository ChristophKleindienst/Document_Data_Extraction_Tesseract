using Microsoft.ML;
using Microsoft.ML.Data;
using Tesseract;

namespace Document_Data_Extraction_Services
{
    public class DocumentProcessorService : IDocumentProcessorService
    {
        public MLContext MlContext { get; set; }
        public ITransformer Model { get; set; } 
        public OCRService OcrService { get; set; }
        public DataOperationsCatalog.TrainTestData TrainTestData { get; set; }
        private static readonly string ClassificationModelPath = Path.Combine(Environment.CurrentDirectory, @"..\..\..\classifier_data\documentClassificationModel.zip");

        public DocumentProcessorService(OCRService ocrService) 
        {
            OcrService = ocrService;
            MlContext = new MLContext();
        }

        // train machine learning model for document classification
        public void TrainDocumentClassifier(string trainingDataPath, bool trainNewModel = true)
        {
            MlContext = new MLContext();

            if (trainNewModel)
            {
                var data = MlContext.Data.LoadFromTextFile<Document>(trainingDataPath, separatorChar: ';', hasHeader: true);

                var textData = new List<Document>();

                foreach (var item in MlContext.Data.CreateEnumerable<Document>(data, reuseRowObject: false))
                {
                    var text = OcrService.ExtractTextAsync(item.FilePath).GetAwaiter().GetResult();

                    textData.Add(new Document { 
                        FilePath = item.FilePath, 
                        Text = text, 
                        Label = item.Label 
                    });
                }

                var trainingData = MlContext.Data.LoadFromEnumerable(textData);

                var dataPipeline = MlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(Document.Text))
                        .Append(MlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(Document.Label)))
                        .AppendCacheCheckpoint(MlContext)
                        .Append(MlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                        .Append(MlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
                Model = dataPipeline.Fit(trainingData);

                MlContext.Model.Save(Model, trainingData.Schema, ClassificationModelPath);
            }
            else
            {
                Model = MlContext.Model.Load(ClassificationModelPath, out _);
            }
        }

        // make prediction for the classification of a new document
        public DocumentPrediction PredictDocumentLabel(Document document)
        {
            var predictionEngine = MlContext.Model.CreatePredictionEngine<Document, DocumentPrediction>(Model);
            return predictionEngine.Predict(document);
        }
        
        public void TrainDocumentClassifierForEvaluation(string trainingDataPath, bool trainNewModel = true)
        {
            MlContext = new MLContext();

            if (trainNewModel)
            {
                var data = MlContext.Data.LoadFromTextFile<Document>(trainingDataPath, separatorChar: ';', hasHeader: true);

                var textData = new List<Document>();

                foreach (var item in MlContext.Data.CreateEnumerable<Document>(data, reuseRowObject: false))
                {
                    var text = OcrService.ExtractTextAsync(item.FilePath).GetAwaiter().GetResult();

                    textData.Add(new Document
                    {
                        FilePath = item.FilePath,
                        Text = text,
                        Label = item.Label
                    });
                }

                var trainingData = MlContext.Data.LoadFromEnumerable(textData);

                TrainTestData = MlContext.Data.TrainTestSplit(trainingData, testFraction: 0.3);

                var dataPipeline = MlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(Document.Text))
                        .Append(MlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: nameof(Document.Label)))
                        .AppendCacheCheckpoint(MlContext)
                        .Append(MlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("LabelKey", "Features"))
                        .Append(MlContext.Transforms.CopyColumns(outputColumnName: "PredictedLabelKey", inputColumnName: "PredictedLabel"))
                        .Append(MlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabelKey"));

                // k-fold cross-validation (k=5)
                var cvResults = MlContext.MulticlassClassification.CrossValidate(trainingData, dataPipeline, labelColumnName: "LabelKey", numberOfFolds: 5);

                foreach (var result in cvResults)
                {
                    Console.WriteLine($"Fold: {result.Fold}, LogLoss: {result.Metrics.LogLoss}");
                }

                Model = dataPipeline.Fit(TrainTestData.TrainSet);

                MlContext.Model.Save(Model, TrainTestData.TrainSet.Schema, ClassificationModelPath);
            }
            else
            {
                Model = MlContext.Model.Load(ClassificationModelPath, out _);
            }
        }

        public MulticlassClassificationMetrics? EvaluateDocumentLabelPrediction()
        {
            var predictions = Model.Transform(TrainTestData.TestSet);

            var predictionPreview = MlContext.Data.CreateEnumerable<DocumentPrediction>(predictions, reuseRowObject: false).ToList();

            foreach (var pred in predictionPreview)
            {
                Console.WriteLine($"PredictedLabel: {pred.PredictedLabel}, Score: {string.Join(", ", pred.Score)}");
            }

            var metrics = MlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelKey", scoreColumnName: "Score");
            
            // Print confusion matrix
            var confusionMatrix = metrics.ConfusionMatrix;
            Console.WriteLine("Confusion Matrix:");
            foreach (var row in confusionMatrix.Counts)
            {
                Console.WriteLine(string.Join("\t", row));
            }

            return metrics;
        }
    }
}
