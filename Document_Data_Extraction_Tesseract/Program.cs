using Microsoft.ML;
using Microsoft.ML.Transforms.Text;
using Document_Data_Extraction_Services;

class Program
{
    public static void ExtractDocumentText(string documentPath, OCRService ocr)
    {
        // executes ocr process and writes the result to the console
        var output = ocr.ExtractText(documentPath);
        Console.WriteLine($"This is the OCR result: {output}");
    }

    public static void ExecuteDocumentClassificationEvaluation(string classificationTrainingDataPath, DocumentProcessorService documentProcessor)
    {

        // executes evaluation of classification prediction process and writes the result to the console
        documentProcessor.TrainDocumentClassifierForEvaluation(classificationTrainingDataPath);
        var classificationMetricResult = documentProcessor.EvaluateDocumentLabelPrediction();
        Console.WriteLine($"Log-loss: {classificationMetricResult?.LogLoss}"); // value between 0 and 1, the lower the value, the better the predictions
        Console.WriteLine($"Per-Class Log-loss: {string.Join(", ", classificationMetricResult.PerClassLogLoss)}"); // each value provides information about the accuracy of predictions for a specific label

    }

    public static void ExecuteDocumentClassification(string classificationTrainingDataPath, string classificationDocumentPath, DocumentProcessorService documentProcessor, OCRService ocr)
    {
        // executes classification prediction process and writes the result to the console
        documentProcessor.TrainDocumentClassifier(classificationTrainingDataPath);
        var newDocument = new Document { FilePath = classificationDocumentPath };
        newDocument.Text = ocr.ExtractTextAsync(newDocument.FilePath).GetAwaiter().GetResult();
        var classificationResult = documentProcessor.PredictDocumentLabel(newDocument);
        Console.WriteLine($"Predicted Document Type: {classificationResult.PredictedLabel}");
    }

    static void Main(string[] args)
    {
        // this provides a path to the language data for tesseract
        var tessDataPath = Path.Combine(Environment.CurrentDirectory, @"..\..\..\tessdata");

        // this provides a path to the image file, used for the ocr process
        var ocrDocumentPath = Path.Combine(Environment.CurrentDirectory, @"..\..\..\test_data\Test_Document-1.png");

        // this provides a path to the image file, used for the classification process
        var classificationDocumentPath = Path.Combine(Environment.CurrentDirectory, @"..\..\..\test_data\invoice_10.png");
        
        // this provides a path to the csv file which contains learning data for the model to classify documents
        string classificationTrainingDataPath = Path.Combine(Environment.CurrentDirectory, @"..\..\..\classifier_data\classifierdata.csv");

        var ocr = new OCRService(tessDataPath);
        var documentProcessor = new DocumentProcessorService(ocr);

        ExecuteDocumentClassificationEvaluation(classificationTrainingDataPath, documentProcessor);

        // executes ocr process and writes all word bounding boxes to a json
        //ocr.GenerateBoundingBoxesDataFileForDataExtraction(ocrDocumentPath);
    }
}