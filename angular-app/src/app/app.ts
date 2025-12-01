import { Component, OnInit, signal, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DrawingCanvas } from './components/drawing-canvas/drawing-canvas';
import { ConfidenceChart } from './components/confidence-chart/confidence-chart';
import { Model, PredictionResult, ModelType } from './services/model';
import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'app-root',
  imports: [
    CommonModule,
    FormsModule,
    DrawingCanvas,
    ConfidenceChart
  ],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App implements OnInit, AfterViewInit {
  @ViewChild('drawingCanvas') drawingCanvas!: DrawingCanvas;
  @ViewChild('preprocessedCanvas') preprocessedCanvas!: ElementRef<HTMLCanvasElement>;
  
  currentPrediction = signal<PredictionResult | null>(null);
  isModelLoading = signal(true);
  resultText = signal('Draw a digit and click Predict');
  tensorStats = signal<any>(null);
  isProcessing = signal(false);
  inferenceTime = signal(0);
  selectedModel = signal<ModelType>('cnn');
  availableModels = signal<{ [key: string]: boolean }>({});
  
  modelOptions: { value: ModelType; label: string; accuracy: string }[] = [
    { value: 'cnn', label: 'CNN', accuracy: '~100%' },
    { value: 'ann', label: 'ANN', accuracy: '~92%' },
    { value: 'svm', label: 'SVM', accuracy: '~92%' },
    { value: 'knn', label: 'KNN', accuracy: '~92%' },
    { value: 'logistic_regression', label: 'Logistic Regression', accuracy: '~80%' },
  ];

  constructor(
    private modelService: Model
  ) {}

  async ngOnInit(): Promise<void> {
    await this.checkAvailableModels();
    await this.loadModel();
  }
  
  async checkAvailableModels(): Promise<void> {
    const models = await this.modelService.getAvailableModels();
    this.availableModels.set(models);
  }
  
  async onModelChange(): Promise<void> {
    const modelType = this.selectedModel();
    this.modelService.setModelType(modelType);
    
    // Clear canvas when changing models
    if (this.drawingCanvas) {
      this.drawingCanvas.clear();
    }
    
    // Clear preprocessed canvas
    if (this.preprocessedCanvas) {
      const canvas = this.preprocessedCanvas.nativeElement;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
    
    this.currentPrediction.set(null);
    this.tensorStats.set(null);
    this.inferenceTime.set(0);
    this.resultText.set(`Model: ${this.getModelLabel(modelType)}. Draw a digit and click Infer.`);
  }
  
  getModelLabel(modelType: ModelType): string {
    const option = this.modelOptions.find(opt => opt.value === modelType);
    return option?.label || modelType;
  }
  
  isModelAvailable(modelType: ModelType): boolean {
    return true;
  }

  ngAfterViewInit(): void {
    // Initialize preprocessed canvas - use larger size for display (scale up 10x)
    if (this.preprocessedCanvas) {
      const canvas = this.preprocessedCanvas.nativeElement;
      const scale = 10; // Scale up 10x for better visibility
      canvas.width = 28 * scale;
      canvas.height = 28 * scale;
    }
  }

  async loadModel(): Promise<void> {
    try {
      await this.modelService.loadModel();
    } catch (error) {
      console.error('Failed to load model:', error);
    } finally {
      this.isModelLoading.set(false);
    }
  }

  async handlePredict(): Promise<void> {
    if (!this.modelService.getModel()) {
      this.resultText.set('Model not loaded yet');
      return;
    }

    if (this.drawingCanvas) {
      const imageData = this.drawingCanvas.getImageData();
      if (imageData) {
        try {
          this.isProcessing.set(true);
          const startTime = performance.now();
          
          // Preprocess and display the preprocessed image
          this.displayPreprocessedImage(imageData);
          
          // Use predictWithIntermediateOutputs to get tensor stats
          const result = await this.modelService.predictWithIntermediateOutputs(imageData);
          
          const endTime = performance.now();
          const elapsed = Math.round(endTime - startTime);
          
          this.currentPrediction.set(result.prediction);
          this.tensorStats.set(result.intermediateOutputs);
          this.inferenceTime.set(elapsed);
          
          const predictedDigit = result.prediction.digit;
          const confidence = (result.prediction.confidence * 100).toFixed(1);
          this.resultText.set(`Predicting you draw ${predictedDigit} with ${confidence}% confidence`);
        } catch (error: any) {
          console.error('Prediction error:', error);
          this.resultText.set(`Error: ${error.message}`);
          this.tensorStats.set(null);
        } finally {
          this.isProcessing.set(false);
        }
      }
    }
  }

  handleClear(): void {
    if (this.drawingCanvas) {
      this.drawingCanvas.clear();
    }
    
    // Clear preprocessed canvas
    if (this.preprocessedCanvas) {
      const canvas = this.preprocessedCanvas.nativeElement;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
    
    this.resultText.set('Draw a digit and click Predict');
    this.currentPrediction.set(null);
    this.tensorStats.set(null);
    this.inferenceTime.set(0);
    // Reset chart to zeros - EXACT COPY from index.html lines 164-168
    // This is handled in confidence-chart component via the probabilities input
  }

  get predictionProbabilities(): number[] {
    return this.currentPrediction()?.probabilities || [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  }

  isPredictionValid(): boolean {
    return this.currentPrediction() !== null;
  }

  getPredictionDisplay(): string {
    const prediction = this.currentPrediction();
    if (!prediction) return '';
    return prediction.digit.toString();
  }

  getPredictionLabel(): string {
    return 'PREDICTED CLASS';
  }

  // Helper function to format decimal numbers
  formatDecimal(value: number | undefined, decimals: number = 4): string {
    if (value === undefined || value === null || isNaN(value)) {
      return '0.0000';
    }
    return value.toFixed(decimals);
  }

  // Display the preprocessed 28x28 image on the canvas
  displayPreprocessedImage(imageData: ImageData): void {
    if (!this.preprocessedCanvas) return;
    
    const canvas = this.preprocessedCanvas.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Preprocess the image data to 28x28 grayscale (same as model input)
    const tensor = tf.tidy(() => {
      // Convert ImageData to tensor
      return tf.browser.fromPixels(imageData, 1)
        .resizeNearestNeighbor([28, 28])
        .expandDims(0)
        .div(255.0);
    });
    
    // Get the preprocessed data (outside tidy so tensor isn't disposed yet)
    const preprocessedTensor = tensor.squeeze([0]); // Remove batch dimension: [28, 28, 1]
    const [height, width] = preprocessedTensor.shape;
    const data = preprocessedTensor.dataSync();
    
    // Create ImageData for display
    const imgData = new ImageData(width, height);
    const imgDataArray = imgData.data;
    
    // Convert normalized float32 [0,1] back to uint8 [0,255] for display
    for (let i = 0; i < data.length; i++) {
      const pixelValue = Math.round(data[i] * 255);
      const idx = i * 4;
      imgDataArray[idx] = pixelValue;     // R
      imgDataArray[idx + 1] = pixelValue; // G
      imgDataArray[idx + 2] = pixelValue; // B
      imgDataArray[idx + 3] = 255;        // A
    }
    
    // Draw the preprocessed image to canvas (scaled up for visibility)
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Scale up by drawing each pixel as a larger block (pixelated look)
    const scale = canvas.width / 28; // Calculate scale factor
    ctx.imageSmoothingEnabled = false;
    
    // Create a temporary canvas with the ImageData
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    if (tempCtx) {
      tempCtx.putImageData(imgData, 0, 0);
      
      // Draw the small image scaled up to the display canvas
      ctx.drawImage(tempCanvas, 0, 0, 28, 28, 0, 0, canvas.width, canvas.height);
    }
    
    // Clean up tensors
    tensor.dispose();
    preprocessedTensor.dispose();
  }
}
