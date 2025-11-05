import { Injectable, signal } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

export interface PredictionResult {
  digit: number;
  confidence: number;
  probabilities: number[];
  timestamp: Date;
}

@Injectable({
  providedIn: 'root',
})
export class Model {
  private model: tf.LayersModel | null = null;
  private isLoading = signal(false);
  private isLoaded = signal(false);
  private error = signal<string | null>(null);

  isLoadingSignal = this.isLoading.asReadonly();
  isLoadedSignal = this.isLoaded.asReadonly();
  errorSignal = this.error.asReadonly();

  async loadModel(): Promise<void> {
    if (this.model) {
      return;
    }

    this.isLoading.set(true);
    this.error.set(null);

    // Load model.json from src/ folder, weights will be fetched from FastAPI server
    try {
      // First, verify the API is accessible
      console.log('🔍 Checking FastAPI server...');
      try {
        const apiCheck = await fetch('http://localhost:6500/');
        const apiStatus = await apiCheck.json();
        console.log('✅ FastAPI server is running:', apiStatus);
      } catch (apiError) {
        console.error('❌ FastAPI server is NOT accessible:', apiError);
        throw new Error('FastAPI server is not running. Please start it with: cd train && python main.py');
      }

      // Load model.json from src folder (served at /model.json)
      // The weights path in model.json points to http://localhost:6500/group1-shard1of1
      console.log('📦 Loading model from /model.json...');
      
      // First, let's verify model.json is accessible and check its contents
      try {
        const modelJsonResponse = await fetch('/model.json');
        const modelJsonData = await modelJsonResponse.json();
        console.log('✅ model.json loaded successfully');
        console.log('📋 Weights manifest path:', modelJsonData.weightsManifest[0].paths[0]);
        console.log('🌐 TensorFlow.js will fetch weights from:', modelJsonData.weightsManifest[0].paths[0]);
        
        // Test the weights endpoint directly to see what we get
        const weightsUrl = modelJsonData.weightsManifest[0].paths[0];
        console.log('🧪 Testing weights endpoint directly...');
        try {
          const weightsTest = await fetch(weightsUrl);
          const arrayBuffer = await weightsTest.arrayBuffer();
          console.log('✅ Weights endpoint test:');
          console.log(`   Status: ${weightsTest.status}`);
          console.log(`   Content-Type: ${weightsTest.headers.get('content-type')}`);
          console.log(`   Content-Length: ${weightsTest.headers.get('content-length')}`);
          console.log(`   ArrayBuffer size: ${arrayBuffer.byteLength} bytes`);
          console.log(`   Divisible by 4: ${arrayBuffer.byteLength % 4 === 0}`);
          
          if (arrayBuffer.byteLength !== 3550120) {
            console.error(`❌ ERROR: Expected 3550120 bytes, got ${arrayBuffer.byteLength} bytes`);
            console.error('   This is why TensorFlow.js is failing!');
          }
        } catch (testError) {
          console.error('❌ Failed to test weights endpoint:', testError);
        }
      } catch (e) {
        console.error('❌ Failed to load model.json:', e);
      }
      
      // Load model using a custom IO handler to ensure absolute URLs work correctly
      console.log('🔄 Calling tf.loadLayersModel with custom IO handler...');
      
      // Create a custom IO handler that properly handles absolute URLs
      const customIOHandler = tf.io.browserHTTPRequest('/model.json', {
        // Don't modify paths - use absolute URLs as-is from model.json
        weightPathPrefix: '',
        fetchFunc: async (input: RequestInfo | URL, init?: RequestInit) => {
          // Convert input to string URL
          let url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url;
          console.log(`🔍 TensorFlow.js fetching: ${url}`);
          
          // Fix: TensorFlow.js prepends "/" to absolute URLs, making them invalid
          // If URL starts with "/http://" or "/https://", remove the leading "/"
          if (url.startsWith('/http://') || url.startsWith('/https://')) {
            console.log(`🔧 Fixing malformed URL: ${url}`);
            url = url.substring(1); // Remove leading "/"
            console.log(`✅ Fixed URL: ${url}`);
          }
          
          // For binary weights, ensure we get ArrayBuffer
          if (url.includes('group1-shard1of1')) {
            console.log(`📥 Fetching weights binary from: ${url}`);
            const response = await fetch(url, {
              ...init,
              // Explicitly request binary data
            });
            
            if (!response.ok) {
              console.error(`❌ Weights fetch failed: ${response.status} ${response.statusText}`);
              throw new Error(`Failed to fetch weights: ${response.status} ${response.statusText}`);
            }
            
            const arrayBuffer = await response.arrayBuffer();
            console.log(`✅ Received ${arrayBuffer.byteLength} bytes (divisible by 4: ${arrayBuffer.byteLength % 4 === 0})`);
            
            if (arrayBuffer.byteLength !== 3550120) {
              console.error(`❌ ERROR: Expected 3550120 bytes, got ${arrayBuffer.byteLength} bytes`);
              throw new Error(`Invalid weights size: expected 3550120, got ${arrayBuffer.byteLength}`);
            }
            
            // Return a Response with the ArrayBuffer
            return new Response(arrayBuffer, {
              status: response.status,
              statusText: response.statusText,
              headers: response.headers,
            });
          }
          
          // For model.json, return as-is
          return fetch(url, init);
        },
      });
      
      this.model = await tf.loadLayersModel(customIOHandler);
      
      console.log('✅ MNIST CNN model loaded successfully');
      console.log('✅ Model loaded from /model.json');
      console.log('✅ Weights fetched from FastAPI server');
    } catch (error: any) {
      console.error('❌ Model loading failed:', error);
      console.error('Error details:', error.message);
      console.error('Error stack:', error.stack);
      
      // Check if it's the Float32Array error
      if (error.message && error.message.includes('Float32Array')) {
        console.error('⚠️ Float32Array error detected - this usually means:');
        console.error('  1. The binary weights file is corrupted');
        console.error('  2. The response from FastAPI is not being read correctly');
        console.error('  3. CORS might be blocking the binary data');
        console.error('');
        console.error('🔧 Debug steps:');
        console.error('  1. Check browser Network tab for http://localhost:6500/group1-shard1of1');
        console.error('  2. Verify the response is 3550120 bytes');
        console.error('  3. Check if CORS headers are present');
      }
      
      console.error('');
      console.error('Make sure:');
      console.error('  1. FastAPI server is running: curl http://localhost:6500/');
      console.error('  2. Weights endpoint works: curl http://localhost:6500/group1-shard1of1');
      console.error('  3. model.json is accessible at /model.json');
      
      // Fallback: create a simple model for demonstration
      console.log('');
      console.log('⚠️ Using fallback model (untrained - will give random predictions)');
      this.model = this.createSimpleModel();
    }

    this.isLoaded.set(true);
    this.isLoading.set(false);
  }

  // EXACT COPY from index.html lines 183-194
  private createSimpleModel(): tf.LayersModel {
    // Create a simple model for demonstration that matches MNIST input shape
    const model = tf.sequential({
      layers: [
        tf.layers.flatten({inputShape: [28, 28, 1]}),
        tf.layers.dense({units: 128, activation: 'relu'}),
        tf.layers.dropout({rate: 0.2}),
        tf.layers.dense({units: 10, activation: 'softmax'})
      ]
    });
    return model;
  }

  async predict(imageData: ImageData): Promise<PredictionResult> {
    if (!this.model) {
      throw new Error('Model not loaded yet');
    }

    // EXACT COPY from index.html lines 246-302
    console.log('Model loaded:', !!this.model);
    console.log('Model input shape:', this.model.inputs[0].shape);

    try {
      // Get image data from canvas
      // Convert to tensor and preprocess for MNIST (28x28 grayscale)
      let tensor = tf.browser.fromPixels(imageData, 1)
        .resizeNearestNeighbor([28, 28])
        .expandDims(0) // [1, 28, 28, 1]
        .div(255.0);
      
      console.log('Tensor shape:', tensor.shape);

      // Make prediction
      console.log('Making prediction...');
      const prediction = this.model.predict(tensor) as tf.Tensor;
      console.log('Prediction tensor shape:', prediction.shape);
      const values = await prediction.data();
      console.log('Raw prediction values:', values);
      
      // Find the digit with highest probability
      const maxIndex = values.indexOf(Math.max(...values));
      const confidence = (values[maxIndex] * 100).toFixed(1);
      
      // Update chart
      console.log('Prediction values:', values);
      const valuesArray = Array.from(values);
      
      // Clean up
      tensor.dispose();
      prediction.dispose();

      return {
        digit: maxIndex,
        confidence: parseFloat(confidence) / 100,
        probabilities: valuesArray as number[],
        timestamp: new Date()
      };
    } catch (error: any) {
      console.error('Prediction error:', error);
      console.error('Error details:', error.message);
      console.error('Error stack:', error.stack);
      throw new Error(`Prediction failed: ${error.message}`);
    }
  }

  async predictWithIntermediateOutputs(imageData: ImageData): Promise<{prediction: PredictionResult, intermediateOutputs: any}> {
    if (!this.model) {
      throw new Error('Model not loaded');
    }

    try {
      // Helper function to calculate stats from tensor values
      const calculateStats = (values: number[]) => {
        const min = Math.min(...values);
        const max = Math.max(...values);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
        const stdDev = Math.sqrt(variance);
        const nonZero = values.filter(v => v > 0.01).length;
        
        // Percentiles
        const sorted = [...values].sort((a, b) => a - b);
        const p25 = sorted[Math.floor(sorted.length * 0.25)];
        const p50 = sorted[Math.floor(sorted.length * 0.50)];
        const p75 = sorted[Math.floor(sorted.length * 0.75)];
        const p90 = sorted[Math.floor(sorted.length * 0.90)];
        const p95 = sorted[Math.floor(sorted.length * 0.95)];
        
        // Top activations
        const sortedDesc = [...values].sort((a, b) => b - a);
        const top1 = sortedDesc[0];
        const top5 = sortedDesc.slice(0, 5).reduce((a, b) => a + b, 0) / 5;
        const top10 = sortedDesc.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
        
        // Additional metrics
        const sparsity = 1 - (nonZero / values.length);
        const range = max - min;
        const coefficientOfVariation = mean > 0 ? stdDev / mean : 0;
        
        // Entropy
        const normalized = values.map(v => Math.max(v, 1e-10));
        const sum = normalized.reduce((a, b) => a + b, 0);
        const probabilities = normalized.map(v => v / sum);
        const entropy = -probabilities.reduce((ent, p) => ent + (p > 0 ? p * Math.log2(p) : 0), 0);
        
        return {
          min, max, mean, stdDev, nonZero,
          p25, p50, p75, p90, p95,
          top1, top5, top10,
          sparsity, range, coefficientOfVariation, entropy
        };
      };
      
      // Use tf.tidy to manage memory for tensor operations
      const { inputTensor, intermediateTensors } = tf.tidy(() => {
        // Preprocess input
        const inputTensor = tf.browser.fromPixels(imageData, 1)
          .resizeNearestNeighbor([28, 28])
          .expandDims(0)
          .div(255.0);
        
        // Get model layers
        const layers = this.model!.layers;
        
        // Log all layer names for debugging
        console.log('Model layers:', layers.map(l => ({ name: l.name, className: l.getClassName() })));
        
        // Process through layers sequentially and collect intermediate tensors
        let currentOutput = inputTensor;
        const intermediateTensors: { [key: string]: tf.Tensor } = {};
        
        // Track layer indices for capturing
        let convCount = 0;
        let poolCount = 0;
        let denseCount = 0;
        
        for (let i = 0; i < layers.length; i++) {
          const layer = layers[i];
          const className = layer.getClassName();
          
          // Apply layer - dropout is automatically disabled during inference
          if (className === 'Dropout') {
            currentOutput = (layer as any).call(currentOutput, { training: false }) as tf.Tensor;
          } else {
            currentOutput = layer.apply(currentOutput) as tf.Tensor;
          }
          
          // Capture layers based on type and position
          if (className === 'Conv2D') {
            convCount++;
            if (convCount === 1) {
              intermediateTensors['conv1'] = currentOutput.clone();
              console.log('Captured conv1:', layer.name, currentOutput.shape);
            } else if (convCount === 3) {
              intermediateTensors['conv2'] = currentOutput.clone();
              console.log('Captured conv2:', layer.name, currentOutput.shape);
            }
          } else if (className === 'MaxPooling2D') {
            poolCount++;
            if (poolCount === 1) {
              intermediateTensors['pool1'] = currentOutput.clone();
              console.log('Captured pool1:', layer.name, currentOutput.shape);
            } else if (poolCount === 2) {
              intermediateTensors['pool2'] = currentOutput.clone();
              console.log('Captured pool2:', layer.name, currentOutput.shape);
            }
          } else if (className === 'Flatten') {
            intermediateTensors['flatten'] = currentOutput.clone();
            console.log('Captured flatten:', layer.name, currentOutput.shape);
          } else if (className === 'Dense') {
            denseCount++;
            if (denseCount === 1) {
              intermediateTensors['dense'] = currentOutput.clone();
              console.log('Captured dense:', layer.name, currentOutput.shape);
            } else if (denseCount === 2) {
              intermediateTensors['output'] = currentOutput.clone();
              intermediateTensors['final'] = currentOutput.clone();
              console.log('Captured output:', layer.name, currentOutput.shape);
            }
          }
        }
        
        return { inputTensor: inputTensor.clone(), intermediateTensors };
      });
      
      // Extract data from tensors (now outside tf.tidy, so they won't be disposed)
      const intermediateOutputs: any = {};
      
      // Input layer stats
      const inputData = inputTensor.dataSync();
      const inputValues = Array.from(inputData);
      const inputStats = calculateStats(inputValues);
      
      intermediateOutputs.input = {
        shape: [1, 28, 28, 1],
        count: 784,
        ...inputStats
      };
      
      console.log('Input stats:', {
        mean: inputStats.mean.toFixed(4),
        top1: inputStats.top1.toFixed(4),
        entropy: inputStats.entropy.toFixed(4)
      });
      
      // Process intermediate tensors
      const layerKeys = ['conv1', 'pool1', 'conv2', 'pool2', 'flatten', 'dense', 'output'];
      
      console.log('Available intermediate tensors:', Object.keys(intermediateTensors));
      
      for (const key of layerKeys) {
        const tensor = intermediateTensors[key];
        if (tensor) {
          const outputData = tensor.dataSync();
          const values = Array.from(outputData);
          const stats = calculateStats(values);
          
          const shape = tensor.shape;
          const count = shape.reduce((a, b) => a * b, 1);
          
          intermediateOutputs[key] = {
            shape: shape,
            count: count,
            ...stats
          };
          
          console.log(`${key} stats:`, {
            mean: stats.mean.toFixed(4),
            top1: stats.top1.toFixed(4),
            entropy: stats.entropy.toFixed(4),
            sparsity: (stats.sparsity * 100).toFixed(1) + '%'
          });
        } else {
          console.warn(`Missing tensor for layer: ${key}`);
        }
      }
      
      console.log('Final intermediateOutputs:', Object.keys(intermediateOutputs));
      
      // Get final prediction
      const finalTensor = intermediateTensors['final'] || intermediateTensors['output'];
      const finalValues = finalTensor ? Array.from(finalTensor.dataSync()) : [];
      const valuesArray = finalValues.length > 0 ? finalValues : [];
      const maxIndex = valuesArray.length > 0 ? valuesArray.indexOf(Math.max(...valuesArray)) : 0;
      const confidence = valuesArray.length > 0 ? valuesArray[maxIndex] : 0;
      
      const prediction: PredictionResult = {
        digit: maxIndex,
        confidence: confidence,
        probabilities: valuesArray,
        timestamp: new Date()
      };
      
      // Clean up cloned tensors
      inputTensor.dispose();
      Object.values(intermediateTensors).forEach(t => t.dispose());
      
      return { prediction, intermediateOutputs };
    } catch (error: any) {
      console.error('Prediction with intermediate outputs error:', error);
      throw new Error(`Prediction failed: ${error.message}`);
    }
  }

  getModel(): tf.LayersModel | null {
    return this.model;
  }
}
