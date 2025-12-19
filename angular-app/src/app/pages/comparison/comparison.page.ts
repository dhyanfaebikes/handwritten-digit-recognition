import { AfterViewInit, Component, ElementRef, OnDestroy, OnInit, ViewChild, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { Chart, ChartConfiguration, registerables } from 'chart.js';
import { DrawingCanvas } from '../../components/drawing-canvas/drawing-canvas';
import { Model, ModelType, PredictionResult } from '../../services/model';

Chart.register(...registerables);

type ComparisonResults = Record<ModelType, PredictionResult>;

@Component({
  selector: 'app-comparison-page',
  imports: [CommonModule, RouterLink, DrawingCanvas],
  templateUrl: './comparison.page.html',
  styleUrl: './comparison.page.scss',
})
export class ComparisonPage implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('drawingCanvas') drawingCanvas!: DrawingCanvas;
  @ViewChild('topConfidenceCanvas') topConfidenceCanvas!: ElementRef<HTMLCanvasElement>;

  isModelLoading = signal(true);
  isComparing = signal(false);
  error = signal<string | null>(null);
  results = signal<ComparisonResults | null>(null);

  private topConfidenceChart: Chart | null = null;

  readonly modelOrder: ModelType[] = ['cnn', 'ann', 'svm', 'knn', 'logistic_regression'];
  readonly modelLabel: Record<ModelType, string> = {
    cnn: 'CNN',
    ann: 'ANN',
    svm: 'SVM',
    knn: 'KNN',
    logistic_regression: 'Logistic Regression',
  };

  constructor(private modelService: Model) {}

  async ngOnInit(): Promise<void> {
    try {
      await this.modelService.loadModel();
    } finally {
      this.isModelLoading.set(false);
    }
  }

  ngAfterViewInit(): void {
    this.initCharts();
  }

  ngOnDestroy(): void {
    this.topConfidenceChart?.destroy();
  }

  handleClear(): void {
    this.drawingCanvas?.clear();
    this.error.set(null);
    this.results.set(null);

    if (this.topConfidenceChart) {
      this.topConfidenceChart.data.datasets[0].data = this.modelOrder.map(() => 0);
      this.topConfidenceChart.update('none');
    }
  }

  async handleCompare(): Promise<void> {
    if (!this.drawingCanvas) return;
    if (!this.modelService.getModel()) {
      this.error.set('Model not loaded yet. Please wait a moment and try again.');
      return;
    }

    const imageData = this.drawingCanvas.getImageData();
    if (!imageData) return;

    this.isComparing.set(true);
    this.error.set(null);

    try {
      const results: Partial<ComparisonResults> = {};
      for (const modelType of this.modelOrder) {
        this.modelService.setModelType(modelType);
        results[modelType] = await this.modelService.predict(imageData);
      }

      this.results.set(results as ComparisonResults);
      this.updateCharts(results as ComparisonResults);
    } catch (e: any) {
      this.error.set(e?.message || 'Comparison failed');
    } finally {
      this.isComparing.set(false);
    }
  }

  private initCharts(): void {
    const topCtx = this.topConfidenceCanvas?.nativeElement.getContext('2d');
    if (!topCtx) return;

    const topConfig: ChartConfiguration<'bar'> = {
      type: 'bar',
      data: {
        labels: this.modelOrder.map((m) => this.modelLabel[m]),
        datasets: [
          {
            label: 'Top Prediction Confidence (%)',
            data: this.modelOrder.map(() => 0),
            backgroundColor: 'rgba(37, 99, 235, 0.6)',
            borderColor: 'rgba(37, 99, 235, 1)',
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        animation: { duration: 500 },
        color: '#eaf0ff',
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            title: { display: true, text: 'Confidence (%)', color: '#eaf0ff' },
            ticks: { color: '#eaf0ff' },
            grid: { color: 'rgba(234, 240, 255, 0.14)' },
          },
          x: {
            title: { display: true, text: 'Model', color: '#eaf0ff' },
            ticks: { color: '#eaf0ff' },
            grid: { color: 'rgba(234, 240, 255, 0.10)' },
          },
        },
        plugins: {
          title: { display: true, text: 'Comparison: Top Prediction Confidence', color: '#eaf0ff' },
          legend: { display: false },
        },
      },
    };

    this.topConfidenceChart = new Chart(topCtx, topConfig);
  }

  private updateCharts(results: ComparisonResults): void {
    if (!this.topConfidenceChart) return;

    const confidences = this.modelOrder.map((m) => (results[m].confidence * 100));
    this.topConfidenceChart.data.datasets[0].data = confidences;
    this.topConfidenceChart.update('active');
  }
}


