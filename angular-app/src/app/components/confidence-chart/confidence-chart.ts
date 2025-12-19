import { Component, input, AfterViewInit, ViewChild, ElementRef, effect } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Chart, ChartConfiguration, registerables } from 'chart.js';

Chart.register(...registerables);

@Component({
  selector: 'app-confidence-chart',
  imports: [CommonModule],
  templateUrl: './confidence-chart.html',
  styleUrl: './confidence-chart.scss',
})
export class ConfidenceChart implements AfterViewInit {
  @ViewChild('chartCanvas', { static: false }) chartCanvas!: ElementRef<HTMLCanvasElement>;
  
  probabilities = input<number[]>([]);
  modelType = input<string>('cnn');
  private chart: Chart | null = null;

  constructor() {
    effect(() => {
      const probs = this.probabilities();
      const model = this.modelType();
      if (this.chart && probs.length > 0) {
        this.updateChart(probs);
        if (this.chart.options.plugins?.title) {
          this.chart.options.plugins.title.text = this.getChartTitle();
          this.chart.update('none');
        }
      }
    });
  }

  ngAfterViewInit(): void {
    this.initChart();
    // Update chart if probabilities are already available
    const probs = this.probabilities();
    if (this.chart && probs.length > 0) {
      this.updateChart(probs);
    }
  }

  private initChart(): void {
    if (!this.chartCanvas) return;

    const ctx = this.chartCanvas.nativeElement.getContext('2d');
    if (!ctx) return;

    // EXACT COPY from index.html lines 196-244
    const config: ChartConfiguration = {
      type: 'bar',
      data: {
        labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        datasets: [{
          label: 'Prediction Confidence',
          data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          backgroundColor: 'rgba(6, 182, 212, 0.35)',
          borderColor: 'rgba(6, 182, 212, 0.95)',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        animation: {
          duration: 500
        },
        color: '#eaf0ff',
        scales: {
          y: {
            beginAtZero: true,
            max: 1.0,
            title: {
              display: true,
              text: 'Confidence',
              color: '#eaf0ff'
            },
            ticks: { color: '#eaf0ff' },
            grid: { color: 'rgba(234, 240, 255, 0.14)' }
          },
          x: {
            title: {
              display: true,
              text: 'Digit',
              color: '#eaf0ff'
            },
            ticks: { color: '#eaf0ff' },
            grid: { color: 'rgba(234, 240, 255, 0.10)' }
          }
        },
        plugins: {
          title: {
            display: true,
            text: this.getChartTitle(),
            color: '#eaf0ff'
          }
        }
      }
    };

    this.chart = new Chart(ctx, config);
    console.log('Chart initialized successfully');
  }

  private getChartTitle(): string {
    const model = this.modelType();
    const titles: { [key: string]: string } = {
      'cnn': 'CNN Prediction',
      'ann': 'ANN Prediction',
      'svm': 'SVM Prediction',
      'knn': 'KNN Prediction',
      'logistic_regression': 'Logistic Regression Prediction'
    };
    return titles[model] || 'Model Prediction';
  }

  private updateChart(probabilities: number[]): void {
    if (!this.chart || probabilities.length !== 10) return;

    this.chart.data.datasets[0].data = probabilities;
    if (this.chart.options.plugins?.title) {
      this.chart.options.plugins.title.text = this.getChartTitle();
    }
    this.chart.update('active');
  }
}
