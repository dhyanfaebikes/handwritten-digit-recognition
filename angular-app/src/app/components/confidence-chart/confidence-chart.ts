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
  private chart: Chart | null = null;

  constructor() {
    effect(() => {
      const probs = this.probabilities();
      if (this.chart && probs.length > 0) {
        this.updateChart(probs);
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
          backgroundColor: 'rgba(54, 162, 235, 0.6)',
          borderColor: 'rgba(54, 162, 235, 1)',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        animation: {
          duration: 500
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 1.0,
            title: {
              display: true,
              text: 'Confidence'
            }
          },
          x: {
            title: {
              display: true,
              text: 'Digit'
            }
          }
        },
        plugins: {
          title: {
            display: true,
            text: 'MNIST CNN Prediction'
          }
        }
      }
    };

    this.chart = new Chart(ctx, config);
    console.log('Chart initialized successfully');
  }

  private updateChart(probabilities: number[]): void {
    if (!this.chart || probabilities.length !== 10) return;

    this.chart.data.datasets[0].data = probabilities;
    this.chart.update('active');
  }
}
