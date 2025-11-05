import { Component, ElementRef, ViewChild, AfterViewInit, output, signal } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-drawing-canvas',
  imports: [CommonModule],
  templateUrl: './drawing-canvas.html',
  styleUrl: './drawing-canvas.scss',
})
export class DrawingCanvas implements AfterViewInit {
  @ViewChild('canvas', { static: false }) canvasRef!: ElementRef<HTMLCanvasElement>;
  
  private ctx: CanvasRenderingContext2D | null = null;
  private isDrawing = false;
  
  imageDataReady = output<ImageData>();
  canvasCleared = output<void>();
  
  brushSize = signal(20);
  brushColor = signal('#ffffff');

  ngAfterViewInit(): void {
    const canvas = this.canvasRef.nativeElement;
    this.ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    if (!this.ctx) {
      return;
    }

    // Set canvas properties - EXACT same as index.html lines 107-110
    this.ctx.strokeStyle = 'white';
    this.ctx.lineWidth = 20;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    
    // NOTE: index.html does NOT fill canvas with black initially
    // The canvas background is set via CSS, but pixels are transparent
    // We don't fill here to match index.html behavior exactly

    // Mouse events
    canvas.addEventListener('mousedown', this.startDrawing.bind(this));
    canvas.addEventListener('mousemove', this.draw.bind(this));
    canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
    canvas.addEventListener('mouseout', this.stopDrawing.bind(this));
    
    // Touch events
    canvas.addEventListener('touchstart', this.handleTouch.bind(this));
    canvas.addEventListener('touchmove', this.handleTouch.bind(this));
    canvas.addEventListener('touchend', this.stopDrawing.bind(this));
  }

  private startDrawing(e: MouseEvent | TouchEvent): void {
    this.isDrawing = true;
    this.draw(e);
  }

  private draw(e: MouseEvent | TouchEvent): void {
    if (!this.isDrawing || !this.ctx || !this.canvasRef) return;
    
    const canvas = this.canvasRef.nativeElement;
    const rect = canvas.getBoundingClientRect();
    
    let x: number, y: number;
    
    if (e instanceof MouseEvent) {
      x = e.clientX - rect.left;
      y = e.clientY - rect.top;
    } else {
      e.preventDefault();
      const touch = e.touches[0];
      x = touch.clientX - rect.left;
      y = touch.clientY - rect.top;
    }

    this.ctx.lineTo(x, y);
    this.ctx.stroke();
    this.ctx.beginPath();
    this.ctx.moveTo(x, y);
  }

  private stopDrawing(): void {
    if (this.isDrawing) {
      this.isDrawing = false;
      if (this.ctx) {
        this.ctx.beginPath();
      }
    }
  }

  // EXACT COPY from index.html lines 149-158
  private handleTouch(e: TouchEvent): void {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(
      e.type === 'touchstart' ? 'mousedown' : 
      e.type === 'touchmove' ? 'mousemove' : 'mouseup',
      {
        clientX: touch.clientX,
        clientY: touch.clientY
      }
    );
    if (this.canvasRef) {
      this.canvasRef.nativeElement.dispatchEvent(mouseEvent);
    }
  }

  clear(): void {
    if (!this.ctx || !this.canvasRef) return;
    
    const canvas = this.canvasRef.nativeElement;
    // EXACT COPY from index.html line 161: just clearRect, no fill
    this.ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Reset canvas properties after clear
    this.ctx.strokeStyle = 'white';
    this.ctx.lineWidth = 20;
    this.canvasCleared.emit();
  }

  getImageData(): ImageData | null {
    if (!this.ctx || !this.canvasRef) return null;
    
    const canvas = this.canvasRef.nativeElement;
    // Get image data - EXACT same as index.html: ctx.getImageData(0, 0, canvas.width, canvas.height)
    return this.ctx.getImageData(0, 0, canvas.width, canvas.height);
  }

  updateBrushSize(size: number): void {
    this.brushSize.set(size);
    if (this.ctx) {
      this.ctx.lineWidth = size;
    }
  }

  updateBrushColor(color: string): void {
    this.brushColor.set(color);
    if (this.ctx) {
      this.ctx.strokeStyle = color;
    }
  }

  captureAndEmit(): void {
    const imageData = this.getImageData();
    if (imageData) {
      this.imageDataReady.emit(imageData);
    }
  }
}
