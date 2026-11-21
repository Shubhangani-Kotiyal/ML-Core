/**
 * ML-Core Visualizer Engine
 * Interactive visualization for 6 fundamental ML algorithms built from scratch in C++.
 * Replicates the exact math and logic of include/MathUtils.h and src/*.cpp
 * Team T170: Shubhangani Kotiyal, Ishanaya Agarwal, Harshita Joshi (GEU)
 */

// ============================================================================
// Default Datasets from datasets/*.csv
// ============================================================================
const DATASETS = {
  linear_regression: [
    { x: 1, y: 2 },
    { x: 2, y: 4 },
    { x: 3, y: 5 },
    { x: 4, y: 8 },
    { x: 5, y: 10 },
    { x: 6, y: 11 }
  ],
  kmeans: [
    { x: 1.0, y: 1.0 },
    { x: 1.5, y: 2.0 },
    { x: 2.0, y: 1.5 },
    { x: 2.5, y: 2.0 },
    { x: 5.0, y: 5.0 },
    { x: 5.5, y: 6.0 },
    { x: 6.0, y: 5.5 },
    { x: 6.5, y: 6.0 }
  ],
  svm: [
    { x: -3, y: -2, label: -1 },
    { x: -2, y: -3, label: -1 },
    { x: -2, y: -1, label: -1 },
    { x: -1, y: -2, label: -1 },
    { x: 2, y: 1, label: 1 },
    { x: 2, y: 3, label: 1 },
    { x: 3, y: 2, label: 1 },
    { x: 1, y: 2, label: 1 }
  ],
  hierarchical: [
    { x: 1.0, y: 1.0, id: 1 },
    { x: 1.2, y: 1.1, id: 2 },
    { x: 0.8, y: 1.3, id: 3 },
    { x: 5.0, y: 5.0, id: 4 },
    { x: 5.2, y: 4.9, id: 5 },
    { x: 4.8, y: 5.1, id: 6 }
  ],
  pca: [
    { x: 2.5, y: 2.4 },
    { x: 0.5, y: 0.7 },
    { x: 2.2, y: 2.9 },
    { x: 1.9, y: 2.2 },
    { x: 3.1, y: 3.0 },
    { x: 2.3, y: 2.7 },
    { x: 2.0, y: 1.6 },
    { x: 1.0, y: 1.1 },
    { x: 1.5, y: 1.6 },
    { x: 1.1, y: 0.9 }
  ],
  knn: [
    { x: 1.0, y: 1.0, label: 0 },
    { x: 1.5, y: 2.0, label: 0 },
    { x: 2.0, y: 1.5, label: 0 },
    { x: 2.5, y: 2.0, label: 0 },
    { x: 5.0, y: 5.0, label: 1 },
    { x: 5.5, y: 6.0, label: 1 },
    { x: 6.0, y: 5.5, label: 1 },
    { x: 6.5, y: 6.0, label: 1 }
  ]
};

// ============================================================================
// MathUtils in JS (Replicating include/MathUtils.h)
// ============================================================================
const MathUtils = {
  mean(values) {
    if (values.length === 0) return 0;
    return values.reduce((sum, v) => sum + v, 0) / values.length;
  },

  variance(values) {
    if (values.length < 2) return 0;
    const m = this.mean(values);
    const sum = values.reduce((acc, v) => acc + Math.pow(v - m, 2), 0);
    return sum / (values.length - 1);
  },

  covariance(x, y) {
    if (x.length < 2) return 0;
    const mx = this.mean(x);
    const my = this.mean(y);
    let sum = 0;
    for (let i = 0; i < x.length; i++) {
      sum += (x[i] - mx) * (y[i] - my);
    }
    return sum / (x.length - 1);
  },

  squaredDistance(x1, y1, x2, y2) {
    return Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2);
  },

  euclideanDistance(x1, y1, x2, y2) {
    return Math.sqrt(this.squaredDistance(x1, y1, x2, y2));
  },

  // Fit line using covariance / variance (Least Squares)
  fitLine(points) {
    const n = points.length;
    if (n < 2) return { invalid: true, error: "At least two data rows are required.", slope: NaN, intercept: NaN, mse: NaN };
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const varX = this.variance(xs);
    if (varX === 0) {
      return { invalid: true, error: "Cannot fit a line: all x values are equal.", slope: NaN, intercept: NaN, mse: NaN, x0: xs[0] };
    }
    const cov = this.covariance(xs, ys);
    const slope = cov / varX;
    const intercept = this.mean(ys) - slope * this.mean(xs);

    let sse = 0;
    for (const p of points) {
      const pred = slope * p.x + intercept;
      sse += Math.pow(p.y - pred, 2);
    }
    const mse = sse / n;

    return { slope, intercept, mse, invalid: false };
  },

  // Train Linear SVM (Hinge loss + L2 regularization)
  trainSVM(points, epochs = 2000, lr = 0.001, lambda = 0.01) {
    let w = [0, 0];
    let b = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (const p of points) {
        const score = w[0] * p.x + w[1] * p.y + b;
        const insideMargin = p.label * score < 1;

        for (let j = 0; j < 2; j++) {
          let grad = lambda * w[j];
          const feat = j === 0 ? p.x : p.y;
          if (insideMargin) grad -= p.label * feat;
          w[j] -= lr * grad;
        }

        if (insideMargin) {
          b += lr * p.label;
        }
      }
    }

    return { weights: w, bias: b };
  },

  // 2D PCA decomposition
  fitPCA2D(points) {
    const n = points.length;
    if (n < 2) return null;
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const meanX = this.mean(xs);
    const meanY = this.mean(ys);

    const varX = this.variance(xs);
    const varY = this.variance(ys);
    const covXY = this.covariance(xs, ys);

    const totalVariance = varX + varY;
    const halfTrace = totalVariance / 2;
    const halfDiff = (varX - varY) / 2;
    const radius = Math.hypot(halfDiff, covXY);

    const lambda1 = halfTrace + radius;
    const lambda2 = halfTrace - radius;

    const angle = 0.5 * Math.atan2(2 * covXY, varX - varY);
    const component = [Math.cos(angle), Math.sin(angle)];
    const explainedRatio = totalVariance > 0 ? lambda1 / totalVariance : 0;

    return {
      center: [meanX, meanY],
      component,
      orthogonal: [-Math.sin(angle), Math.cos(angle)],
      eigenvalues: [lambda1, lambda2],
      explainedRatio,
      covMatrix: [[varX, covXY], [covXY, varY]]
    };
  }
};

// ============================================================================
// Application State
// ============================================================================
const App = {
  currentAlgo: 'linear_regression',
  points: [],
  isAddingPoints: true,
  isCustomDataset: false,
  customDatasetName: '',
  hoverPoint: null,
  dragTarget: null,

  // Algorithm specific state
  regression: {
    queryX: 7.0
  },
  kmeans: {
    k: 2,
    step: 0,
    centroids: [],
    history: [],
    isPlaying: false,
    playTimer: null
  },
  svm: {
    queryPoint: { x: 2.0, y: 2.0 },
    model: null
  },
  hierarchical: {
    targetK: 2,
    merges: [],
    stepIndex: 0
  },
  pca: {
    projected: false,
    animProgress: 0
  },
  knn: {
    k: 3,
    queryPoint: { x: 2.0, y: 2.0 }
  },

  // Viewport / Coordinate bounding box
  bounds: {
    minX: 0, maxX: 10,
    minY: 0, maxY: 15
  }
};

// ============================================================================
// Canvas Manager
// ============================================================================
const Canvas = {
  el: null,
  ctx: null,
  width: 0,
  height: 0,
  dpr: 1,
  padding: { top: 40, right: 40, bottom: 50, left: 55 },

  init() {
    this.el = document.getElementById('mainCanvas');
    this.ctx = this.el.getContext('2d');
    this.handleResize();
    window.addEventListener('resize', () => this.handleResize());
    this.bindEvents();
  },

  handleResize() {
    const rect = this.el.parentElement.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.width = rect.width;
    this.height = rect.height;

    this.el.width = this.width * this.dpr;
    this.el.height = this.height * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);

    this.draw();
  },

  dataToCanvas(x, y) {
    const { minX, maxX, minY, maxY } = App.bounds;
    const plotW = this.width - this.padding.left - this.padding.right;
    const plotH = this.height - this.padding.top - this.padding.bottom;

    const px = this.padding.left + ((x - minX) / (maxX - minX)) * plotW;
    const py = this.height - this.padding.bottom - ((y - minY) / (maxY - minY)) * plotH;
    return { px, py };
  },

  canvasToData(px, py) {
    const { minX, maxX, minY, maxY } = App.bounds;
    const plotW = this.width - this.padding.left - this.padding.right;
    const plotH = this.height - this.padding.top - this.padding.bottom;

    const x = minX + ((px - this.padding.left) / plotW) * (maxX - minX);
    const y = minY + ((this.height - this.padding.bottom - py) / plotH) * (maxY - minY);
    return { x, y };
  },

  bindEvents() {
    this.el.addEventListener('mousedown', (e) => this.onMouseDown(e));
    this.el.addEventListener('mousemove', (e) => this.onMouseMove(e));
    window.addEventListener('mouseup', () => this.onMouseUp());
  },

  getMousePos(e) {
    const rect = this.el.getBoundingClientRect();
    return {
      px: e.clientX - rect.left,
      py: e.clientY - rect.top
    };
  },

  onMouseDown(e) {
    const { px, py } = this.getMousePos(e);
    const data = this.canvasToData(px, py);

    // Check if dragging query point for SVM / KNN
    if (App.currentAlgo === 'svm' || App.currentAlgo === 'knn') {
      const q = App.currentAlgo === 'svm' ? App.svm.queryPoint : App.knn.queryPoint;
      const qPos = this.dataToCanvas(q.x, q.y);
      if (Math.hypot(px - qPos.px, py - qPos.py) < 18) {
        App.dragTarget = 'query';
        return;
      }
    }

    // Check if dragging query for Linear Regression
    if (App.currentAlgo === 'linear_regression') {
      const qPos = this.dataToCanvas(App.regression.queryX, 0);
      if (Math.abs(px - qPos.px) < 16) {
        App.dragTarget = 'reg_x';
        return;
      }
    }

    // If click to add points is active
    if (App.isAddingPoints) {
      if (App.currentAlgo === 'svm') {
        const label = e.shiftKey ? -1 : 1;
        App.points.push({ x: +data.x.toFixed(2), y: +data.y.toFixed(2), label });
        recalculateAndRender();
      } else if (App.currentAlgo === 'knn') {
        const label = e.shiftKey ? 1 : 0;
        App.points.push({ x: +data.x.toFixed(2), y: +data.y.toFixed(2), label });
        recalculateAndRender();
      } else if (App.currentAlgo === 'hierarchical') {
        App.points.push({ x: +data.x.toFixed(2), y: +data.y.toFixed(2), id: App.points.length + 1 });
        initHierarchical();
        recalculateAndRender();
      } else {
        App.points.push({ x: +data.x.toFixed(2), y: +data.y.toFixed(2) });
        if (App.currentAlgo === 'kmeans') initKMeans();
        recalculateAndRender();
      }
    }
  },

  onMouseMove(e) {
    const { px, py } = this.getMousePos(e);
    const data = this.canvasToData(px, py);

    // Update overlay coords
    const overlay = document.getElementById('canvasOverlay');
    overlay.textContent = `X: ${data.x.toFixed(2)}, Y: ${data.y.toFixed(2)}`;

    if (App.dragTarget === 'query') {
      if (App.currentAlgo === 'svm') {
        App.svm.queryPoint = { x: +data.x.toFixed(2), y: +data.y.toFixed(2) };
      } else if (App.currentAlgo === 'knn') {
        App.knn.queryPoint = { x: +data.x.toFixed(2), y: +data.y.toFixed(2) };
      }
      recalculateAndRender();
      return;
    }

    if (App.dragTarget === 'reg_x') {
      App.regression.queryX = Math.max(0, Math.min(10, +data.x.toFixed(2)));
      const slider = document.getElementById('regQueryXSlider');
      if (slider) slider.value = App.regression.queryX;
      recalculateAndRender();
      return;
    }
  },

  onMouseUp() {
    App.dragTarget = null;
  },

  // Master Draw method
  draw() {
    if (!this.ctx) return;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    this.drawGrid();

    switch (App.currentAlgo) {
      case 'linear_regression':
        drawLinearRegression(ctx);
        break;
      case 'kmeans':
        drawKMeans(ctx);
        break;
      case 'svm':
        drawSVM(ctx);
        break;
      case 'hierarchical':
        drawHierarchical(ctx);
        break;
      case 'pca':
        drawPCA(ctx);
        break;
      case 'knn':
        drawKNN(ctx);
        break;
    }
  },

  drawGrid() {
    const ctx = this.ctx;
    const { minX, maxX, minY, maxY } = App.bounds;

    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#64748b';
    ctx.font = '10px "JetBrains Mono", monospace';

    // X Axis Ticks & Lines
    const stepX = (maxX - minX) <= 10 ? 1 : 2;
    for (let x = Math.ceil(minX); x <= maxX; x += stepX) {
      const { px, py } = this.dataToCanvas(x, minY);
      const topPos = this.dataToCanvas(x, maxY);

      ctx.beginPath();
      ctx.moveTo(px, this.height - this.padding.bottom);
      ctx.lineTo(px, this.padding.top);
      ctx.stroke();

      ctx.textAlign = 'center';
      ctx.fillText(x.toString(), px, this.height - this.padding.bottom + 18);
    }

    // Y Axis Ticks & Lines
    const stepY = (maxY - minY) <= 10 ? 1 : 2;
    for (let y = Math.ceil(minY); y <= maxY; y += stepY) {
      const { px, py } = this.dataToCanvas(minX, y);

      ctx.beginPath();
      ctx.moveTo(this.padding.left, py);
      ctx.lineTo(this.width - this.padding.right, py);
      ctx.stroke();

      ctx.textAlign = 'right';
      ctx.fillText(y.toString(), this.padding.left - 10, py + 3);
    }

    // Main Axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1.5;

    // Zero axes if in range
    if (minX <= 0 && maxX >= 0) {
      const zeroX = this.dataToCanvas(0, 0).px;
      ctx.beginPath();
      ctx.moveTo(zeroX, this.padding.top);
      ctx.lineTo(zeroX, this.height - this.padding.bottom);
      ctx.stroke();
    }
    if (minY <= 0 && maxY >= 0) {
      const zeroY = this.dataToCanvas(0, 0).py;
      ctx.beginPath();
      ctx.moveTo(this.padding.left, zeroY);
      ctx.lineTo(this.width - this.padding.right, zeroY);
      ctx.stroke();
    }

    ctx.restore();
  }
};

// ============================================================================
// 1. Linear Regression Visualizer
// ============================================================================
function drawLinearRegression(ctx) {
  const model = MathUtils.fitLine(App.points);
  const { minX, maxX } = App.bounds;

  if (model.invalid) {
    // 1. Draw Data Points in Red
    for (const p of App.points) {
      const { px, py } = Canvas.dataToCanvas(p.x, p.y);
      ctx.save();
      ctx.fillStyle = '#f43f5e';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    // 2. Draw vertical dashed alert line through the common X value
    const x0 = App.points.length > 0 ? App.points[0].x : 0;
    const pTop = Canvas.dataToCanvas(x0, App.bounds.maxY);
    const pBottom = Canvas.dataToCanvas(x0, App.bounds.minY);

    ctx.save();
    ctx.setLineDash([6, 6]);
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = 'rgba(244, 63, 94, 0.6)';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.moveTo(pTop.px, pTop.py);
    ctx.lineTo(pBottom.px, pBottom.py);
    ctx.stroke();
    ctx.restore();

    // 3. Draw Centered Error Banner
    ctx.save();
    const cx = Canvas.width / 2;
    const cy = Canvas.height / 2;
    const bw = 390, bh = 60;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.94)';
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 1.5;
    ctx.fillRect(cx - bw / 2, cy - bh / 2, bw, bh);
    ctx.strokeRect(cx - bw / 2, cy - bh / 2, bw, bh);

    ctx.fillStyle = '#f43f5e';
    ctx.font = 'bold 12px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('⚠️ Cannot fit a line: all X values are equal.', cx, cy - 8);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px -apple-system, sans-serif';
    ctx.fillText(`Variance in X = 0 (vertical line at x = ${x0}). Add points with different X.`, cx, cy + 14);
    ctx.restore();
    return;
  }

  // 1. Draw Residual Error Segments (MSE lines)
  for (const p of App.points) {
    const predY = model.slope * p.x + model.intercept;
    const p1 = Canvas.dataToCanvas(p.x, p.y);
    const p2 = Canvas.dataToCanvas(p.x, predY);

    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = p.y >= predY ? 'rgba(52, 211, 153, 0.6)' : 'rgba(244, 63, 94, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(p1.px, p1.py);
    ctx.lineTo(p2.px, p2.py);
    ctx.stroke();
    ctx.restore();
  }

  // 2. Draw Regression Line
  const yStart = model.slope * minX + model.intercept;
  const yEnd = model.slope * maxX + model.intercept;
  const pStart = Canvas.dataToCanvas(minX, yStart);
  const pEnd = Canvas.dataToCanvas(maxX, yEnd);

  ctx.save();
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 3;
  ctx.shadowColor = 'rgba(56, 189, 248, 0.6)';
  ctx.shadowBlur = 12;
  ctx.beginPath();
  ctx.moveTo(pStart.px, pStart.py);
  ctx.lineTo(pEnd.px, pEnd.py);
  ctx.stroke();
  ctx.restore();

  // 3. Draw Data Points
  for (const p of App.points) {
    const { px, py } = Canvas.dataToCanvas(p.x, p.y);
    ctx.save();
    ctx.fillStyle = '#f8fafc';
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  // 4. Draw Query X Point & Prediction
  const qX = App.regression.queryX;
  const qY = model.slope * qX + model.intercept;
  const qPos = Canvas.dataToCanvas(qX, qY);
  const qXPos = Canvas.dataToCanvas(qX, App.bounds.minY);

  // Vertical probe line
  ctx.save();
  ctx.strokeStyle = 'rgba(251, 191, 36, 0.7)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(qXPos.px, qXPos.py);
  ctx.lineTo(qPos.px, qPos.py);
  ctx.stroke();

  // Predicted point on line
  ctx.fillStyle = '#fbbf24';
  ctx.shadowColor = '#fbbf24';
  ctx.shadowBlur = 15;
  ctx.beginPath();
  ctx.arc(qPos.px, qPos.py, 8, 0, Math.PI * 2);
  ctx.fill();

  // Callout badge
  ctx.fillStyle = '#0f172a';
  ctx.strokeStyle = '#fbbf24';
  ctx.lineWidth = 1;
  ctx.fillRect(qPos.px + 12, qPos.py - 24, 110, 24);
  ctx.strokeRect(qPos.px + 12, qPos.py - 24, 110, 24);

  ctx.fillStyle = '#f8fafc';
  ctx.font = 'bold 11px "JetBrains Mono", monospace';
  ctx.fillText(`x=${qX} ➔ ŷ=${qY.toFixed(2)}`, qPos.px + 18, qPos.py - 8);
  ctx.restore();
}

// ============================================================================
// 2. K-Means Visualizer (Lloyd's Algorithm)
// ============================================================================
const CLUSTER_COLORS = [
  '#34d399', '#38bdf8', '#fbbf24', '#f43f5e', '#a855f7'
];

function initKMeans() {
  const k = App.kmeans.k;
  App.kmeans.step = 0;
  App.kmeans.history = [];
  App.kmeans.centroids = [];

  // Repeated rows must not create identical initial centroids (matching C++ src/kmeans.cpp lines 28-45)
  const distinctPoints = [];
  for (const p of App.points) {
    if (!distinctPoints.some(d => d.x === p.x && d.y === p.y)) {
      distinctPoints.push({ x: p.x, y: p.y });
    }
  }

  if (distinctPoints.length < k && App.points.length > 0) {
    showToast(`Warning: K (${k}) exceeds distinct points (${distinctPoints.length})!`);
  }

  for (let i = 0; i < k; i++) {
    if (i < distinctPoints.length) {
      App.kmeans.centroids.push({ x: distinctPoints[i].x, y: distinctPoints[i].y });
    } else if (i < App.points.length) {
      App.kmeans.centroids.push({ x: App.points[i].x, y: App.points[i].y });
    } else {
      App.kmeans.centroids.push({ x: 2 + i, y: 2 + i });
    }
  }

  // Pre-calculate initial cluster assignments
  assignKMeansClusters();
}

function assignKMeansClusters() {
  for (const p of App.points) {
    let minDist = Infinity;
    let bestC = 0;
    App.kmeans.centroids.forEach((c, idx) => {
      const d = MathUtils.squaredDistance(p.x, p.y, c.x, c.y);
      if (d < minDist) {
        minDist = d;
        bestC = idx;
      }
    });
    p.cluster = bestC;
  }
}

function stepKMeansNext() {
  const k = App.kmeans.k;
  const isAssignStep = (App.kmeans.step % 2 === 0);

  if (isAssignStep) {
    // Centroid update step
    let changed = false;
    for (let c = 0; c < k; c++) {
      const pts = App.points.filter(p => p.cluster === c);
      if (pts.length > 0) {
        const newX = MathUtils.mean(pts.map(p => p.x));
        const newY = MathUtils.mean(pts.map(p => p.y));
        if (Math.abs(App.kmeans.centroids[c].x - newX) > 0.0001 ||
            Math.abs(App.kmeans.centroids[c].y - newY) > 0.0001) {
          changed = true;
        }
        App.kmeans.centroids[c] = { x: newX, y: newY };
      }
    }
    App.kmeans.step++;
    updatePlaybackStatus(`Step ${App.kmeans.step}: Centroids updated to cluster centers`);
    if (!changed && App.kmeans.step > 1) {
      updatePlaybackStatus(`Converged! Centroids remained stable.`);
      stopAutoPlay();
    }
  } else {
    // Assignment step
    assignKMeansClusters();

    // Rebalance empty clusters (matching C++ src/kmeans.cpp lines 71-96)
    const counts = new Array(k).fill(0);
    App.points.forEach(p => {
      if (p.cluster !== undefined && p.cluster < k) counts[p.cluster]++;
    });

    let filledEmpty = false;
    for (let c = 0; c < k; c++) {
      if (counts[c] !== 0) continue;
      let candidate = -1;
      let farthest = -1;
      for (let i = 0; i < App.points.length; i++) {
        const donor = App.points[i].cluster;
        if (counts[donor] <= 1) continue;
        const d = MathUtils.squaredDistance(
          App.points[i].x, App.points[i].y,
          App.kmeans.centroids[donor].x, App.kmeans.centroids[donor].y
        );
        if (d > farthest) {
          farthest = d;
          candidate = i;
        }
      }
      if (candidate !== -1) {
        counts[App.points[candidate].cluster]--;
        App.points[candidate].cluster = c;
        counts[c]++;
        filledEmpty = true;
      }
    }

    App.kmeans.step++;
    if (filledEmpty) {
      updatePlaybackStatus(`Step ${App.kmeans.step}: Points assigned & empty cluster filled from farthest donor point`);
    } else {
      updatePlaybackStatus(`Step ${App.kmeans.step}: Points assigned to nearest centroids`);
    }
  }

  recalculateAndRender();
}

function drawKMeans(ctx) {
  const k = App.kmeans.k;

  // 1. Draw Voronoi boundary or lines connecting points to centroids
  for (const p of App.points) {
    if (p.cluster !== undefined && App.kmeans.centroids[p.cluster]) {
      const c = App.kmeans.centroids[p.cluster];
      const p1 = Canvas.dataToCanvas(p.x, p.y);
      const p2 = Canvas.dataToCanvas(c.x, c.y);

      ctx.save();
      ctx.strokeStyle = `${CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length]}33`;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(p1.px, p1.py);
      ctx.lineTo(p2.px, p2.py);
      ctx.stroke();
      ctx.restore();
    }
  }

  // 2. Draw Points colored by cluster
  for (const p of App.points) {
    const { px, py } = Canvas.dataToCanvas(p.x, p.y);
    const color = p.cluster !== undefined ? CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length] : '#94a3b8';

    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  // 3. Draw Centroid Markers (Crosshair / Star)
  App.kmeans.centroids.forEach((c, idx) => {
    const { px, py } = Canvas.dataToCanvas(c.x, c.y);
    const color = CLUSTER_COLORS[idx % CLUSTER_COLORS.length];

    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.shadowColor = color;
    ctx.shadowBlur = 18;

    // Diamond / Star shape
    ctx.beginPath();
    ctx.arc(px, py, 11, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Cross inside centroid
    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(px - 5, py);
    ctx.lineTo(px + 5, py);
    ctx.moveTo(px, py - 5);
    ctx.lineTo(px, py + 5);
    ctx.stroke();

    // Label
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 11px "JetBrains Mono", monospace';
    ctx.fillText(`C${idx + 1} (${c.x.toFixed(2)}, ${c.y.toFixed(2)})`, px + 15, py - 8);
    ctx.restore();
  });
}

// ============================================================================
// 3. Linear Support Vector Machine (SVM) Visualizer
// ============================================================================
function drawSVM(ctx) {
  const model = MathUtils.trainSVM(App.points);
  App.svm.model = model;
  const { minX, maxX, minY, maxY } = App.bounds;
  const w = model.weights;
  const b = model.bias;

  // Hyperplane: w0 * x + w1 * y + b = 0 => y = -(w0 * x + b) / w1
  if (Math.abs(w[1]) > 0.0001) {
    const getY = (x, offset) => -(w[0] * x + b - offset) / w[1];

    // Margin corridor polygon
    const p1 = Canvas.dataToCanvas(minX, getY(minX, 1));
    const p2 = Canvas.dataToCanvas(maxX, getY(maxX, 1));
    const p3 = Canvas.dataToCanvas(maxX, getY(maxX, -1));
    const p4 = Canvas.dataToCanvas(minX, getY(minX, -1));

    ctx.save();
    ctx.fillStyle = 'rgba(168, 85, 247, 0.08)';
    ctx.beginPath();
    ctx.moveTo(p1.px, p1.py);
    ctx.lineTo(p2.px, p2.py);
    ctx.lineTo(p3.px, p3.py);
    ctx.lineTo(p4.px, p4.py);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    // Positive Margin (w*x + b = +1)
    ctx.save();
    ctx.strokeStyle = 'rgba(52, 211, 153, 0.7)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(p1.px, p1.py);
    ctx.lineTo(p2.px, p2.py);
    ctx.stroke();

    // Negative Margin (w*x + b = -1)
    ctx.strokeStyle = 'rgba(244, 63, 94, 0.7)';
    ctx.beginPath();
    ctx.moveTo(p4.px, p4.py);
    ctx.lineTo(p3.px, p3.py);
    ctx.stroke();

    // Central Decision Boundary Hyperplane (w*x + b = 0)
    const h1 = Canvas.dataToCanvas(minX, getY(minX, 0));
    const h2 = Canvas.dataToCanvas(maxX, getY(maxX, 0));
    ctx.setLineDash([]);
    ctx.strokeStyle = '#a855f7';
    ctx.lineWidth = 3;
    ctx.shadowColor = '#a855f7';
    ctx.shadowBlur = 14;
    ctx.beginPath();
    ctx.moveTo(h1.px, h1.py);
    ctx.lineTo(h2.px, h2.py);
    ctx.stroke();
    ctx.restore();
  }

  // Draw Training Points with Support Vector Halos
  for (const p of App.points) {
    const { px, py } = Canvas.dataToCanvas(p.x, p.y);
    const score = w[0] * p.x + w[1] * p.y + b;
    const isSV = Math.abs(score) <= 1.05; // Point near or on margin

    ctx.save();
    if (isSV) {
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.8)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px, py, 11, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.fillStyle = p.label === 1 ? '#34d399' : '#f43f5e';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  // Draw Draggable Query Point
  const q = App.svm.queryPoint;
  const qScore = w[0] * q.x + w[1] * q.y + b;
  const qPred = qScore >= 0 ? 1 : -1;
  const { px: qpx, py: qpy } = Canvas.dataToCanvas(q.x, q.y);

  ctx.save();
  ctx.fillStyle = qPred === 1 ? '#34d399' : '#f43f5e';
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 3;
  ctx.shadowColor = qPred === 1 ? '#34d399' : '#f43f5e';
  ctx.shadowBlur = 15;
  ctx.beginPath();
  ctx.arc(qpx, qpy, 9, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Callout
  ctx.fillStyle = '#0f172a';
  ctx.strokeStyle = '#a855f7';
  ctx.lineWidth = 1;
  ctx.fillRect(qpx + 14, qpy - 26, 120, 26);
  ctx.strokeRect(qpx + 14, qpy - 26, 120, 26);

  ctx.fillStyle = '#f8fafc';
  ctx.font = 'bold 11px "JetBrains Mono", monospace';
  ctx.fillText(`Class: ${qPred > 0 ? '+1' : '-1'} (s=${qScore.toFixed(2)})`, qpx + 20, qpy - 8);
  ctx.restore();
}

// ============================================================================
// 4. Hierarchical Clustering Visualizer
// ============================================================================
function initHierarchical() {
  const n = App.points.length;
  if (n < 2) return;

  // Compute full agglomerative single-linkage tree
  let activeClusters = App.points.map((p, i) => ({
    id: i,
    members: [i],
    points: [{ x: p.x, y: p.y }]
  }));

  App.hierarchical.merges = [];

  while (activeClusters.length > 1) {
    let bestA = -1;
    let bestB = -1;
    let minD = Infinity;

    for (let i = 0; i < activeClusters.length; i++) {
      for (let j = i + 1; j < activeClusters.length; j++) {
        // Single linkage distance: minimum distance between any pair of members
        for (const ptA of activeClusters[i].points) {
          for (const ptB of activeClusters[j].points) {
            const d = MathUtils.euclideanDistance(ptA.x, ptA.y, ptB.x, ptB.y);
            if (d < minD) {
              minD = d;
              bestA = i;
              bestB = j;
            }
          }
        }
      }
    }

    const merged = {
      members: [...activeClusters[bestA].members, ...activeClusters[bestB].members],
      points: [...activeClusters[bestA].points, ...activeClusters[bestB].points],
      dist: minD,
      mergedPair: [activeClusters[bestA].members, activeClusters[bestB].members]
    };

    App.hierarchical.merges.push(merged);

    // Replace bestA with merged, remove bestB
    activeClusters[bestA] = merged;
    activeClusters.splice(bestB, 1);
  }

  cutHierarchicalTree();
}

function cutHierarchicalTree() {
  const targetK = App.hierarchical.targetK;
  const n = App.points.length;
  const mergesToTake = Math.max(0, n - targetK);

  // Group membership
  const parent = Array.from({ length: n }, (_, i) => i);
  function find(i) {
    if (parent[i] === i) return i;
    return parent[i] = find(parent[i]);
  }
  function union(i, j) {
    const rootI = find(i);
    const rootJ = find(j);
    if (rootI !== rootJ) parent[rootI] = rootJ;
  }

  for (let i = 0; i < mergesToTake && i < App.hierarchical.merges.length; i++) {
    const m = App.hierarchical.merges[i];
    union(m.members[0], m.members[m.members.length - 1]);
  }

  // Assign cluster IDs
  const roots = {};
  let clusterCounter = 0;
  App.points.forEach((p, idx) => {
    const root = find(idx);
    if (roots[root] === undefined) {
      roots[root] = clusterCounter++;
    }
    p.cluster = roots[root];
  });
}

function drawHierarchical(ctx) {
  // 1. Draw cluster convex bubble hulls
  const targetK = App.hierarchical.targetK;
  for (let c = 0; c < targetK; c++) {
    const pts = App.points.filter(p => p.cluster === c);
    if (pts.length > 1) {
      ctx.save();
      ctx.fillStyle = `${CLUSTER_COLORS[c % CLUSTER_COLORS.length]}18`;
      ctx.strokeStyle = `${CLUSTER_COLORS[c % CLUSTER_COLORS.length]}66`;
      ctx.lineWidth = 1.5;

      const meanX = MathUtils.mean(pts.map(p => p.x));
      const meanY = MathUtils.mean(pts.map(p => p.y));
      const { px: mpx, py: mpy } = Canvas.dataToCanvas(meanX, meanY);

      let maxR = 0;
      for (const pt of pts) {
        const { px, py } = Canvas.dataToCanvas(pt.x, pt.y);
        const r = Math.hypot(px - mpx, py - mpy);
        if (r > maxR) maxR = r;
      }

      ctx.beginPath();
      ctx.arc(mpx, mpy, maxR + 18, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }
  }

  // 2. Draw Points
  App.points.forEach((p, idx) => {
    const { px, py } = Canvas.dataToCanvas(p.x, p.y);
    const color = CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length] || '#fbbf24';

    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px, py, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 10px "JetBrains Mono", monospace';
    ctx.fillText(`P${idx + 1}`, px + 10, py - 6);
    ctx.restore();
  });
}

// ============================================================================
// 5. Principal Component Analysis (PCA) Visualizer
// ============================================================================
function drawPCA(ctx) {
  const pca = MathUtils.fitPCA2D(App.points);
  if (!pca) return;

  const [cx, cy] = pca.center;
  const [u1, u2] = pca.component;
  const { px: cpx, py: cpy } = Canvas.dataToCanvas(cx, cy);

  // 1. Draw PC1 Axis (Primary Component line)
  const len = 4.0;
  const p1 = Canvas.dataToCanvas(cx - u1 * len, cy - u2 * len);
  const p2 = Canvas.dataToCanvas(cx + u1 * len, cy + u2 * len);

  ctx.save();
  ctx.strokeStyle = '#f43f5e';
  ctx.lineWidth = 3;
  ctx.shadowColor = '#f43f5e';
  ctx.shadowBlur = 12;
  ctx.beginPath();
  ctx.moveTo(p1.px, p1.py);
  ctx.lineTo(p2.px, p2.py);
  ctx.stroke();

  // Label PC1
  ctx.fillStyle = '#f43f5e';
  ctx.font = 'bold 12px "JetBrains Mono", monospace';
  ctx.fillText('PC1 Axis (96.3% Var)', p2.px + 10, p2.py);
  ctx.restore();

  // 2. Draw Orthogonal PC2 Axis (dashed)
  const [o1, o2] = pca.orthogonal;
  const op1 = Canvas.dataToCanvas(cx - o1 * 1.5, cy - o2 * 1.5);
  const op2 = Canvas.dataToCanvas(cx + o1 * 1.5, cy + o2 * 1.5);

  ctx.save();
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(op1.px, op1.py);
  ctx.lineTo(op2.px, op2.py);
  ctx.stroke();
  ctx.restore();

  // 3. Draw Points & Projections onto PC1
  for (const p of App.points) {
    const orig = Canvas.dataToCanvas(p.x, p.y);

    // Projected point on PC1: center + dot((p - center), u) * u
    const dot = (p.x - cx) * u1 + (p.y - cy) * u2;
    const projX = cx + dot * u1;
    const projY = cy + dot * u2;
    const proj = Canvas.dataToCanvas(projX, projY);

    if (App.pca.projected) {
      // Draw projection drop line
      ctx.save();
      ctx.strokeStyle = 'rgba(244, 63, 94, 0.4)';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(orig.px, orig.py);
      ctx.lineTo(proj.px, proj.py);
      ctx.stroke();

      // Projected dot on line
      ctx.fillStyle = '#f43f5e';
      ctx.beginPath();
      ctx.arc(proj.px, proj.py, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Original Point
    ctx.save();
    ctx.fillStyle = '#f8fafc';
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(orig.px, orig.py, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  // 4. Mean Center Marker
  ctx.save();
  ctx.fillStyle = '#fbbf24';
  ctx.strokeStyle = '#0f172a';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cpx, cpy, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#fbbf24';
  ctx.font = 'bold 11px "JetBrains Mono", monospace';
  ctx.fillText(`Mean (${cx.toFixed(2)}, ${cy.toFixed(2)})`, cpx + 10, cpy + 15);
  ctx.restore();
}

// ============================================================================
// 6. K-Nearest Neighbors (KNN) Visualizer
// ============================================================================
function drawKNN(ctx) {
  const k = App.knn.k;
  const q = App.knn.queryPoint;
  const { px: qpx, py: qpy } = Canvas.dataToCanvas(q.x, q.y);

  // Compute distances to all points
  const distances = App.points.map((p, idx) => ({
    idx,
    point: p,
    dist: MathUtils.euclideanDistance(q.x, q.y, p.x, p.y)
  }));

  distances.sort((a, b) => a.dist - b.dist);
  const nearest = distances.slice(0, Math.min(k, distances.length));
  const maxKDist = nearest.length > 0 ? nearest[nearest.length - 1].dist : 0;

  // 1. Draw Neighborhood Radar Radius Circle
  if (nearest.length > 0) {
    const edgePos = Canvas.dataToCanvas(q.x + maxKDist, q.y);
    const radiusPx = Math.abs(edgePos.px - qpx);

    ctx.save();
    ctx.fillStyle = 'rgba(99, 102, 241, 0.08)';
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.arc(qpx, qpy, radiusPx, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  // 2. Draw Distance Rays to the K Nearest Neighbors
  for (const item of nearest) {
    const { px, py } = Canvas.dataToCanvas(item.point.x, item.point.y);
    ctx.save();
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(qpx, qpy);
    ctx.lineTo(px, py);
    ctx.stroke();
    ctx.restore();
  }

  // 3. Draw Training Points
  for (const p of App.points) {
    const { px, py } = Canvas.dataToCanvas(p.x, p.y);
    const isNeighbor = nearest.some(n => n.point === p);

    ctx.save();
    if (isNeighbor) {
      ctx.strokeStyle = '#6366f1';
      ctx.lineWidth = 3;
      ctx.shadowColor = '#6366f1';
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(px, py, 11, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.fillStyle = p.label === 0 ? '#38bdf8' : '#fbbf24';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  // 4. Determine Majority Vote
  const votes = { 0: 0, 1: 0 };
  nearest.forEach(n => votes[n.point.label] = (votes[n.point.label] || 0) + 1);
  const predictedClass = votes[0] >= votes[1] ? 0 : 1;

  // 5. Draw Draggable Query Marker
  ctx.save();
  ctx.fillStyle = predictedClass === 0 ? '#38bdf8' : '#fbbf24';
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 3;
  ctx.shadowColor = predictedClass === 0 ? '#38bdf8' : '#fbbf24';
  ctx.shadowBlur = 16;
  ctx.beginPath();
  ctx.arc(qpx, qpy, 9, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Callout Box
  ctx.fillStyle = '#0f172a';
  ctx.strokeStyle = '#6366f1';
  ctx.lineWidth = 1;
  ctx.fillRect(qpx + 14, qpy - 28, 140, 28);
  ctx.strokeRect(qpx + 14, qpy - 28, 140, 28);

  ctx.fillStyle = '#f8fafc';
  ctx.font = 'bold 11px "JetBrains Mono", monospace';
  ctx.fillText(`Class: ${predictedClass} (${votes[0]} vs ${votes[1]})`, qpx + 20, qpy - 9);
  ctx.restore();
}

// ============================================================================
// UI Dynamic Controls & Telemetry Updates
// ============================================================================
function updateDynamicControls() {
  const container = document.getElementById('dynamicControls');
  const algo = App.currentAlgo;

  let html = '';

  if (algo === 'linear_regression') {
    html = `
      <div class="ctrl-row">
        <div class="ctrl-header">
          <span>Query Value (x)</span>
          <span class="ctrl-val" id="valRegX">${App.regression.queryX}</span>
        </div>
        <input type="range" class="range-slider" id="regQueryXSlider" min="0" max="10" step="0.1" value="${App.regression.queryX}">
      </div>
      <div class="btn-group">
        <button class="group-btn active" id="btnPresetReg">Repo CSV (n=6)</button>
        <button class="group-btn" id="btnRandomReg">Generate Noisy Line</button>
      </div>
    `;
  } else if (algo === 'kmeans') {
    html = `
      <div class="ctrl-row">
        <div class="ctrl-header">
          <span>Number of Clusters (K)</span>
          <span class="ctrl-val" id="valKMeansK">${App.kmeans.k}</span>
        </div>
        <input type="range" class="range-slider" id="kmeansKSlider" min="2" max="4" step="1" value="${App.kmeans.k}">
      </div>
      <div class="btn-group">
        <button class="group-btn active" id="btnPresetKMeans">Repo CSV (2 Clusters)</button>
        <button class="group-btn" id="btnRandomKMeans">Synthetic 3-Clusters</button>
      </div>
    `;
  } else if (algo === 'svm') {
    html = `
      <div class="ctrl-row">
        <div class="ctrl-header">
          <span>Test Point X</span>
          <span class="ctrl-val" id="valSVMX">${App.svm.queryPoint.x}</span>
        </div>
        <input type="range" class="range-slider" id="svmXSlider" min="-4" max="4" step="0.2" value="${App.svm.queryPoint.x}">
      </div>
      <div class="ctrl-row">
        <div class="ctrl-header">
          <span>Test Point Y</span>
          <span class="ctrl-val" id="valSVMY">${App.svm.queryPoint.y}</span>
        </div>
        <input type="range" class="range-slider" id="svmYSlider" min="-4" max="4" step="0.2" value="${App.svm.queryPoint.y}">
      </div>
      <div class="btn-group">
        <button class="group-btn active" id="btnPresetSVM">Repo CSV (Linearly Separable)</button>
      </div>
    `;
  } else if (algo === 'hierarchical') {
    html = `
      <div class="ctrl-row">
        <div class="ctrl-header">
          <span>Target Clusters (Cut)</span>
          <span class="ctrl-val" id="valHierK">${App.hierarchical.targetK}</span>
        </div>
        <input type="range" class="range-slider" id="hierKSlider" min="1" max="${App.points.length}" step="1" value="${App.hierarchical.targetK}">
      </div>
      <div class="btn-group">
        <button class="group-btn active" id="btnPresetHier">Repo CSV (n=6)</button>
      </div>
    `;
  } else if (algo === 'pca') {
    html = `
      <div class="ctrl-row">
        <div class="ctrl-header">
          <span>1D Subspace Projection</span>
          <span class="ctrl-val">${App.pca.projected ? 'Active' : 'Original 2D'}</span>
        </div>
        <button class="group-btn ${App.pca.projected ? 'active' : ''}" id="btnTogglePCAProj">
          ${App.pca.projected ? 'Show Original 2D Scatter' : 'Project to 1D Principal Axis'}
        </button>
      </div>
      <div class="btn-group">
        <button class="group-btn active" id="btnPresetPCA">Repo CSV (Shlens 10-Point)</button>
      </div>
    `;
  } else if (algo === 'knn') {
    html = `
      <div class="ctrl-row">
        <div class="ctrl-header">
          <span>Neighbors (K)</span>
          <span class="ctrl-val" id="valKNNK">${App.knn.k}</span>
        </div>
        <input type="range" class="range-slider" id="knnKSlider" min="1" max="7" step="2" value="${App.knn.k}">
      </div>
      <div class="btn-group">
        <button class="group-btn active" id="btnPresetKNN">Repo CSV (2 Classes)</button>
      </div>
    `;
  }

  container.innerHTML = html;
  bindDynamicEvents();
}

function bindDynamicEvents() {
  const algo = App.currentAlgo;

  if (algo === 'linear_regression') {
    const slider = document.getElementById('regQueryXSlider');
    if (slider) {
      slider.addEventListener('input', (e) => {
        App.regression.queryX = parseFloat(e.target.value);
        document.getElementById('valRegX').textContent = App.regression.queryX;
        recalculateAndRender();
      });
    }
  } else if (algo === 'kmeans') {
    const slider = document.getElementById('kmeansKSlider');
    if (slider) {
      slider.addEventListener('input', (e) => {
        App.kmeans.k = parseInt(e.target.value);
        document.getElementById('valKMeansK').textContent = App.kmeans.k;
        initKMeans();
        recalculateAndRender();
      });
    }
  } else if (algo === 'svm') {
    const sx = document.getElementById('svmXSlider');
    const sy = document.getElementById('svmYSlider');
    if (sx && sy) {
      sx.addEventListener('input', (e) => {
        App.svm.queryPoint.x = parseFloat(e.target.value);
        document.getElementById('valSVMX').textContent = App.svm.queryPoint.x;
        recalculateAndRender();
      });
      sy.addEventListener('input', (e) => {
        App.svm.queryPoint.y = parseFloat(e.target.value);
        document.getElementById('valSVMY').textContent = App.svm.queryPoint.y;
        recalculateAndRender();
      });
    }
  } else if (algo === 'hierarchical') {
    const slider = document.getElementById('hierKSlider');
    if (slider) {
      slider.addEventListener('input', (e) => {
        App.hierarchical.targetK = parseInt(e.target.value);
        document.getElementById('valHierK').textContent = App.hierarchical.targetK;
        cutHierarchicalTree();
        recalculateAndRender();
      });
    }
  } else if (algo === 'pca') {
    const btn = document.getElementById('btnTogglePCAProj');
    if (btn) {
      btn.addEventListener('click', () => {
        App.pca.projected = !App.pca.projected;
        updateDynamicControls();
        recalculateAndRender();
      });
    }
  } else if (algo === 'knn') {
    const slider = document.getElementById('knnKSlider');
    if (slider) {
      slider.addEventListener('input', (e) => {
        App.knn.k = parseInt(e.target.value);
        document.getElementById('valKNNK').textContent = App.knn.k;
        recalculateAndRender();
      });
    }
  }
}

function updateMetricsGrid() {
  const container = document.getElementById('metricsGrid');
  const algo = App.currentAlgo;
  let html = '';

  if (algo === 'linear_regression') {
    const model = MathUtils.fitLine(App.points);
    if (model.invalid) {
      const x0 = App.points.length > 0 ? App.points[0].x : 0;
      html = `
        <div class="metric-item">
          <span class="metric-label">Slope (m)</span>
          <span class="metric-value metric-error">Undefined (Vertical)</span>
          <span class="metric-sub">Zero variance in X</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Intercept (c)</span>
          <span class="metric-value metric-error">None</span>
          <span class="metric-sub">Vertical line at x=${x0}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Training MSE</span>
          <span class="metric-value metric-error">Undefined</span>
          <span class="metric-sub">Fit failed (var(x)=0)</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Fit Status</span>
          <span class="metric-value metric-error">Error</span>
          <span class="metric-sub">All X identical (${App.points.length} pts)</span>
        </div>
      `;
    } else {
      const predY = model.slope * App.regression.queryX + model.intercept;
      html = `
        <div class="metric-item">
          <span class="metric-label">Slope (m)</span>
          <span class="metric-value">${model.slope.toFixed(4)}</span>
          <span class="metric-sub">Rate of change</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Intercept (c)</span>
          <span class="metric-value">${model.intercept.toFixed(4)}</span>
          <span class="metric-sub">Y at X = 0</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Training MSE</span>
          <span class="metric-value">${model.mse.toFixed(6)}</span>
          <span class="metric-sub">Mean Squared Error</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Predicted ŷ</span>
          <span class="metric-value" style="color: #fbbf24;">${predY.toFixed(4)}</span>
          <span class="metric-sub">For x = ${App.regression.queryX}</span>
        </div>
      `;
    }
  } else if (algo === 'kmeans') {
    let wcss = 0;
    App.points.forEach(p => {
      if (p.cluster !== undefined && App.kmeans.centroids[p.cluster]) {
        wcss += MathUtils.squaredDistance(p.x, p.y, App.kmeans.centroids[p.cluster].x, App.kmeans.centroids[p.cluster].y);
      }
    });

    html = `
      <div class="metric-item">
        <span class="metric-label">Clusters (K)</span>
        <span class="metric-value">${App.kmeans.k}</span>
        <span class="metric-sub">Active partitions</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">Iteration Step</span>
        <span class="metric-value">${App.kmeans.step}</span>
        <span class="metric-sub">Lloyd's updates</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">WCSS (Inertia)</span>
        <span class="metric-value">${wcss.toFixed(4)}</span>
        <span class="metric-sub">Within-cluster sum sq</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">Data Points</span>
        <span class="metric-value">${App.points.length}</span>
        <span class="metric-sub">Total 2D coordinates</span>
      </div>
    `;
  } else if (algo === 'svm') {
    const model = App.svm.model || MathUtils.trainSVM(App.points);
    const w = model.weights;
    const b = model.bias;
    const norm = Math.hypot(w[0], w[1]);
    const margin = norm > 0 ? (2 / norm) : 0;

    let correct = 0;
    App.points.forEach(p => {
      const s = w[0] * p.x + w[1] * p.y + b;
      if ((s >= 0 ? 1 : -1) === p.label) correct++;
    });
    const acc = App.points.length > 0 ? (correct / App.points.length) * 100 : 0;

    html = `
      <div class="metric-item">
        <span class="metric-label">Weights [w₁, w₂]</span>
        <span class="metric-value">[${w[0].toFixed(3)}, ${w[1].toFixed(3)}]</span>
        <span class="metric-sub">Hyperplane normal</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">Bias (b)</span>
        <span class="metric-value">${b.toFixed(4)}</span>
        <span class="metric-sub">Intercept offset</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">Margin Width</span>
        <span class="metric-value">${margin.toFixed(3)}</span>
        <span class="metric-sub">2 / ||w||</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">Train Accuracy</span>
        <span class="metric-value" style="color: #34d399;">${acc.toFixed(0)}%</span>
        <span class="metric-sub">${correct}/${App.points.length} Classified</span>
      </div>
    `;
  } else if (algo === 'hierarchical') {
    html = `
      <div class="metric-item">
        <span class="metric-label">Target Clusters</span>
        <span class="metric-value">${App.hierarchical.targetK}</span>
        <span class="metric-sub">Tree partition level</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">Linkage Criterion</span>
        <span class="metric-value">Single</span>
        <span class="metric-sub">Minimum distance</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">Total Points</span>
        <span class="metric-value">${App.points.length}</span>
        <span class="metric-sub">Hierarchical items</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">Merges Recorded</span>
        <span class="metric-value">${App.hierarchical.merges.length}</span>
        <span class="metric-sub">Agglomerations</span>
      </div>
    `;
  } else if (algo === 'pca') {
    const pca = MathUtils.fitPCA2D(App.points);
    if (pca) {
      const evr = (pca.explainedRatio * 100).toFixed(2);
      html = `
        <div class="metric-item">
          <span class="metric-label">Explained Variance</span>
          <span class="metric-value" style="color: #f43f5e;">${evr}%</span>
          <span class="metric-sub">Ratio: ${pca.explainedRatio.toFixed(6)}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">PC1 Eigenvector</span>
          <span class="metric-value">[${pca.component[0].toFixed(3)}, ${pca.component[1].toFixed(3)}]</span>
          <span class="metric-sub">Unit vector direction</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Mean (x̄, ȳ)</span>
          <span class="metric-value">(${pca.center[0].toFixed(2)}, ${pca.center[1].toFixed(2)})</span>
          <span class="metric-sub">Center of mass</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Eigenvalues</span>
          <span class="metric-value">${pca.eigenvalues[0].toFixed(2)}, ${pca.eigenvalues[1].toFixed(2)}</span>
          <span class="metric-sub">Principal &amp; secondary</span>
        </div>
      `;
    }
  } else if (algo === 'knn') {
    const q = App.knn.queryPoint;
    const distances = App.points.map(p => ({
      label: p.label,
      dist: MathUtils.euclideanDistance(q.x, q.y, p.x, p.y)
    })).sort((a, b) => a.dist - b.dist);

    const nearest = distances.slice(0, App.knn.k);
    const votes0 = nearest.filter(n => n.label === 0).length;
    const votes1 = nearest.filter(n => n.label === 1).length;
    const pred = votes0 >= votes1 ? 0 : 1;

    html = `
      <div class="metric-item">
        <span class="metric-label">Nearest K</span>
        <span class="metric-value">${App.knn.k}</span>
        <span class="metric-sub">Voting neighbors</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">Predicted Class</span>
        <span class="metric-value" style="color: #6366f1;">Class ${pred}</span>
        <span class="metric-sub">Majority decision</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">Vote Tally</span>
        <span class="metric-value">${votes0} vs ${votes1}</span>
        <span class="metric-sub">Class 0 vs Class 1</span>
      </div>
      <div class="metric-item">
        <span class="metric-label">Query Point</span>
        <span class="metric-value">(${q.x.toFixed(1)}, ${q.y.toFixed(1)})</span>
        <span class="metric-sub">Interactive probe</span>
      </div>
    `;
  }

  container.innerHTML = html;
}

function updateFormulaCard() {
  const box = document.getElementById('formulaBox');
  const algo = App.currentAlgo;

  if (algo === 'linear_regression') {
    box.innerHTML = `
      <div><span class="formula-highlight">Slope (m):</span> Cov(X, Y) / Var(X)</div>
      <div><span class="formula-highlight">Intercept (c):</span> ȳ - m · x̄</div>
      <div><span class="formula-highlight">Prediction:</span> ŷ = m · x + c</div>
      <div><span class="formula-highlight">MSE:</span> (1/n) · Σ (yᵢ - ŷᵢ)²</div>
    `;
  } else if (algo === 'kmeans') {
    box.innerHTML = `
      <div><span class="formula-highlight">Assignment:</span> cᵢ = argminⱼ ||xᵢ - μⱼ||²</div>
      <div><span class="formula-highlight">Centroid Update:</span> μⱼ = (1 / |Sⱼ|) · Σ xᵢ</div>
      <div><span class="formula-highlight">Convergence:</span> ||μ_new - μ_old|| &lt; ε</div>
    `;
  } else if (algo === 'svm') {
    box.innerHTML = `
      <div><span class="formula-highlight">Decision Rule:</span> f(x) = sign(w · x + b)</div>
      <div><span class="formula-highlight">Hinge Loss:</span> max(0, 1 - yᵢ(w · xᵢ + b))</div>
      <div><span class="formula-highlight">L2 Gradient:</span> ∇w = λw - yᵢxᵢ (if active)</div>
    `;
  } else if (algo === 'hierarchical') {
    box.innerHTML = `
      <div><span class="formula-highlight">Single Linkage:</span> d(A, B) = min ||a - b||</div>
      <div><span class="formula-highlight">Agglomeration:</span> Merge closest pair (A*, B*)</div>
      <div><span class="formula-highlight">Partition:</span> Stop when |Clusters| = Target K</div>
    `;
  } else if (algo === 'pca') {
    box.innerHTML = `
      <div><span class="formula-highlight">Covariance Matrix:</span> C = (1 / (n-1)) · XᵀX</div>
      <div><span class="formula-highlight">Eigen Decomposition:</span> C · v = λ · v</div>
      <div><span class="formula-highlight">Explained Variance:</span> λ₁ / (λ₁ + λ₂)</div>
    `;
  } else if (algo === 'knn') {
    box.innerHTML = `
      <div><span class="formula-highlight">Euclidean Metric:</span> d(p, q) = √Σ (pᵢ - qᵢ)²</div>
      <div><span class="formula-highlight">Neighbor Set:</span> Nₖ(x) = K smallest d(x, xᵢ)</div>
      <div><span class="formula-highlight">Majority Vote:</span> ŷ = mode({ yᵢ : i ∈ Nₖ })</div>
    `;
  }
}

function updateLegend() {
  const legend = document.getElementById('canvasLegend');
  const algo = App.currentAlgo;

  if (algo === 'linear_regression') {
    legend.innerHTML = `
      <div class="legend-item"><span class="legend-swatch" style="background:#38bdf8;"></span> Regression Line (Least Squares)</div>
      <div class="legend-item"><span class="legend-swatch" style="background:#f8fafc;"></span> Training Point (x, y)</div>
      <div class="legend-item"><span class="legend-swatch" style="background:#fbbf24;"></span> Query Point (x=7)</div>
    `;
  } else if (algo === 'kmeans') {
    legend.innerHTML = `
      <div class="legend-item"><span class="legend-swatch" style="background:#34d399;"></span> Cluster 1</div>
      <div class="legend-item"><span class="legend-swatch" style="background:#38bdf8;"></span> Cluster 2</div>
      <div class="legend-item"><span class="legend-swatch" style="background:#ffffff; border: 1px solid #000;"></span> Centroid ⨁</div>
    `;
  } else if (algo === 'svm') {
    legend.innerHTML = `
      <div class="legend-item"><span class="legend-swatch" style="background:#34d399;"></span> Class +1</div>
      <div class="legend-item"><span class="legend-swatch" style="background:#f43f5e;"></span> Class -1</div>
      <div class="legend-item"><span class="legend-swatch" style="background:#a855f7;"></span> Hyperplane w·x+b=0</div>
      <div class="legend-item"><span class="legend-swatch" style="background:rgba(168,85,247,0.3);"></span> Margin Corridor</div>
    `;
  } else if (algo === 'hierarchical') {
    legend.innerHTML = `
      <div class="legend-item"><span class="legend-swatch" style="background:#34d399;"></span> Group 1</div>
      <div class="legend-item"><span class="legend-swatch" style="background:#38bdf8;"></span> Group 2</div>
      <div class="legend-item"><span class="legend-swatch" style="background:rgba(251,191,36,0.3);"></span> Convex Bounding Hull</div>
    `;
  } else if (algo === 'pca') {
    legend.innerHTML = `
      <div class="legend-item"><span class="legend-swatch" style="background:#f43f5e;"></span> PC1 Principal Axis (96.3% Var)</div>
      <div class="legend-item"><span class="legend-swatch" style="background:#fbbf24;"></span> Mean Anchor (x̄, ȳ)</div>
    `;
  } else if (algo === 'knn') {
    legend.innerHTML = `
      <div class="legend-item"><span class="legend-swatch" style="background:#38bdf8;"></span> Class 0</div>
      <div class="legend-item"><span class="legend-swatch" style="background:#fbbf24;"></span> Class 1</div>
      <div class="legend-item"><span class="legend-swatch" style="background:#6366f1;"></span> Neighborhood Radius</div>
    `;
  }
}

// ============================================================================
// Auto-Scale Canvas Viewport Bounds for Custom Datasets
// ============================================================================
function autoScaleBounds(points) {
  if (!points || points.length === 0) return;
  const xs = points.map(p => p.x);
  const ys = points.map(p => p.y);
  let minX = Math.min(...xs);
  let maxX = Math.max(...xs);
  let minY = Math.min(...ys);
  let maxY = Math.max(...ys);

  let spanX = maxX - minX;
  let spanY = maxY - minY;
  if (spanX <= 0) spanX = 2;
  if (spanY <= 0) spanY = 2;

  const padX = spanX * 0.18;
  const padY = spanY * 0.18;

  App.bounds = {
    minX: Math.floor((minX - padX) * 10) / 10,
    maxX: Math.ceil((maxX + padX) * 10) / 10,
    minY: Math.floor((minY - padY) * 10) / 10,
    maxY: Math.ceil((maxY + padY) * 10) / 10
  };

  if (App.currentAlgo === 'linear_regression') {
    App.regression.queryX = +(minX + spanX * 0.7).toFixed(1);
    const slider = document.getElementById('regQueryXSlider');
    if (slider) {
      slider.min = Math.floor(minX);
      slider.max = Math.ceil(maxX + padX);
      slider.value = App.regression.queryX;
    }
  } else if (App.currentAlgo === 'svm' || App.currentAlgo === 'knn') {
    const qX = +(minX + spanX * 0.5).toFixed(1);
    const qY = +(minY + spanY * 0.5).toFixed(1);
    if (App.currentAlgo === 'svm') App.svm.queryPoint = { x: qX, y: qY };
    if (App.currentAlgo === 'knn') App.knn.queryPoint = { x: qX, y: qY };
  }
}

// ============================================================================
// Switch Algorithm Handler
// ============================================================================
function switchAlgorithm(algoKey, keepCustomDataset = false) {
  App.currentAlgo = algoKey;

  // Set accent color variable
  const root = document.documentElement;
  const accents = {
    linear_regression: 'var(--accent-reg)',
    kmeans: 'var(--accent-kmeans)',
    svm: 'var(--accent-svm)',
    hierarchical: 'var(--accent-hier)',
    pca: 'var(--accent-pca)',
    knn: 'var(--accent-knn)'
  };
  root.style.setProperty('--active-accent', accents[algoKey]);

  // Update nav buttons
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.algo === algoKey);
  });

  // Algorithm header title and category
  const titles = {
    linear_regression: 'Linear Regression (Least Squares)',
    kmeans: 'K-Means Clustering (Lloyd’s Iterative)',
    svm: 'Linear Support Vector Machine (SVM)',
    hierarchical: 'Hierarchical Clustering (Agglomerative)',
    pca: 'Principal Component Analysis (PCA 2D ➔ 1D)',
    knn: 'K-Nearest Neighbors (KNN Classifier)'
  };
  const categories = {
    linear_regression: 'SUPERVISED REGRESSION',
    kmeans: 'UNSUPERVISED CLUSTERING',
    svm: 'SUPERVISED CLASSIFICATION',
    hierarchical: 'UNSUPERVISED CLUSTERING',
    pca: 'DIMENSIONALITY REDUCTION',
    knn: 'SUPERVISED CLASSIFICATION'
  };

  document.getElementById('algoTitle').textContent = titles[algoKey];
  document.getElementById('algoCategoryTag').textContent = categories[algoKey];

  if (!keepCustomDataset) {
    App.isCustomDataset = false;
    const banner = document.getElementById('customDatasetBanner');
    if (banner) banner.style.display = 'none';

    // Set default bounds
    if (algoKey === 'svm') {
      App.bounds = { minX: -4, maxX: 4, minY: -4, maxY: 4 };
      App.svm.queryPoint = { x: 2.0, y: 2.0 };
    } else if (algoKey === 'pca') {
      App.bounds = { minX: 0, maxX: 4, minY: 0, maxY: 4 };
    } else if (algoKey === 'linear_regression') {
      App.bounds = { minX: 0, maxX: 8, minY: 0, maxY: 16 };
      App.regression.queryX = 7.0;
    } else {
      App.bounds = { minX: 0, maxX: 8, minY: 0, maxY: 8 };
      if (algoKey === 'knn') App.knn.queryPoint = { x: 2.0, y: 2.0 };
    }

    // Load default dataset
    loadDefaultDataset(algoKey);
  } else {
    autoScaleBounds(App.points);
  }

  // Playback bar visibility
  const pbBar = document.getElementById('playbackBar');
  if (algoKey === 'kmeans') {
    pbBar.style.display = 'flex';
    initKMeans();
    updatePlaybackStatus(`K-Means Ready (K=${App.kmeans.k})`);
  } else {
    pbBar.style.display = 'none';
    stopAutoPlay();
  }

  if (algoKey === 'hierarchical') initHierarchical();

  updateDynamicControls();
  updateMetricsGrid();
  updateFormulaCard();
  updateLegend();
  Canvas.draw();
}

function loadDefaultDataset(algoKey) {
  App.points = JSON.parse(JSON.stringify(DATASETS[algoKey]));
  App.isCustomDataset = false;
}

function recalculateAndRender() {
  updateMetricsGrid();
  Canvas.draw();
}

// ============================================================================
// Playback Controls (Auto-Play / Step)
// ============================================================================
function updatePlaybackStatus(msg) {
  const status = document.getElementById('playbackStatus');
  const desc = document.getElementById('playbackDesc');
  if (status) status.textContent = `Iteration ${App.kmeans.step}`;
  if (desc) desc.textContent = msg;
}

function stopAutoPlay() {
  if (App.kmeans.playTimer) {
    clearInterval(App.kmeans.playTimer);
    App.kmeans.playTimer = null;
  }
  App.kmeans.isPlaying = false;
  const playText = document.getElementById('playAutoText');
  if (playText) playText.textContent = 'Auto Play';
}

function toggleAutoPlay() {
  if (App.kmeans.isPlaying) {
    stopAutoPlay();
  } else {
    App.kmeans.isPlaying = true;
    const playText = document.getElementById('playAutoText');
    if (playText) playText.textContent = 'Pause';
    App.kmeans.playTimer = setInterval(() => {
      stepKMeansNext();
    }, 850);
  }
}

// ============================================================================
// Modal & Code Viewer Handlers
// ============================================================================
const C_CODE_SNIPPETS = {
  linear_regression: `// From src/linear_regression.cpp
#include "MathUtils.h"
#include "MLAlgorithm.h"

class LinearRegression : public MLAlgorithm {
private:
    double slope, intercept;

public:
    LinearRegression() : slope(0), intercept(0) {}

    bool fit() {
        return MathUtils::fitLine(x, y, n, slope, intercept);
    }

    double predict(double value) const {
        return MathUtils::predictLine(value, slope, intercept);
    }

    double calculateMSE() const {
        double predicted[100];
        for (int i = 0; i < n; i++) {
            predicted[i] = predict(x[i]);
        }
        return MathUtils::meanSquaredError(y, predicted, n);
    }

    void display() const {
        std::cout << "\\nSlope: " << slope;
        std::cout << "\\nIntercept: " << intercept;
        std::cout << "\\nTraining MSE: " << calculateMSE() << "\\n";
    }

    int execute(const char filename[]) override {
        // Enforce 100-row limit & 2 numeric columns via MLAlgorithm base class
        if (!loadCSV(filename, 2, 0)) return 1;

        if (!fit()) {
            std::cout << "Cannot fit a line: all x values are equal.\\n";
            return 1;
        }

        display();
        return 0;
    }
};`,

  kmeans: `// From src/kmeans.cpp
#include "MathUtils.h"
#include "MLAlgorithm.h"

class KMeans : public MLAlgorithm {
private:
    double centerX[10], centerY[10];
    int cluster[100];
    int k;

public:
    KMeans() : k(0) {}

    bool fit(int numberOfClusters) {
        if (numberOfClusters < 1 || numberOfClusters > 10 || numberOfClusters > n)
            return false;
        k = numberOfClusters;

        // Distinct-centroid initialization: repeated rows must not create identical initial centroids
        int selected = 0;
        for (int i = 0; i < n && selected < k; i++) {
            bool duplicate = false;
            for (int c = 0; c < selected; c++) {
                if (x[i] == centerX[c] && y[i] == centerY[c]) duplicate = true;
            }
            if (!duplicate) {
                centerX[selected] = x[i];
                centerY[selected] = y[i];
                selected++;
            }
        }
        if (selected < k) {
            std::cout << "K cannot exceed the number of distinct points.\\n";
            return false;
        }

        // Assignment and centroid update loop
        for (int iteration = 0; iteration < 100; iteration++) {
            bool changed = false;
            // 1. Assign each point to its nearest centroid
            for (int i = 0; i < n; i++) {
                int nearest = MathUtils::nearestCentroid(x[i], y[i], centerX, centerY, k);
                if (cluster[i] != nearest) { cluster[i] = nearest; changed = true; }
            }

            int counts[10] = {0};
            for (int i = 0; i < n; i++) counts[cluster[i]]++;

            // 2. Donor rebalancing: fill empty cluster without emptying its donor
            for (int c = 0; c < k; c++) {
                if (counts[c] != 0) continue;
                int candidate = -1;
                double farthest = -1;
                for (int i = 0; i < n; i++) {
                    int donor = cluster[i];
                    if (counts[donor] <= 1) continue;
                    double distance = MathUtils::squaredDistance(x[i], y[i], centerX[donor], centerY[donor]);
                    if (distance > farthest) { farthest = distance; candidate = i; }
                }
                if (candidate == -1) return false;
                counts[cluster[candidate]]--;
                cluster[candidate] = c;
                counts[c]++;
                changed = true;
            }

            if (!changed) return true;

            // 3. Recalculate centroids as cluster means
            for (int c = 0; c < k; c++) {
                double clusterX[100], clusterY[100];
                int clusterSize = 0;
                for (int i = 0; i < n; i++) {
                    if (cluster[i] == c) {
                        clusterX[clusterSize] = x[i];
                        clusterY[clusterSize] = y[i];
                        clusterSize++;
                    }
                }
                centerX[c] = MathUtils::mean(clusterX, clusterSize);
                centerY[c] = MathUtils::mean(clusterY, clusterSize);
            }
        }
        return true;
    }

    int execute(const char filename[]) override {
        if (!loadCSV(filename, 1, 0)) return 1;
        // Prompt K, fit(k), and display cluster groups...
        return 0;
    }
};`,

  svm: `// From src/svm.cpp
#include "MathUtils.h"
#include "MLAlgorithm.h"

class SVM : public MLAlgorithm {
private:
    double weights[2], bias;
    double learningRate, lambda;
    int epochs;

public:
    SVM() : bias(0), learningRate(0.001), lambda(0.01), epochs(2000) {
        weights[0] = 0; weights[1] = 0;
    }

    void fit() {
        weights[0] = 0; weights[1] = 0; bias = 0;
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < n; i++) {
                MathUtils::updateSVM(x[i], y[i], labels[i],
                                    weights, bias, learningRate, lambda);
            }
        }
    }

    int predict(double newX, double newY) const {
        double score = MathUtils::svmScore(newX, newY, weights, bias);
        return score >= 0 ? 1 : -1;
    }

    int execute(const char filename[]) override {
        // Enforces labelMode 1: exactly 3 columns (x, y, label in {-1, 1}) and both classes present
        if (!loadCSV(filename, 2, 1)) return 1;
        fit();
        // classify test coordinates and calculate training accuracy...
        return 0;
    }
};`,

  hierarchical: `// From src/hierarchical_clustering.cpp
#include "MathUtils.h"
#include "MLAlgorithm.h"

class HierarchicalClustering : public MLAlgorithm {
private:
    int members[100][100];
    int count[100];
    bool active[100];
    int clusterCount;

public:
    HierarchicalClustering() : clusterCount(0) {
        for (int i = 0; i < 100; i++) { count[i] = 0; active[i] = false; }
    }

    bool fit(int targetClusters) {
        if (targetClusters < 1 || targetClusters > n) return false;
        for (int i = 0; i < n; i++) {
            members[i][0] = i; count[i] = 1; active[i] = true;
        }
        clusterCount = n;

        // Agglomerative single-linkage merges
        while (clusterCount > targetClusters) {
            int bestA = -1, bestB = -1;
            double minDistance = 0;
            bool foundFirst = false;

            for (int i = 0; i < n; i++) {
                if (!active[i]) continue;
                for (int j = i + 1; j < n; j++) {
                    if (!active[j]) continue;
                    double d = MathUtils::singleLinkDistance(x, y, members[i], count[i], members[j], count[j]);
                    if (!foundFirst || d < minDistance) {
                        minDistance = d; bestA = i; bestB = j; foundFirst = true;
                    }
                }
            }
            if (bestA == -1) return false;
            // Merge cluster B into cluster A
            for (int i = 0; i < count[bestB]; i++) {
                members[bestA][count[bestA]++] = members[bestB][i];
            }
            active[bestB] = false;
            clusterCount--;
        }
        return true;
    }

    int execute(const char filename[]) override {
        if (!loadCSV(filename, 2, 0)) return 1;
        // Prompt targetClusters, fit, and display groups...
        return 0;
    }
};`,

  pca: `// From src/pca.cpp
#include "MathUtils.h"
#include "MLAlgorithm.h"

class PCA : public MLAlgorithm {
private:
    double center[2], component[2], explainedRatio;

public:
    PCA() : explainedRatio(0) {
        center[0] = 0; center[1] = 0;
        component[0] = 0; component[1] = 0;
    }

    bool fit() {
        return MathUtils::fitPCA2D(x, y, n, center, component, explainedRatio);
    }

    double project(double newX, double newY) const {
        return MathUtils::projectPoint2D(newX, newY, center, component);
    }

    int execute(const char filename[]) override {
        if (!loadCSV(filename, 2, 0)) return 1;
        if (!fit()) return 1;
        // Output PC1 axis, center, and explained variance ratio...
        return 0;
    }
};`,

  knn: `// From src/knn.cpp
#include "MathUtils.h"
#include "MLAlgorithm.h"

class KNN : public MLAlgorithm {
public:
    KNN() {}

    int predict(double newX, double newY, int k) const {
        double distance[100];
        int index[100];
        for (int i = 0; i < n; i++) {
            distance[i] = MathUtils::euclideanDistance(newX, newY, x[i], y[i]);
            index[i] = i;
        }
        // Partial selection sort for K nearest neighbors
        for (int i = 0; i < k; i++) {
            int minIdx = i;
            for (int j = i + 1; j < n; j++) {
                if (distance[j] < distance[minIdx]) minIdx = j;
            }
            std::swap(distance[i], distance[minIdx]);
            std::swap(index[i], index[minIdx]);
        }
        // Count votes among the K nearest neighbors and return majority class
        return MathUtils::majorityVote(labels, index, k);
    }

    int execute(const char filename[]) override {
        // Enforces labelMode 2: 3 columns with non-negative integer labels
        if (!loadCSV(filename, 1, 2)) return 1;
        // Prompt test coordinates & K, predict, and report votes...
        return 0;
    }
};`
};

const ML_ALGORITHM_CODE = `// From include/MLAlgorithm.h (Abstract Base Class)
#ifndef ML_ALGORITHM_H
#define ML_ALGORITHM_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <climits>

class MLAlgorithm {
protected:
    enum { MAX_ROWS = 100 };
    double x[MAX_ROWS], y[MAX_ROWS];
    int labels[MAX_ROWS];
    int n;

    // labelMode: 0 = two numeric columns, 1 = SVM, 2 = KNN.
    bool loadCSV(const char filename[], int minimumRows, int labelMode = 0) {
        n = 0;
        std::ifstream file(filename);
        if (!file) {
            std::cout << "Unable to open CSV: " << filename << "\\n";
            return false;
        }
        std::string line;
        int lineNumber = 0, count = 0;
        while (std::getline(file, line)) {
            ++lineNumber;
            if (line.find_first_not_of(" \\t\\r") == std::string::npos)
                continue;
            if (count == MAX_ROWS) {
                std::cout << "Maximum 100 data rows allowed. File rejected.\\n";
                return false;
            }
            std::stringstream row(line);
            double values[3] = {0, 0, 0};
            int columns = labelMode == 0 ? 2 : 3;
            for (int j = 0; j < columns; ++j) {
                if (!(row >> values[j]) || !std::isfinite(values[j])) {
                    std::cout << "Invalid or missing number on CSV line "
                              << lineNumber << ".\\n";
                    return false;
                }
                if (j < columns - 1) {
                    char comma;
                    if (!(row >> comma) || comma != ',') {
                        std::cout << "Expected comma on CSV line "
                                  << lineNumber << ".\\n";
                        return false;
                    }
                }
            }
            row >> std::ws;
            if (!row.eof()) {
                std::cout << "Extra data on CSV line " << lineNumber << ".\\n";
                return false;
            }
            if (labelMode != 0) {
                double label = values[2];
                if (std::floor(label) != label || label < INT_MIN || label > INT_MAX) {
                    std::cout << "Class labels must be integers in the int range.\\n";
                    return false;
                }
                if (labelMode == 1 && label != -1 && label != 1) {
                    std::cout << "SVM labels must be -1 or 1.\\n";
                    return false;
                }
                labels[count] = static_cast<int>(label);
            }
            x[count] = values[0];
            y[count] = values[1];
            ++count;
        }
        if (file.bad()) {
            std::cout << "Error reading CSV file.\\n";
            return false;
        }
        if (count < minimumRows) {
            std::cout << "At least " << minimumRows << " data rows required.\\n";
            return false;
        }
        if (labelMode == 1) {
            bool positive = false, negative = false;
            for (int i = 0; i < count; ++i) {
                if (labels[i] == 1) positive = true;
                if (labels[i] == -1) negative = true;
            }
            if (!positive || !negative) {
                std::cout << "SVM data must contain both classes.\\n";
                return false;
            }
        }
        n = count;
        return true;
    }

public:
    MLAlgorithm() : n(0) {}
    virtual ~MLAlgorithm() {}
    virtual int execute(const char filename[]) = 0;
};

// Factory functions implemented in each algorithm's source file
MLAlgorithm& getLinearRegression();
MLAlgorithm& getKMeans();
MLAlgorithm& getSVM();
MLAlgorithm& getHierarchicalClustering();
MLAlgorithm& getPCA();
MLAlgorithm& getKNN();

#endif`;

const MATH_UTILS_CODE = `// From include/MathUtils.h
namespace MathUtils {

inline double mean(const double values[], int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += values[i];
    return sum / n;
}

inline double squaredDistance(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return dx * dx + dy * dy;
}

inline double euclideanDistance(double x1, double y1, double x2, double y2) {
    return std::sqrt(squaredDistance(x1, y1, x2, y2));
}

inline double covariance(const double x[], const double y[], int n) {
    double meanX = mean(x, n), meanY = mean(y, n);
    double total = 0;
    for (int i = 0; i < n; i++) total += (x[i] - meanX) * (y[i] - meanY);
    return total / (n - 1);
}

inline bool fitLine(const double x[], const double y[], int n,
                    double &slope, double &intercept) {
    if (n < 2) return false;
    double varX = variance(x, n);
    if (varX == 0) return false;
    slope = covariance(x, y, n) / varX;
    intercept = mean(y, n) - slope * mean(x, n);
    return true;
}

inline void updateSVM(double x, double y, int label, double weights[],
                      double &bias, double learningRate, double lambda) {
    double features[2] = {x, y};
    bool insideMargin = label * (weights[0] * x + weights[1] * y + bias) < 1;
    for (int j = 0; j < 2; j++) {
        double grad = lambda * weights[j];
        if (insideMargin) grad -= label * features[j];
        weights[j] -= learningRate * grad;
    }
    if (insideMargin) bias += learningRate * label;
}

} // namespace MathUtils`;

function initModals() {
  const benchmarkModal = document.getElementById('benchmarkModal');
  const codeModal = document.getElementById('codeModal');

  document.getElementById('btnBenchmarkModal').addEventListener('click', () => {
    benchmarkModal.style.display = 'flex';
  });
  document.getElementById('btnCloseBenchmark').addEventListener('click', () => {
    benchmarkModal.style.display = 'none';
  });

  document.getElementById('btnCodeModal').addEventListener('click', () => {
    showCodeModal('algo');
    codeModal.style.display = 'flex';
  });
  document.getElementById('btnCloseCode').addEventListener('click', () => {
    codeModal.style.display = 'none';
  });

  const tabAlgo = document.getElementById('tabAlgoCode');
  const tabBase = document.getElementById('tabBaseCode');
  const tabMath = document.getElementById('tabMathCode');

  if (tabAlgo) {
    tabAlgo.addEventListener('click', (e) => {
      document.querySelectorAll('.code-tab').forEach(t => t.classList.remove('active'));
      e.target.classList.add('active');
      showCodeModal('algo');
    });
  }

  if (tabBase) {
    tabBase.addEventListener('click', (e) => {
      document.querySelectorAll('.code-tab').forEach(t => t.classList.remove('active'));
      e.target.classList.add('active');
      showCodeModal('base');
    });
  }

  if (tabMath) {
    tabMath.addEventListener('click', (e) => {
      document.querySelectorAll('.code-tab').forEach(t => t.classList.remove('active'));
      e.target.classList.add('active');
      showCodeModal('math');
    });
  }

  // Close modals on clicking backdrop
  window.addEventListener('click', (e) => {
    if (e.target === benchmarkModal) benchmarkModal.style.display = 'none';
    if (e.target === codeModal) codeModal.style.display = 'none';
  });
}

function showCodeModal(tab) {
  const viewer = document.getElementById('codeViewer');
  const title = document.getElementById('codeModalTitle');
  const sub = document.getElementById('codeModalSub');

  if (tab === 'algo') {
    title.textContent = `C++ Class: ${App.currentAlgo.replace('_', ' ').toUpperCase()}`;
    sub.textContent = `From src/${App.currentAlgo}.cpp (inherits from MLAlgorithm)`;
    viewer.textContent = C_CODE_SNIPPETS[App.currentAlgo] || '// C++ snippet';
  } else if (tab === 'base') {
    title.textContent = 'include/MLAlgorithm.h (Abstract Base Class)';
    sub.textContent = 'Base class providing loadCSV(), 100-row limit, column verification & virtual execute()';
    viewer.textContent = ML_ALGORITHM_CODE;
  } else {
    title.textContent = 'include/MathUtils.h';
    sub.textContent = 'Shared vector, covariance, and distance calculations';
    viewer.textContent = MATH_UTILS_CODE;
  }
}
  } else {
    title.textContent = 'include/MathUtils.h';
    sub.textContent = 'Shared vector, covariance, and distance calculations';
    viewer.textContent = MATH_UTILS_CODE;
  }
}

// ============================================================================
// Toast Popup Notification
// ============================================================================
function showToast(msg) {
  const toast = document.getElementById('toastPopup');
  const msgEl = document.getElementById('toastMessage');
  if (!toast || !msgEl) return;
  msgEl.textContent = msg;
  toast.style.display = 'flex';
  setTimeout(() => {
    toast.style.display = 'none';
  }, 3500);
}

// ============================================================================
// Custom CSV Upload & Dataset Replacement Manager
// ============================================================================
const SAMPLE_PRESETS = {
  housing: `10, 15
12, 18
15, 22
18, 28
20, 32
24, 39
28, 45
30, 50
35, 60
40, 72`,
  blobs: `2.0, 2.0
2.2, 2.5
1.8, 2.3
2.5, 1.9
6.0, 3.0
6.2, 3.4
5.8, 2.8
6.5, 3.2
4.0, 7.0
4.2, 7.3
3.8, 6.8
4.5, 7.2`,
  classification: `-2.5, -2.0, -1
-1.8, -2.5, -1
-2.0, -1.2, -1
-1.2, -1.8, -1
-0.8, -2.2, -1
2.0, 1.8, 1
2.5, 2.4, 1
1.8, 2.8, 1
2.8, 1.9, 1
3.0, 2.5, 1`,
  pca_ellipse: `1.0, 1.2
1.5, 1.8
2.0, 2.3
2.5, 3.1
3.0, 3.7
3.5, 4.4
4.0, 5.0
4.5, 5.8
2.2, 2.9
2.8, 3.4
3.2, 4.0
1.8, 2.5`
};

let uploadedData = {
  rawText: '',
  fileName: '',
  targetAlgo: 'linear_regression',
  parsedPoints: []
};

function parseCSVText(text, targetAlgo) {
  if (!text || typeof text !== 'string') {
    return { points: [], error: 'CSV file or input is empty.' };
  }

  const rawLines = text.split(/\r?\n/);
  const dataLines = [];

  for (let idx = 0; idx < rawLines.length; idx++) {
    const raw = rawLines[idx];
    // Skip empty lines (matching C++ line.find_first_not_of(" \\t\\r") == npos)
    if (raw.find ? false : /^\s*$/.test(raw)) continue;
    // Skip comment lines
    if (raw.trim().startsWith('#') || raw.trim().startsWith('//')) continue;
    dataLines.push({ lineNumber: idx + 1, text: raw.trim() });
  }

  if (dataLines.length === 0) {
    return { points: [], error: 'At least 1 data row required.' };
  }

  // Enforce 100-row maximum limit (matching C++ MAX_ROWS = 100 in include/MLAlgorithm.h)
  if (dataLines.length > 100) {
    return { points: [], error: 'Maximum 100 data rows allowed. File rejected.' };
  }

  // labelMode: 0 = two numeric columns, 1 = SVM, 2 = KNN.
  const isSVM = targetAlgo === 'svm';
  const isKNN = targetAlgo === 'knn';
  const labelMode = isSVM ? 1 : (isKNN ? 2 : 0);
  const columns = labelMode === 0 ? 2 : 3;

  // Minimum row requirements (matching C++ loadCSV(filename, minRows, labelMode))
  let minimumRows = 1;
  if (targetAlgo === 'linear_regression' || targetAlgo === 'svm' || targetAlgo === 'hierarchical' || targetAlgo === 'pca') {
    minimumRows = 2;
  }
  if (dataLines.length < minimumRows) {
    return { points: [], error: `At least ${minimumRows} data rows required.` };
  }

  const points = [];

  for (let i = 0; i < dataLines.length; i++) {
    const { lineNumber, text: line } = dataLines[i];

    // Must be comma-separated
    if (!line.includes(',')) {
      return { points: [], error: `Expected comma on CSV line ${lineNumber}.` };
    }

    const tokens = line.split(',');
    if (tokens.length < columns) {
      return { points: [], error: `Invalid or missing number on CSV line ${lineNumber}.` };
    }
    if (tokens.length > columns) {
      return { points: [], error: `Extra data on CSV line ${lineNumber}.` };
    }

    const values = [];
    for (let j = 0; j < columns; j++) {
      const trimmed = tokens[j].trim();
      if (trimmed === '') {
        return { points: [], error: `Invalid or missing number on CSV line ${lineNumber}.` };
      }
      const num = Number(trimmed);
      if (!Number.isFinite(num)) {
        return { points: [], error: `Invalid or missing number on CSV line ${lineNumber}.` };
      }
      values.push(num);
    }

    if (labelMode !== 0) {
      const label = values[2];
      if (!Number.isInteger(label) || label < -2147483648 || label > 2147483647) {
        return { points: [], error: `Class labels must be integers in the int range (line ${lineNumber}).` };
      }
      if (labelMode === 1 && label !== -1 && label !== 1) {
        return { points: [], error: `SVM labels must be -1 or 1 (line ${lineNumber}).` };
      }
      if (labelMode === 2 && label < 0) {
        return { points: [], error: `KNN labels must be non-negative integers (line ${lineNumber}).` };
      }
    }

    if (targetAlgo === 'svm') {
      points.push({ x: values[0], y: values[1], label: values[2] });
    } else if (targetAlgo === 'knn') {
      points.push({ x: values[0], y: values[1], label: values[2] });
    } else if (targetAlgo === 'hierarchical') {
      points.push({ x: values[0], y: values[1], id: points.length + 1 });
    } else {
      points.push({ x: values[0], y: values[1] });
    }
  }

  // Validate that SVM dataset contains both classes
  if (labelMode === 1) {
    let positive = false, negative = false;
    for (const p of points) {
      if (p.label === 1) positive = true;
      if (p.label === -1) negative = true;
    }
    if (!positive || !negative) {
      return { points: [], error: 'SVM data must contain both classes.' };
    }
  }

  return { points, error: null };
}

function updateUploadPreview() {
  const previewContainer = document.getElementById('dataPreviewContainer');
  const previewTitle = document.getElementById('previewTitle');
  const badge = document.getElementById('previewValidationBadge');
  const errorBox = document.getElementById('previewErrorBox');
  const tableWrap = document.getElementById('previewTableWrap');
  const table = document.getElementById('previewTable');
  const thead = table.querySelector('thead');
  const tbody = table.querySelector('tbody');
  const btnApply = document.getElementById('btnApplyDataset');

  if (!uploadedData.rawText || uploadedData.rawText.trim() === '') {
    previewContainer.style.display = 'none';
    if (errorBox) errorBox.style.display = 'none';
    btnApply.disabled = true;
    return;
  }

  previewContainer.style.display = 'block';

  const res = parseCSVText(uploadedData.rawText, uploadedData.targetAlgo);

  if (res.error) {
    uploadedData.parsedPoints = [];
    previewTitle.textContent = 'CSV Validation Failed';
    badge.textContent = 'Invalid CSV Data';
    badge.classList.add('invalid');
    if (errorBox) {
      errorBox.innerHTML = `⚠️ <strong>Validation Error:</strong> ${res.error}`;
      errorBox.style.display = 'flex';
    }
    if (tableWrap) tableWrap.style.display = 'none';
    btnApply.disabled = true;
    return;
  }

  // Valid parsed points
  const points = res.points;
  uploadedData.parsedPoints = points;

  previewTitle.textContent = `Parsed Data: ${points.length} rows ready`;
  badge.textContent = `Valid for ${uploadedData.targetAlgo.replace('_', ' ').toUpperCase()} (≤100 rows)`;
  badge.classList.remove('invalid');
  if (errorBox) errorBox.style.display = 'none';
  if (tableWrap) tableWrap.style.display = 'block';
  btnApply.disabled = false;

  // Render Table Preview
  const isLabeled = (uploadedData.targetAlgo === 'svm' || uploadedData.targetAlgo === 'knn');
  thead.innerHTML = `<tr><th>#</th><th>X Feature</th><th>Y Feature</th>${isLabeled ? '<th>Label / Class</th>' : ''}</tr>`;

  let rowsHtml = '';
  const previewRows = points.slice(0, 6);
  previewRows.forEach((p, idx) => {
    rowsHtml += `<tr>
      <td>${idx + 1}</td>
      <td><code>${p.x}</code></td>
      <td><code>${p.y}</code></td>
      ${isLabeled ? `<td><span class="tag-match">${p.label}</span></td>` : ''}
    </tr>`;
  });
  if (points.length > 6) {
    rowsHtml += `<tr><td colspan="${isLabeled ? 4 : 3}" style="text-align:center; color:#64748b;">... and ${points.length - 6} more rows</td></tr>`;
  }
  tbody.innerHTML = rowsHtml;
}

function initUploadModal() {
  const uploadModal = document.getElementById('uploadModal');
  const btnOpen = document.getElementById('btnOpenUploadModal');
  const btnCanvasUpload = document.getElementById('btnCanvasUpload');
  const btnClose = document.getElementById('btnCloseUpload');
  const btnCancel = document.getElementById('btnCancelUpload');
  const btnApply = document.getElementById('btnApplyDataset');

  const tabUpload = document.getElementById('tabUploadFile');
  const tabPaste = document.getElementById('tabPasteText');
  const tabSample = document.getElementById('tabLoadSample');

  const secUpload = document.getElementById('sectionUploadFile');
  const secPaste = document.getElementById('sectionPasteText');
  const secSample = document.getElementById('sectionLoadSample');

  const dropzone = document.getElementById('csvDropzone');
  const fileInput = document.getElementById('csvFileInput');
  const fileLoadedInfo = document.getElementById('fileLoadedInfo');
  const fileNameDisplay = document.getElementById('fileNameDisplay');
  const fileStatDisplay = document.getElementById('fileStatDisplay');
  const btnRemoveFile = document.getElementById('btnRemoveFile');

  const textInput = document.getElementById('csvTextInput');
  const btnParseText = document.getElementById('btnParseText');

  const openModal = () => {
    uploadedData.targetAlgo = App.currentAlgo;
    // Set active card in algo grid
    document.querySelectorAll('.algo-choice-card').forEach(card => {
      card.classList.toggle('active', card.dataset.algo === uploadedData.targetAlgo);
    });
    uploadModal.style.display = 'flex';
  };

  if (btnOpen) btnOpen.addEventListener('click', openModal);
  if (btnCanvasUpload) btnCanvasUpload.addEventListener('click', openModal);

  const closeModal = () => {
    uploadModal.style.display = 'none';
  };

  if (btnClose) btnClose.addEventListener('click', closeModal);
  if (btnCancel) btnCancel.addEventListener('click', closeModal);
  window.addEventListener('click', (e) => {
    if (e.target === uploadModal) closeModal();
  });

  // Switch Input Tabs
  tabUpload.addEventListener('click', () => {
    [tabUpload, tabPaste, tabSample].forEach(t => t.classList.remove('active'));
    tabUpload.classList.add('active');
    secUpload.style.display = 'block';
    secPaste.style.display = 'none';
    secSample.style.display = 'none';
  });

  tabPaste.addEventListener('click', () => {
    [tabUpload, tabPaste, tabSample].forEach(t => t.classList.remove('active'));
    tabPaste.classList.add('active');
    secUpload.style.display = 'none';
    secPaste.style.display = 'block';
    secSample.style.display = 'none';
  });

  tabSample.addEventListener('click', () => {
    [tabUpload, tabPaste, tabSample].forEach(t => t.classList.remove('active'));
    tabSample.classList.add('active');
    secUpload.style.display = 'none';
    secPaste.style.display = 'none';
    secSample.style.display = 'block';
  });

  // File Upload Handlers
  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  function handleFile(file) {
    uploadedData.fileName = file.name;
    const reader = new FileReader();
    reader.onload = (event) => {
      uploadedData.rawText = event.target.result;
      dropzone.style.display = 'none';
      fileLoadedInfo.style.display = 'flex';
      fileNameDisplay.textContent = file.name;
      fileStatDisplay.textContent = `${(file.size / 1024).toFixed(1)} KB`;
      updateUploadPreview();
    };
    reader.readAsText(file);
  }

  btnRemoveFile.addEventListener('click', () => {
    fileInput.value = '';
    uploadedData.rawText = '';
    uploadedData.fileName = '';
    fileLoadedInfo.style.display = 'none';
    dropzone.style.display = 'flex';
    updateUploadPreview();
  });

  // Paste Text Parse Button
  btnParseText.addEventListener('click', () => {
    uploadedData.rawText = textInput.value;
    uploadedData.fileName = 'Pasted Text Data';
    updateUploadPreview();
  });

  // Sample Preset Buttons
  document.querySelectorAll('.sample-card-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const sampleKey = btn.dataset.sample;
      if (SAMPLE_PRESETS[sampleKey]) {
        uploadedData.rawText = SAMPLE_PRESETS[sampleKey];
        uploadedData.fileName = `Sample: ${sampleKey}.csv`;

        // Auto-select fitting algo
        if (sampleKey === 'housing') uploadedData.targetAlgo = 'linear_regression';
        if (sampleKey === 'blobs') uploadedData.targetAlgo = 'kmeans';
        if (sampleKey === 'classification') uploadedData.targetAlgo = 'svm';
        if (sampleKey === 'pca_ellipse') uploadedData.targetAlgo = 'pca';

        document.querySelectorAll('.algo-choice-card').forEach(card => {
          card.classList.toggle('active', card.dataset.algo === uploadedData.targetAlgo);
        });

        updateUploadPreview();
        showToast(`Loaded preset "${sampleKey}" into preview!`);
      }
    });
  });

  // Algorithm Choice Selection Cards
  document.querySelectorAll('.algo-choice-card').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.algo-choice-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      uploadedData.targetAlgo = card.dataset.algo;
      updateUploadPreview();
    });
  });

  // Apply Dataset Button
  btnApply.addEventListener('click', () => {
    const res = parseCSVText(uploadedData.rawText, uploadedData.targetAlgo);
    if (res.error || !res.points || res.points.length === 0) {
      showToast(`Cannot apply dataset: ${res.error || 'Invalid CSV data'}`);
      return;
    }

    // Replace original dataset with validated points
    App.points = JSON.parse(JSON.stringify(res.points));
    App.isCustomDataset = true;
    App.customDatasetName = uploadedData.fileName || 'Custom CSV';

    // Switch to chosen algorithm and keep custom dataset
    switchAlgorithm(uploadedData.targetAlgo, true);

    // Show Custom Active Banner
    const banner = document.getElementById('customDatasetBanner');
    const bannerText = document.getElementById('customBannerText');
    if (banner && bannerText) {
      banner.style.display = 'flex';
      bannerText.textContent = `Active: ${App.customDatasetName} (${App.points.length} rows loaded) • Replaced default repository dataset`;
    }

    closeModal();
    showToast(`Custom dataset loaded! Running ${uploadedData.targetAlgo.replace('_', ' ').toUpperCase()}`);
  });

  // Banner Restore Default Button
  const btnRestoreBanner = document.getElementById('btnRestoreFromBanner');
  if (btnRestoreBanner) {
    btnRestoreBanner.addEventListener('click', () => {
      loadDefaultDataset(App.currentAlgo);
      switchAlgorithm(App.currentAlgo, false);
      showToast('Default repository CSV restored.');
    });
  }
}

// ============================================================================
// Initialization Entrypoint
// ============================================================================
window.addEventListener('DOMContentLoaded', () => {
  Canvas.init();
  initModals();
  initUploadModal();

  // Navigation tab clicks
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      switchAlgorithm(btn.dataset.algo);
    });
  });

  // Action Buttons
  document.getElementById('btnResetDataset').addEventListener('click', () => {
    loadDefaultDataset(App.currentAlgo);
    switchAlgorithm(App.currentAlgo, false);
    showToast('Default repository CSV restored.');
  });

  document.getElementById('btnClearPoints').addEventListener('click', () => {
    App.points = [];
    if (App.currentAlgo === 'kmeans') initKMeans();
    if (App.currentAlgo === 'hierarchical') initHierarchical();
    recalculateAndRender();
  });

  const btnToggleAdd = document.getElementById('btnToggleAddPoint');
  btnToggleAdd.addEventListener('click', () => {
    App.isAddingPoints = !App.isAddingPoints;
    btnToggleAdd.classList.toggle('active', App.isAddingPoints);
  });

  // Playback buttons for K-Means
  document.getElementById('btnStepNext').addEventListener('click', () => {
    stepKMeansNext();
  });
  document.getElementById('btnPlayAuto').addEventListener('click', () => {
    toggleAutoPlay();
  });

  // Start on Linear Regression
  switchAlgorithm('linear_regression');
});
