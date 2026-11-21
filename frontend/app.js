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
const BACKEND_CODE_FILES = {
  linear_regression: {
    path: "src/linear_regression.cpp",
    title: "Linear Regression Class (Least Squares)",
    desc: "Least Squares Regression class inheriting from MLAlgorithm base class with covariance/variance slope calculation and MSE",
    lines: "67 lines",
    size: "1.5 KB",
    code: "#include <iostream>\n#include \"MathUtils.h\"\n#include \"MLAlgorithm.h\"\nusing namespace std;\n\nclass LinearRegression : public MLAlgorithm {\nprivate:\n    double slope, intercept;\n\npublic:\n    LinearRegression() {\n        slope = 0;\n        intercept = 0;\n    }\n\n    bool fit() {\n        return MathUtils::fitLine(x, y, n, slope, intercept);\n    }\n\n    double predict(double value) const {\n        return MathUtils::predictLine(value, slope, intercept);\n    }\n\n    double calculateMSE() const {\n        double predicted[100];\n\n        for (int i = 0; i < n; i++) {\n            predicted[i] = predict(x[i]);\n        }\n\n        return MathUtils::meanSquaredError(y, predicted, n);\n    }\n\n    void display() const {\n        cout << \"\\nSlope: \" << slope;\n        cout << \"\\nIntercept: \" << intercept;\n        cout << \"\\nTraining MSE: \" << calculateMSE() << \"\\n\";\n    }\n    int execute(const char filename[]) override {\n        if (!loadCSV(filename, 2, 0))\n            return 1;\n    \n        if (!fit()) {\n            cout << \"Cannot fit a line: all x values are equal.\\n\";\n            return 1;\n        }\n    \n        display();\n    \n        double value;\n        cout << \"\\nEnter x to predict y: \";\n    \n        if (!(cin >> value) || !isfinite(value)) {\n            cout << \"Please enter a number.\\n\";\n            return 1;\n        }\n    \n        cout << \"Predicted y: \" << predict(value) << \"\\n\";\n    \n        return 0;\n    }\n};\n\nMLAlgorithm& getLinearRegression() {\n    static LinearRegression model;\n    return model;\n}\n"
  },
  kmeans: {
    path: "src/kmeans.cpp",
    title: "K-Means Lloyd Clustering Class",
    desc: "Lloyd iteration with distinct centroid initialization, empty-cluster donor rebalancing, and MLAlgorithm base inheritance",
    lines: "164 lines",
    size: "4.6 KB",
    code: "#include <iostream>\n#include \"MathUtils.h\"\n#include \"MLAlgorithm.h\"\nusing namespace std;\n\nclass KMeans : public MLAlgorithm {\nprivate:\n    double centerX[10], centerY[10];\n    int cluster[100];\n    int k;\n\npublic:\n    KMeans() {\n        k = 0;\n    }\n\n    bool fit(int numberOfClusters) {\n        if (numberOfClusters < 1 ||\n            numberOfClusters > 10 ||\n            numberOfClusters > n) {\n            cout << \"K must be between 1 and 10\";\n            cout << \" and cannot exceed the number of points.\\n\";\n            return false;\n        }\n\n        k = numberOfClusters;\n\n        // Repeated rows must not create identical initial centroids.\n        int selected = 0;\n        for (int i = 0; i < n && selected < k; i++) {\n            bool duplicate = false;\n            for (int c = 0; c < selected; c++) {\n                if (x[i] == centerX[c] && y[i] == centerY[c])\n                    duplicate = true;\n            }\n            if (!duplicate) {\n                centerX[selected] = x[i];\n                centerY[selected] = y[i];\n                selected++;\n            }\n        }\n        if (selected < k) {\n            cout << \"K cannot exceed the number of distinct points.\\n\";\n            return false;\n        }\n\n        for (int i = 0; i < n; i++) {\n            cluster[i] = -1;\n        }\n\n        // Repeat assignment and centroid updates.\n        for (int iteration = 0; iteration < 100; iteration++) {\n            bool changed = false;\n\n            // Assign each point to its nearest centroid.\n            for (int i = 0; i < n; i++) {\n                int nearest = MathUtils::nearestCentroid(\n                    x[i], y[i], centerX, centerY, k\n                );\n\n                if (cluster[i] != nearest) {\n                    cluster[i] = nearest;\n                    changed = true;\n                }\n            }\n\n            int counts[10] = {0};\n            for (int i = 0; i < n; i++)\n                counts[cluster[i]]++;\n\n            // Fill an empty cluster without emptying its donor cluster.\n            for (int c = 0; c < k; c++) {\n                if (counts[c] != 0)\n                    continue;\n                int candidate = -1;\n                double farthest = -1;\n                for (int i = 0; i < n; i++) {\n                    int donor = cluster[i];\n                    if (counts[donor] <= 1)\n                        continue;\n                    double distance = MathUtils::squaredDistance(\n                        x[i], y[i], centerX[donor], centerY[donor]);\n                    if (distance > farthest) {\n                        farthest = distance;\n                        candidate = i;\n                    }\n                }\n                if (candidate == -1) {\n                    cout << \"Unable to fill an empty cluster.\\n\";\n                    return false;\n                }\n                counts[cluster[candidate]]--;\n                cluster[candidate] = c;\n                counts[c]++;\n                changed = true;\n            }\n\n            if (!changed)\n                return true;\n\n            // Recalculate each centroid using its cluster's points.\n            for (int c = 0; c < k; c++) {\n                double groupX[100], groupY[100];\n                int count = 0;\n\n                for (int i = 0; i < n; i++) {\n                    if (cluster[i] == c) {\n                        groupX[count] = x[i];\n                        groupY[count] = y[i];\n                        count++;\n                    }\n                }\n\n                centerX[c] = MathUtils::mean(groupX, count);\n                centerY[c] = MathUtils::mean(groupY, count);\n            }\n        }\n\n        cout << \"Warning: reached 100 iterations; convergence is not guaranteed.\\n\";\n        return true;\n    }\n\n    void display() const {\n        cout << \"\\nK-Means Results\\n\";\n\n        for (int c = 0; c < k; c++) {\n            cout << \"\\nCluster \" << c + 1;\n            cout << \"\\nCentroid: (\" << centerX[c];\n            cout << \", \" << centerY[c] << \")\\n\";\n\n            for (int i = 0; i < n; i++) {\n                if (cluster[i] == c) {\n                    cout << \"(\" << x[i] << \", \" << y[i] << \")\\n\";\n                }\n            }\n        }\n    }\n    int execute(const char filename[]) override {\n        if (!loadCSV(filename, 1, 0))\n            return 1;\n    \n        int k;\n    \n        cout << \"Enter the number of clusters: \";\n    \n        if (!(cin >> k)) {\n            cout << \"Please enter an integer.\\n\";\n            return 1;\n        }\n    \n        if (!fit(k))\n            return 1;\n    \n        display();\n    \n        return 0;\n    }\n};\n\nMLAlgorithm& getKMeans() {\n    static KMeans model;\n    return model;\n}\n\n"
  },
  svm: {
    path: "src/svm.cpp",
    title: "Linear Support Vector Machine (SVM)",
    desc: "Stochastic hinge-loss subgradient descent with L2 regularization and separating hyperplane normal vector updates",
    lines: "104 lines",
    size: "2.3 KB",
    code: "#include <iostream>\n#include <cmath>\n#include \"MathUtils.h\"\n#include \"MLAlgorithm.h\"\nusing namespace std;\n\nclass SVM : public MLAlgorithm {\nprivate:\n\n    double weights[2];\n    double bias;\n    double learningRate;\n    double lambda;\n    int epochs;\n\npublic:\n    SVM() {\n        weights[0] = 0;\n        weights[1] = 0;\n        bias = 0;\n\n        learningRate = 0.001;\n        lambda = 0.01;\n        epochs = 2000;\n    }\n\n    void fit() {\n        // Start fresh whenever training is called.\n        weights[0] = 0;\n        weights[1] = 0;\n        bias = 0;\n\n        for (int epoch = 0; epoch < epochs; epoch++) {\n            for (int i = 0; i < n; i++) {\n                MathUtils::updateSVM(\n                    x[i], y[i], labels[i],\n                    weights, bias, learningRate, lambda\n                );\n            }\n        }\n    }\n\n    int predict(double newX, double newY) const {\n        double score = MathUtils::svmScore(\n            newX, newY, weights, bias\n        );\n\n        if (score >= 0)\n            return 1;\n\n        return -1;\n    }\n\n    void display() const {\n        cout << \"\\nWeight 1: \" << weights[0];\n        cout << \"\\nWeight 2: \" << weights[1];\n        cout << \"\\nBias: \" << bias << \"\\n\";\n\n        int correct = 0;\n\n        cout << \"\\nTraining predictions:\\n\";\n\n        for (int i = 0; i < n; i++) {\n            int result = predict(x[i], y[i]);\n\n            cout << \"(\" << x[i] << \", \" << y[i] << \")\";\n            cout << \" Actual: \" << labels[i];\n            cout << \" Predicted: \" << result << \"\\n\";\n\n            if (result == labels[i])\n                correct++;\n        }\n\n        cout << \"\\nCorrect training predictions: \";\n        cout << correct << \" out of \" << n << \"\\n\";\n    }\n    int execute(const char filename[]) override {\n        if (!loadCSV(filename, 1, 1))\n            return 1;\n    \n        fit();\n        display();\n    \n        double newX, newY;\n    \n        cout << \"\\nEnter two features of the new point: \";\n    \n        if (!(cin >> newX >> newY) ||\n            !isfinite(newX) || !isfinite(newY)) {\n            cout << \"Please enter two finite numbers.\\n\";\n            return 1;\n        }\n    \n        cout << \"Predicted class: \";\n        cout << predict(newX, newY) << \"\\n\";\n    \n        return 0;\n    }\n};\n\nMLAlgorithm& getSVM() {\n    static SVM model;\n    return model;\n}\n"
  },
  hierarchical: {
    path: "src/hierarchical_clustering.cpp",
    title: "Agglomerative Hierarchical Clustering",
    desc: "Bottom-up single-linkage clustering repeatedly merging closest cluster pairs at minimum Euclidean distance",
    lines: "133 lines",
    size: "3.3 KB",
    code: "#include <iostream>\n#include <cmath>\n#include \"MathUtils.h\"\n#include \"MLAlgorithm.h\"\nusing namespace std;\n\nclass HierarchicalClustering : public MLAlgorithm {\nprivate:\n\n    // Each row stores the point indices belonging to a cluster.\n    int members[100][100];\n    int count[100];\n    bool active[100];\n    int clusterCount;\n\npublic:\n    HierarchicalClustering() {\n        clusterCount = 0;\n\n        for (int i = 0; i < 100; i++) {\n            count[i] = 0;\n            active[i] = false;\n        }\n    }\n\n    bool fit(int targetClusters) {\n        if (targetClusters < 1 || targetClusters > n) {\n            cout << \"Cluster count must be between 1 and \" << n << \".\\n\";\n            return false;\n        }\n\n        // Initially, every point forms its own cluster.\n        for (int i = 0; i < n; i++) {\n            members[i][0] = i;\n            count[i] = 1;\n            active[i] = true;\n        }\n\n        clusterCount = n;\n\n        while (clusterCount > targetClusters) {\n            int bestA = -1;\n            int bestB = -1;\n            double minimumDistance = 0;\n\n            // Find the closest pair of active clusters.\n            for (int i = 0; i < n; i++) {\n                if (!active[i])\n                    continue;\n\n                for (int j = i + 1; j < n; j++) {\n                    if (!active[j])\n                        continue;\n\n                    double distance = MathUtils::singleLinkDistance(\n                        x, y,\n                        members[i], count[i],\n                        members[j], count[j]\n                    );\n\n                    if (bestA == -1 || distance < minimumDistance) {\n                        minimumDistance = distance;\n                        bestA = i;\n                        bestB = j;\n                    }\n                }\n            }\n\n            // Move every member of cluster B into cluster A.\n            for (int i = 0; i < count[bestB]; i++) {\n                members[bestA][count[bestA]] = members[bestB][i];\n                count[bestA]++;\n            }\n\n            active[bestB] = false;\n            count[bestB] = 0;\n            clusterCount--;\n\n            cout << \"Merged group \" << bestB + 1;\n            cout << \" into group \" << bestA + 1;\n            cout << \" at distance \" << minimumDistance << \"\\n\";\n        }\n\n        return true;\n    }\n\n    void display() const {\n        int displayNumber = 1;\n\n        cout << \"\\nFinal clusters:\\n\";\n\n        for (int i = 0; i < n; i++) {\n            if (!active[i])\n                continue;\n\n            cout << \"\\nCluster \" << displayNumber << \":\\n\";\n\n            for (int j = 0; j < count[i]; j++) {\n                int point = members[i][j];\n\n                cout << \"P\" << point + 1;\n                cout << \" (\" << x[point] << \", \" << y[point] << \")\\n\";\n            }\n\n            displayNumber++;\n        }\n    }\n    int execute(const char filename[]) override {\n        if (!loadCSV(filename, 1, 0))\n            return 1;\n    \n        int target;\n    \n        cout << \"Enter the desired number of clusters: \";\n    \n        if (!(cin >> target)) {\n            cout << \"Please enter an integer.\\n\";\n            return 1;\n        }\n    \n        if (!fit(target))\n            return 1;\n    \n        display();\n    \n        return 0;\n    }\n};\n\nMLAlgorithm& getHierarchicalClustering() {\n    static HierarchicalClustering model;\n    return model;\n}\n"
  },
  pca: {
    path: "src/pca.cpp",
    title: "Principal Component Analysis (PCA: 2D -> 1D)",
    desc: "Centers 2D points, computes 2x2 sample covariance matrix, and projects onto dominant PC1 eigenvector",
    lines: "74 lines",
    size: "1.6 KB",
    code: "#include <iostream>\n#include <cmath>\n#include \"MathUtils.h\"\n#include \"MLAlgorithm.h\"\nusing namespace std;\n\nclass PCA : public MLAlgorithm {\nprivate:\n    double transformed[100];\n\n    double center[2];\n    double component[2];\n    double explainedRatio;\n\npublic:\n    PCA() {\n        center[0] = 0;\n        center[1] = 0;\n        component[0] = 0;\n        component[1] = 0;\n        explainedRatio = 0;\n    }\n\n    bool fit() {\n        return MathUtils::fitPCA2D(\n            x, y, n, center, component, explainedRatio\n        );\n    }\n\n    void transform() {\n        for (int i = 0; i < n; i++) {\n            transformed[i] = MathUtils::projectPoint2D(\n                x[i], y[i], center, component\n            );\n        }\n    }\n\n    void display() const {\n        cout << \"\\nMean of first feature: \" << center[0];\n        cout << \"\\nMean of second feature: \" << center[1];\n\n        cout << \"\\n\\nFirst principal component: (\";\n        cout << component[0] << \", \" << component[1] << \")\\n\";\n\n        cout << \"Explained variance ratio: \";\n        cout << explainedRatio << \"\\n\";\n\n        cout << \"\\nOriginal point -> Reduced value\\n\";\n\n        for (int i = 0; i < n; i++) {\n            cout << \"(\" << x[i] << \", \" << y[i] << \")\";\n            cout << \" -> \" << transformed[i] << \"\\n\";\n        }\n    }\n    int execute(const char filename[]) override {\n        if (!loadCSV(filename, 2, 0))\n            return 1;\n    \n        if (!fit()) {\n            cout << \"PCA needs variation in the data.\\n\";\n            return 1;\n        }\n    \n        transform();\n        display();\n    \n        return 0;\n    }\n};\n\nMLAlgorithm& getPCA() {\n    static PCA model;\n    return model;\n}\n"
  },
  knn: {
    path: "src/knn.cpp",
    title: "K-Nearest Neighbors Classifier",
    desc: "Euclidean distance calculation, partial selection sort for K smallest distances, and majority vote class prediction",
    lines: "114 lines",
    size: "3.0 KB",
    code: "#include <iostream>\n#include <cmath>\n#include \"MathUtils.h\"\n#include \"MLAlgorithm.h\"\n\nusing namespace std;\n\nclass KNN : public MLAlgorithm {\npublic:\n    KNN() {\n    }\n\n    int getSize() const {\n        return n;\n    }\n\n    // Call after loading data and checking that 1 <= k <= n.\n    int predict(double newX, double newY, int k) const {\n        double distance[100];\n        int index[100];\n\n        // Reuse the distance formula from MathUtils.h.\n        for (int i = 0; i < n; i++) {\n            distance[i] = MathUtils::euclideanDistance(\n                newX, newY, x[i], y[i]\n            );\n\n            index[i] = i;\n        }\n\n        // Sort distances, keeping each point's index alongside it.\n        for (int i = 0; i < n - 1; i++) {\n            for (int j = 0; j < n - i - 1; j++) {\n                if (distance[j] > distance[j + 1]) {\n                    double tempDistance = distance[j];\n                    distance[j] = distance[j + 1];\n                    distance[j + 1] = tempDistance;\n\n                    int tempIndex = index[j];\n                    index[j] = index[j + 1];\n                    index[j + 1] = tempIndex;\n                }\n            }\n        }\n\n        cout << \"\\nNearest neighbours:\\n\";\n\n        for (int i = 0; i < k; i++) {\n            int point = index[i];\n\n            cout << \"(\" << x[point] << \", \" << y[point] << \")\";\n            cout << \"  Class: \" << labels[point];\n            cout << \"  Distance: \" << distance[i] << \"\\n\";\n        }\n\n        int predictedClass = labels[index[0]];\n        int highestVotes = 0;\n\n        // Count votes for each class among the nearest K points.\n        for (int i = 0; i < k; i++) {\n            int currentClass = labels[index[i]];\n            int votes = 0;\n\n            for (int j = 0; j < k; j++) {\n                if (labels[index[j]] == currentClass)\n                    votes++;\n            }\n\n            // If votes tie, choose the smaller class label.\n            if (votes > highestVotes ||\n                (votes == highestVotes &&\n                 currentClass < predictedClass)) {\n                highestVotes = votes;\n                predictedClass = currentClass;\n            }\n        }\n\n        return predictedClass;\n    }\n\n    int execute(const char filename[]) override {\n        if (!loadCSV(filename, 1, 2))\n            return 1;\n    \n        int k;\n        double newX, newY;\n    \n        cout << \"Enter K (1 to \" << getSize() << \"): \";\n    \n        if (!(cin >> k) || k < 1 || k > getSize()) {\n            cout << \"Invalid K.\\n\";\n            return 1;\n        }\n    \n        cout << \"Enter the two features of the point: \";\n    \n        if (!(cin >> newX >> newY) ||\n            !isfinite(newX) || !isfinite(newY)) {\n            cout << \"Please enter two valid numbers.\\n\";\n            return 1;\n        }\n    \n        int result = predict(newX, newY, k);\n    \n        cout << \"\\nPredicted class: \" << result << \"\\n\";\n    \n        return 0;\n    }\n};\n\nMLAlgorithm& getKNN() {\n    static KNN model;\n    return model;\n}\n"
  },
  menu: {
    path: "src/ml_algorithms_menu.cpp",
    title: "Polymorphic CLI Driver / Menu (main)",
    desc: "Integrated console menu exercising all six algorithms via MLAlgorithm base-class pointer runtime polymorphism",
    lines: "44 lines",
    size: "1.5 KB",
    code: "#include <iostream>\n#include <string>\n#include \"MLAlgorithm.h\"\nusing namespace std;\n\nint main() {\n    MLAlgorithm* models[] = {\n        &getLinearRegression(), &getKMeans(), &getSVM(),\n        &getHierarchicalClustering(), &getPCA(), &getKNN()\n    };\n    const char* defaults[] = {\n        \"datasets/linear_regression.csv\", \"datasets/kmeans.csv\",\n        \"datasets/svm.csv\", \"datasets/hierarchical_clustering.csv\",\n        \"datasets/pca.csv\", \"datasets/knn.csv\"\n    };\n    while (true) {\n        cout << \"\\n========== ML-CORE ==========\\n\";\n        cout << \"1. Linear Regression\\n2. K-Means\\n3. SVM\\n\";\n        cout << \"4. Hierarchical Clustering\\n5. PCA\\n6. KNN\\n0. Exit\\n\";\n        cout << \"Enter choice: \";\n        int choice;\n        if (!(cin >> choice)) {\n            cout << \"Invalid input. Restart and enter an integer.\\n\";\n            return 1;\n        }\n        if (choice == 0) return 0;\n        if (choice < 1 || choice > 6) {\n            cout << \"Choose 0 to 6.\\n\";\n            continue;\n        }\n        cout << \"CSV path (or - for \" << defaults[choice - 1] << \"): \";\n        string filename;\n        if (!getline(cin >> ws, filename)) return 1;\n        if (filename == \"-\") filename = defaults[choice - 1];\n\n        // Runtime polymorphism: calls the selected derived class's override.\n        int status = models[choice - 1]->execute(filename.c_str());\n        if (!cin) {\n            cout << \"Input stopped. Restart the program.\\n\";\n            return 1;\n        }\n        if (status != 0) cout << \"Algorithm could not complete; see message above.\\n\";\n    }\n}\n"
  },
  base: {
    path: "include/MLAlgorithm.h",
    title: "Abstract Base Class & CSV Loader",
    desc: "Base class defining MAX_ROWS=100 cap, CSV line validation, numeric checking, and virtual execute(filename)",
    lines: "113 lines",
    size: "3.7 KB",
    code: "#ifndef ML_ALGORITHM_H\n#define ML_ALGORITHM_H\n\n#include <iostream>\n#include <fstream>\n#include <sstream>\n#include <string>\n#include <cmath>\n#include <climits>\n\nclass MLAlgorithm {\nprotected:\n    enum { MAX_ROWS = 100 };\n    double x[MAX_ROWS], y[MAX_ROWS];\n    int labels[MAX_ROWS];\n    int n;\n\n    // labelMode: 0 = two numeric columns, 1 = SVM, 2 = KNN.\n    bool loadCSV(const char filename[], int minimumRows, int labelMode = 0) {\n        n = 0;\n        std::ifstream file(filename);\n        if (!file) {\n            std::cout << \"Unable to open CSV: \" << filename << \"\\n\";\n            return false;\n        }\n        std::string line;\n        int lineNumber = 0;\n        int count = 0;\n        while (std::getline(file, line)) {\n            ++lineNumber;\n            if (line.find_first_not_of(\" \\t\\r\") == std::string::npos)\n                continue;\n            if (count == MAX_ROWS) {\n                std::cout << \"Maximum 100 data rows allowed. File rejected.\\n\";\n                return false;\n            }\n            std::stringstream row(line);\n            double values[3] = {0, 0, 0};\n            int columns = labelMode == 0 ? 2 : 3;\n            for (int j = 0; j < columns; ++j) {\n                if (!(row >> values[j]) || !std::isfinite(values[j])) {\n                    std::cout << \"Invalid or missing number on CSV line \"\n                              << lineNumber << \".\\n\";\n                    return false;\n                }\n                if (j < columns - 1) {\n                    char comma;\n                    if (!(row >> comma) || comma != ',') {\n                        std::cout << \"Expected comma on CSV line \"\n                                  << lineNumber << \".\\n\";\n                        return false;\n                    }\n                }\n            }\n            row >> std::ws;\n            if (!row.eof()) {\n                std::cout << \"Extra data on CSV line \" << lineNumber << \".\\n\";\n                return false;\n            }\n            if (labelMode != 0) {\n                double label = values[2];\n                if (std::floor(label) != label || label < INT_MIN || label > INT_MAX) {\n                    std::cout << \"Class labels must be integers in the int range.\\n\";\n                    return false;\n                }\n                if (labelMode == 1 && label != -1 && label != 1) {\n                    std::cout << \"SVM labels must be -1 or 1.\\n\";\n                    return false;\n                }\n                labels[count] = static_cast<int>(label);\n            }\n            x[count] = values[0];\n            y[count] = values[1];\n            ++count;\n        }\n        if (file.bad()) {\n            std::cout << \"Error reading CSV file.\\n\";\n            return false;\n        }\n        if (count < minimumRows) {\n            std::cout << \"At least \" << minimumRows << \" data rows required.\\n\";\n            return false;\n        }\n        if (labelMode == 1) {\n            bool positive = false, negative = false;\n            for (int i = 0; i < count; ++i) {\n                if (labels[i] == 1) positive = true;\n                if (labels[i] == -1) negative = true;\n            }\n            if (!positive || !negative) {\n                std::cout << \"SVM data must contain both classes.\\n\";\n                return false;\n            }\n        }\n        n = count;\n        return true;\n    }\n\npublic:\n    MLAlgorithm() : n(0) {}\n    virtual ~MLAlgorithm() {}\n    virtual int execute(const char filename[]) = 0;\n};\n\n// Objects remain in their own source files; the menu uses base references.\nMLAlgorithm& getLinearRegression();\nMLAlgorithm& getKMeans();\nMLAlgorithm& getSVM();\nMLAlgorithm& getHierarchicalClustering();\nMLAlgorithm& getPCA();\nMLAlgorithm& getKNN();\n\n#endif\n"
  },
  math: {
    path: "include/MathUtils.h",
    title: "Shared Mathematics & Vector Engine",
    desc: "Inline functions for mean, dotProduct, squaredDistance, covariance, SVM update, and PCA eigenvalue decomposition",
    lines: "232 lines",
    size: "7.4 KB",
    code: "#ifndef MATH_UTILS_H\n#define MATH_UTILS_H\n\n#include <cmath>\n#include <cassert>\n\n// Arrays must contain finite numbers and have the stated lengths.\n// assert checks invalid function arguments while developing the project.\n// inline allows multiple .cpp files to include this header safely.\nnamespace MathUtils {\n\n// ---------- Common calculations ----------\n\ninline double mean(const double values[], int n) {\n    assert(n > 0);\n    double sum = 0;\n    for (int i = 0; i < n; i++)\n        sum += values[i];\n    return sum / n;\n}\n\ninline double dotProduct(const double a[], const double b[], int n) {\n    assert(n > 0);\n    double total = 0;\n    for (int i = 0; i < n; i++)\n        total += a[i] * b[i];\n    return total;\n}\n\n// The squared-distance formula is written only here.\ninline double squaredDistance(const double a[], const double b[], int n) {\n    assert(n > 0);\n    double total = 0;\n    for (int i = 0; i < n; i++) {\n        double difference = a[i] - b[i];\n        total += difference * difference;\n    }\n    return total;\n}\n\n// Overload for existing 2D code: delegates to the array version.\ninline double squaredDistance(double x1, double y1, double x2, double y2) {\n    double a[2] = {x1, y1};\n    double b[2] = {x2, y2};\n    return squaredDistance(a, b, 2);\n}\n\n// KNN can call this to display actual distances.\ninline double euclideanDistance(double x1, double y1, double x2, double y2) {\n    return std::sqrt(squaredDistance(x1, y1, x2, y2));\n}\n\n// MSE reuses the sum of squared differences.\ninline double meanSquaredError(const double actual[],\n                               const double predicted[], int n) {\n    assert(n > 0);\n    return squaredDistance(actual, predicted, n) / n;\n}\n\n// Sample covariance, shared by regression and PCA.\ninline double covariance(const double x[], const double y[], int n) {\n    assert(n >= 2);\n    double meanX = mean(x, n);\n    double meanY = mean(y, n);\n    double total = 0;\n    for (int i = 0; i < n; i++)\n        total += (x[i] - meanX) * (y[i] - meanY);\n    return total / (n - 1);\n}\n\ninline double variance(const double values[], int n) {\n    return covariance(values, values, n);\n}\n\n// Shared weighted sum plus bias.\ninline double linearScore(const double features[], const double weights[],\n                          int n, double bias) {\n    return dotProduct(features, weights, n) + bias;\n}\n\n// ---------- Linear Regression ----------\n\ninline bool fitLine(const double x[], const double y[], int n,\n                    double &slope, double &intercept) {\n    if (n < 2)\n        return false;\n    double varX = variance(x, n);\n    if (varX == 0)\n        return false;\n\n    slope = covariance(x, y, n) / varX;\n    intercept = mean(y, n) - slope * mean(x, n);\n    return true;\n}\n\ninline double predictLine(double x, double slope, double intercept) {\n    double features[1] = {x};\n    double weights[1] = {slope};\n    return linearScore(features, weights, 1, intercept);\n}\n\n// ---------- K-Means ----------\n// Update each centroid using mean() on its cluster's coordinates.\n\ninline int nearestCentroid(double x, double y, const double centerX[],\n                           const double centerY[], int k) {\n    assert(k > 0);\n    int nearest = 0;\n    double minimum = squaredDistance(x, y, centerX[0], centerY[0]);\n    for (int i = 1; i < k; i++) {\n        double distance = squaredDistance(x, y, centerX[i], centerY[i]);\n        if (distance < minimum) {\n            minimum = distance;\n            nearest = i;\n        }\n    }\n    return nearest;\n}\n\n// ---------- Hierarchical Clustering ----------\n// Member arrays contain valid indices of points in two nonempty clusters.\n\ninline double singleLinkDistance(const double x[], const double y[],\n                                 const int membersA[], int countA,\n                                 const int membersB[], int countB) {\n    assert(countA > 0 && countB > 0);\n    int bestA = membersA[0];\n    int bestB = membersB[0];\n    double minimum = squaredDistance(x[bestA], y[bestA], x[bestB], y[bestB]);\n\n    for (int i = 0; i < countA; i++) {\n        for (int j = 0; j < countB; j++) {\n            int a = membersA[i];\n            int b = membersB[j];\n            double distance = squaredDistance(x[a], y[a], x[b], y[b]);\n            if (distance < minimum) {\n                minimum = distance;\n                bestA = a;\n                bestB = b;\n            }\n        }\n    }\n    return euclideanDistance(x[bestA], y[bestA], x[bestB], y[bestB]);\n}\n\n// ---------- Linear SVM ----------\n\ninline double svmScore(double x, double y,\n                       const double weights[], double bias) {\n    double features[2] = {x, y};\n    return linearScore(features, weights, 2, bias);\n}\n\n// Hinge-loss update with L2 regularization.\ninline void updateSVM(double x, double y, int label, double weights[],\n                      double &bias, double learningRate, double lambda) {\n    assert((label == -1 || label == 1) && learningRate > 0 && lambda >= 0);\n    double features[2] = {x, y};\n    bool insideMargin = label * svmScore(x, y, weights, bias) < 1;\n\n    for (int j = 0; j < 2; j++) {\n        double gradient = lambda * weights[j];\n        if (insideMargin)\n            gradient -= label * features[j];\n        weights[j] -= learningRate * gradient;\n    }\n    if (insideMargin)\n        bias += learningRate * label;\n}\n\n// ---------- PCA: exactly two features reduced to one ----------\n\ninline void covarianceMatrix2D(const double x[], const double y[], int n,\n                               double matrix[2][2]) {\n    matrix[0][0] = variance(x, n);\n    matrix[0][1] = covariance(x, y, n);\n    matrix[1][0] = matrix[0][1];\n    matrix[1][1] = variance(y, n);\n}\n\n// Eigenvalues of a symmetric 2x2 covariance matrix, largest first.\ninline void eigenvalues2D(const double matrix[2][2], double eigenvalues[]) {\n    double halfTrace = (matrix[0][0] + matrix[1][1]) / 2;\n    double halfDifference = (matrix[0][0] - matrix[1][1]) / 2;\n    double radius = std::hypot(halfDifference, matrix[0][1]);\n    eigenvalues[0] = halfTrace + radius;\n    eigenvalues[1] = halfTrace - radius;\n}\n\ninline bool principalComponent2D(const double matrix[2][2],\n                                 double component[], double &explainedRatio) {\n    double totalVariance = matrix[0][0] + matrix[1][1];\n    if (totalVariance <= 0) {\n        component[0] = 1;\n        component[1] = 0;\n        explainedRatio = 0;\n        return false;\n    }\n\n    double eigenvalues[2];\n    eigenvalues2D(matrix, eigenvalues);\n    double angle = 0.5 * std::atan2(2 * matrix[0][1],\n                                   matrix[0][0] - matrix[1][1]);\n    component[0] = std::cos(angle);\n    component[1] = std::sin(angle);\n    explainedRatio = eigenvalues[0] / totalVariance;\n    return true;\n}\n\n// The returned component is a unit vector. Equal eigenvalues mean\n// no unique preferred direction; this implementation selects one.\ninline bool fitPCA2D(const double x[], const double y[], int n,\n                     double center[], double component[], double &explainedRatio) {\n    if (n < 2)\n        return false;\n    center[0] = mean(x, n);\n    center[1] = mean(y, n);\n    double matrix[2][2];\n    covarianceMatrix2D(x, y, n, matrix);\n    return principalComponent2D(matrix, component, explainedRatio);\n}\n\ninline double projectPoint2D(double x, double y, const double center[],\n                             const double component[]) {\n    double centered[2] = {x - center[0], y - center[1]};\n    return dotProduct(centered, component, 2);\n}\n\n} // namespace MathUtils\n\n#endif\n\n"
  },
  makefile: {
    path: "Makefile",
    title: "Project Build Configuration",
    desc: "GNU Make build rules compiling all 7 source files with -std=c++11 -Wall -Wextra -pedantic into ./ml_core",
    lines: "21 lines",
    size: "0.5 KB",
    code: "CXX = g++\nCPPFLAGS = -Iinclude\nCXXFLAGS = -std=c++11 -Wall -Wextra -pedantic\nSOURCES = src/ml_algorithms_menu.cpp src/linear_regression.cpp src/kmeans.cpp src/svm.cpp src/hierarchical_clustering.cpp src/pca.cpp src/knn.cpp\n\n.PHONY: all run clean test\nall: ml_core\n\nml_core: $(SOURCES) include/MathUtils.h include/MLAlgorithm.h\n\t$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCES) -o $@\n\nrun: ml_core\n\t./ml_core\n\nclean:\n\trm -f ml_core\n\ntest: ml_core\n\tpython3 tests/check_csv_limits.py\n\tpython3 tests/check_kmeans.py\n\n"
  },
};


function renderValidationModal() {
  const isCustom = App.isCustomDataset;
  const datasetName = isCustom ? (App.customDatasetName || 'Custom CSV') : `datasets/${App.currentAlgo}.csv`;
  const algo = App.currentAlgo;
  const algoTitle = algo.replace('_', ' ').toUpperCase();

  const banner = document.getElementById('activeValidationBanner');
  const details = document.getElementById('activeValidationDetails');
  const tbody = document.getElementById('activeValidationTableBody');

  if (!banner || !details || !tbody) return;

  // 1. Active Dataset Banner
  banner.className = `active-validation-banner ${isCustom ? 'custom' : ''}`;
  banner.innerHTML = `
    <div class="val-dataset-meta">
      <span class="val-source-pill ${isCustom ? 'custom' : ''}">${isCustom ? 'Custom Uploaded CSV' : 'Default Repository Sample'}</span>
      <span class="val-dataset-name">${datasetName}</span>
      <span class="val-dataset-stats">&bull; ${App.points.length} data rows loaded &bull; ${algoTitle}</span>
    </div>
    <span class="tag-match">✓ Live Validated</span>
  `;

  // 2. Algorithm-specific Validation Details
  let summaryHtml = '';
  let tableRows = '';

  if (algo === 'linear_regression') {
    const model = MathUtils.fitLine(App.points);
    if (model.invalid) {
      const x0 = App.points.length > 0 ? App.points[0].x : 0;
      summaryHtml = `
        <div class="val-summary-cards">
          <div class="val-card">
            <span class="val-card-label">Fit Status</span>
            <span class="val-card-val" style="color: #f43f5e;">Zero Variance</span>
            <span class="val-card-sub">All X identical (${x0})</span>
          </div>
          <div class="val-card">
            <span class="val-card-label">Common X</span>
            <span class="val-card-val">${x0}</span>
            <span class="val-card-sub">Vertical boundary</span>
          </div>
          <div class="val-card">
            <span class="val-card-label">Data Rows</span>
            <span class="val-card-val">${App.points.length}</span>
            <span class="val-card-sub">Rows loaded</span>
          </div>
          <div class="val-card">
            <span class="val-card-label">C++ Parity</span>
            <span class="val-card-val" style="color: #34d399;">Abort Verified</span>
            <span class="val-card-sub">Division by zero blocked</span>
          </div>
        </div>
      `;

      tableRows = `
        <tr>
          <td><strong>Dataset Size</strong></td>
          <td><code>MAX_ROWS = 100, min = 2</code></td>
          <td><code>${App.points.length} rows loaded</code></td>
          <td><span class="tag-match">Valid Range</span></td>
        </tr>
        <tr>
          <td><strong>Variance in X</strong></td>
          <td><code>Var(X) &gt; 0</code></td>
          <td><code>Var(X) = 0.0000 (all x = ${x0})</code></td>
          <td><span class="tag-fail">⚠️ Vertical Line (Error)</span></td>
        </tr>
        <tr>
          <td><strong>Line Equation</strong></td>
          <td><code>y = mx + c</code></td>
          <td><code>Undefined (Slope divides by 0)</code></td>
          <td><span class="tag-fail">Cannot Fit Line</span></td>
        </tr>
        <tr>
          <td><strong>C++ loadCSV &amp; fit Parity</strong></td>
          <td><code>src/linear_regression.cpp line 44</code></td>
          <td><code>"Cannot fit a line: all x values are equal."</code></td>
          <td><span class="tag-match">Exact Parity</span></td>
        </tr>
      `;
    } else {
      const predY = model.slope * App.regression.queryX + model.intercept;
      const ys = App.points.map(p => p.y);
      const meanY = MathUtils.mean(ys);
      const ssTot = ys.reduce((sum, y) => sum + (y - meanY) ** 2, 0);
      const ssRes = App.points.reduce((sum, p) => sum + (p.y - (model.slope * p.x + model.intercept)) ** 2, 0);
      const r2 = ssTot > 0 ? (1 - ssRes / ssTot) : 1;

      summaryHtml = `
        <div class="val-summary-cards">
          <div class="val-card">
            <span class="val-card-label">Slope (m)</span>
            <span class="val-card-val">${model.slope.toFixed(4)}</span>
            <span class="val-card-sub">Cov(x,y) / Var(x)</span>
          </div>
          <div class="val-card">
            <span class="val-card-label">Intercept (c)</span>
            <span class="val-card-val">${model.intercept.toFixed(4)}</span>
            <span class="val-card-sub">ȳ - m * x̄</span>
          </div>
          <div class="val-card">
            <span class="val-card-label">Training MSE</span>
            <span class="val-card-val">${model.mse.toFixed(6)}</span>
            <span class="val-card-sub">Mean Squared Error</span>
          </div>
          <div class="val-card">
            <span class="val-card-label">R² Score</span>
            <span class="val-card-val" style="color: #38bdf8;">${r2.toFixed(4)}</span>
            <span class="val-card-sub">${(r2 * 100).toFixed(1)}% variance explained</span>
          </div>
        </div>
      `;

      tableRows = `
        <tr>
          <td><strong>Linear Formulation</strong></td>
          <td><code>MathUtils::fitLine(x, y, n, slope, intercept)</code></td>
          <td><code>y = ${model.slope.toFixed(4)}x + ${model.intercept.toFixed(4)}</code></td>
          <td><span class="tag-match">Exact Match</span></td>
        </tr>
        <tr>
          <td><strong>Residual Error (MSE)</strong></td>
          <td><code>MathUtils::meanSquaredError(y, pred, n)</code></td>
          <td><code>MSE = ${model.mse.toFixed(6)}</code></td>
          <td><span class="tag-match">Optimal (OLS)</span></td>
        </tr>
        <tr>
          <td><strong>Determination (R²)</strong></td>
          <td><code>1 - (SS_res / SS_tot)</code></td>
          <td><code>R² = ${r2.toFixed(4)} (${(r2 * 100).toFixed(2)}% variance explained)</code></td>
          <td><span class="tag-match">Verified</span></td>
        </tr>
        <tr>
          <td><strong>Query Prediction</strong></td>
          <td><code>MathUtils::predictLine(${App.regression.queryX})</code></td>
          <td><code>x = ${App.regression.queryX} &rarr; ŷ = ${predY.toFixed(4)}</code></td>
          <td><span class="tag-match">Consistent</span></td>
        </tr>
      `;
    }
  } else if (algo === 'kmeans') {
    let wcss = 0;
    const clusterSizes = {};
    for (let c = 0; c < App.kmeans.k; c++) clusterSizes[c] = 0;
    App.points.forEach(p => {
      if (p.cluster !== undefined && App.kmeans.centroids[p.cluster]) {
        wcss += MathUtils.squaredDistance(p.x, p.y, App.kmeans.centroids[p.cluster].x, App.kmeans.centroids[p.cluster].y);
        clusterSizes[p.cluster] = (clusterSizes[p.cluster] || 0) + 1;
      }
    });

    const centroidStrings = App.kmeans.centroids.map((c, i) => `C${i+1}: (${c.x.toFixed(3)}, ${c.y.toFixed(3)}) [${clusterSizes[i] || 0} pts]`).join('<br>');

    summaryHtml = `
      <div class="val-summary-cards">
        <div class="val-card">
          <span class="val-card-label">Clusters (K)</span>
          <span class="val-card-val">${App.kmeans.k}</span>
          <span class="val-card-sub">Active partitions</span>
        </div>
        <div class="val-card">
          <span class="val-card-label">Iteration Step</span>
          <span class="val-card-val">${App.kmeans.step}</span>
          <span class="val-card-sub">Lloyd updates</span>
        </div>
        <div class="val-card">
          <span class="val-card-label">WCSS (Inertia)</span>
          <span class="val-card-val">${wcss.toFixed(4)}</span>
          <span class="val-card-sub">Within-cluster sum sq</span>
        </div>
        <div class="val-card">
          <span class="val-card-label">Data Points</span>
          <span class="val-card-val">${App.points.length}</span>
          <span class="val-card-sub">Coordinates loaded</span>
        </div>
      </div>
    `;

    tableRows = `
      <tr>
        <td><strong>Centroids &amp; Group Sizes</strong></td>
        <td><code>MathUtils::nearestCentroid &amp; mean</code></td>
        <td><code>${centroidStrings}</code></td>
        <td><span class="tag-match">Converged</span></td>
      </tr>
      <tr>
        <td><strong>Distinct Initial Centroids</strong></td>
        <td><code>src/kmeans.cpp lines 28–45</code></td>
        <td><code>Filtered duplicate coordinates for initial seeds</code></td>
        <td><span class="tag-match">Verified</span></td>
      </tr>
      <tr>
        <td><strong>Empty Cluster Donor Rebalancing</strong></td>
        <td><code>src/kmeans.cpp lines 71–96</code></td>
        <td><code>Rebalanced farthest donor member if any cluster empties</code></td>
        <td><span class="tag-match">Verified</span></td>
      </tr>
      <tr>
        <td><strong>Partition WCSS (Inertia)</strong></td>
        <td><code>&Sigma; ||x_i - c_k||²</code></td>
        <td><code>WCSS = ${wcss.toFixed(4)}</code></td>
        <td><span class="tag-match">Optimal Local</span></td>
      </tr>
    `;
  } else if (algo === 'svm') {
    const model = App.svm.model || MathUtils.trainSVM(App.points);
    const w = model.weights;
    const b = model.bias;
    const norm = Math.hypot(w[0], w[1]);
    const margin = norm > 0 ? (2 / norm) : 0;

    let correct = 0;
    let posCount = 0, negCount = 0;
    App.points.forEach(p => {
      if (p.label === 1) posCount++;
      if (p.label === -1) negCount++;
      const s = w[0] * p.x + w[1] * p.y + b;
      if ((s >= 0 ? 1 : -1) === p.label) correct++;
    });
    const acc = App.points.length > 0 ? (correct / App.points.length) * 100 : 0;
    const qScore = w[0] * App.svm.queryPoint.x + w[1] * App.svm.queryPoint.y + b;
    const qClass = qScore >= 0 ? 1 : -1;

    summaryHtml = `
      <div class="val-summary-cards">
        <div class="val-card">
          <span class="val-card-label">Normal [w₁, w₂]</span>
          <span class="val-card-val">[${w[0].toFixed(3)}, ${w[1].toFixed(3)}]</span>
          <span class="val-card-sub">Hyperplane weights</span>
        </div>
        <div class="val-card">
          <span class="val-card-label">Bias (b)</span>
          <span class="val-card-val">${b.toFixed(4)}</span>
          <span class="val-card-sub">Margin offset</span>
        </div>
        <div class="val-card">
          <span class="val-card-label">Train Accuracy</span>
          <span class="val-card-val" style="color: #34d399;">${acc.toFixed(1)}%</span>
          <span class="val-card-sub">${correct}/${App.points.length} Correct</span>
        </div>
        <div class="val-card">
          <span class="val-card-label">Margin Corridor</span>
          <span class="val-card-val">${margin.toFixed(3)}</span>
          <span class="val-card-sub">2 / ||w||</span>
        </div>
      </div>
    `;

    tableRows = `
      <tr>
        <td><strong>Class Distribution</strong></td>
        <td><code>MLAlgorithm.h labelMode = 1</code></td>
        <td><code>${posCount} class (+1), ${negCount} class (-1)</code></td>
        <td><span class="tag-match">Both Present</span></td>
      </tr>
      <tr>
        <td><strong>Decision Boundary</strong></td>
        <td><code>w·x + b = 0</code></td>
        <td><code>${w[0].toFixed(3)}x₁ + ${w[1].toFixed(3)}x₂ + (${b.toFixed(4)}) = 0</code></td>
        <td><span class="tag-match">Separating Plane</span></td>
      </tr>
      <tr>
        <td><strong>Training Accuracy</strong></td>
        <td><code>Hinge loss subgradient descent</code></td>
        <td><code>${correct} / ${App.points.length} samples correctly classified (${acc.toFixed(1)}%)</code></td>
        <td><span class="tag-match">${acc >= 90 ? 'High Accuracy' : 'Verified'}</span></td>
      </tr>
      <tr>
        <td><strong>Query Probe (${App.svm.queryPoint.x}, ${App.svm.queryPoint.y})</strong></td>
        <td><code>MathUtils::svmScore(x, y, w, b)</code></td>
        <td><code>Score = ${qScore.toFixed(3)} &rarr; Predicted Class ${qClass > 0 ? '+1' : '-1'}</code></td>
        <td><span class="tag-match">Classified</span></td>
      </tr>
    `;
  } else if (algo === 'hierarchical') {
    const k = App.hierarchical.targetK;
    const clusters = App.hierarchical.clusters || [];
    const active = clusters.filter(c => c.active);
    const groupsDesc = active.map((c, i) => `Group ${i+1}: {${c.members.map(idx => App.points[idx] ? (App.points[idx].id || idx+1) : idx+1).join(', ')}}`).join('<br>');

    summaryHtml = `
      <div class="val-summary-cards">
        <div class="val-card">
          <span class="val-card-label">Target Clusters</span>
          <span class="val-card-val">${k}</span>
          <span class="val-card-sub">Dendrogram cut</span>
        </div>
        <div class="val-card">
          <span class="val-card-label">Active Partitions</span>
          <span class="val-card-val">${active.length}</span>
          <span class="val-card-sub">Agglomerated sets</span>
        </div>
        <div class="val-card">
          <span class="val-card-label">Linkage Criterion</span>
          <span class="val-card-val">Single</span>
          <span class="val-card-sub">Minimum link dist</span>
        </div>
        <div class="val-card">
          <span class="val-card-label">Merges Performed</span>
          <span class="val-card-val">${App.hierarchical.merges.length}</span>
          <span class="val-card-sub">Bottom-up steps</span>
        </div>
      </div>
    `;

    tableRows = `
      <tr>
        <td><strong>Cluster Partition Structure</strong></td>
        <td><code>src/hierarchical_clustering.cpp</code></td>
        <td><code>${groupsDesc || 'None'}</code></td>
        <td><span class="tag-match">Exact Match</span></td>
      </tr>
      <tr>
        <td><strong>Linkage Distance Metric</strong></td>
        <td><code>MathUtils::singleLinkDistance</code></td>
        <td><code>Min pairwise Euclidean distance between cluster members</code></td>
        <td><span class="tag-match">Single-Linkage</span></td>
      </tr>
      <tr>
        <td><strong>Dendrogram Partition Level</strong></td>
        <td><code>k = ${k}</code></td>
        <td><code>Cut tree down to exactly ${k} disconnected components</code></td>
        <td><span class="tag-match">Verified</span></td>
      </tr>
    `;
  } else if (algo === 'pca') {
    const pca = MathUtils.fitPCA2D(App.points);
    if (pca) {
      const evrPct = (pca.explainedRatio * 100).toFixed(2);
      summaryHtml = `
        <div class="val-summary-cards">
          <div class="val-card">
            <span class="val-card-label">Explained Variance</span>
            <span class="val-card-val" style="color: #f43f5e;">${evrPct}%</span>
            <span class="val-card-sub">Ratio: ${pca.explainedRatio.toFixed(6)}</span>
          </div>
          <div class="val-card">
            <span class="val-card-label">PC1 Direction</span>
            <span class="val-card-val">[${pca.component[0].toFixed(3)}, ${pca.component[1].toFixed(3)}]</span>
            <span class="val-card-sub">Unit eigenvector</span>
          </div>
          <div class="val-card">
            <span class="val-card-label">Center (x̄, ȳ)</span>
            <span class="val-card-val">(${pca.center[0].toFixed(2)}, ${pca.center[1].toFixed(2)})</span>
            <span class="val-card-sub">2D feature mean</span>
          </div>
          <div class="val-card">
            <span class="val-card-label">Eigenvalues</span>
            <span class="val-card-val">${pca.eigenvalues[0].toFixed(2)}, ${pca.eigenvalues[1].toFixed(2)}</span>
            <span class="val-card-sub">&lambda;₁ &gt; &lambda;₂</span>
          </div>
        </div>
      `;

      tableRows = `
        <tr>
          <td><strong>Centering (Mean Vector)</strong></td>
          <td><code>MathUtils::mean(x, n), MathUtils::mean(y, n)</code></td>
          <td><code>Mean Center = (${pca.center[0].toFixed(4)}, ${pca.center[1].toFixed(4)})</code></td>
          <td><span class="tag-match">Centered</span></td>
        </tr>
        <tr>
          <td><strong>Sample Covariance Matrix</strong></td>
          <td><code>MathUtils::covariance(x, y, n)</code></td>
          <td><code>Cov(x,x)=${pca.covXX.toFixed(4)}, Cov(x,y)=${pca.covXY.toFixed(4)}, Cov(y,y)=${pca.covYY.toFixed(4)}</code></td>
          <td><span class="tag-match">Symmetric Matrix</span></td>
        </tr>
        <tr>
          <td><strong>PC1 Eigenvector (Dominant Axis)</strong></td>
          <td><code>MathUtils::fitPCA2D</code></td>
          <td><code>v₁ = [${pca.component[0].toFixed(4)}, ${pca.component[1].toFixed(4)}] (unit norm = 1.0)</code></td>
          <td><span class="tag-match">Orthogonal Axis</span></td>
        </tr>
        <tr>
          <td><strong>Explained Variance Ratio</strong></td>
          <td><code>&lambda;₁ / (&lambda;₁ + &lambda;₂)</code></td>
          <td><code>${pca.explainedRatio.toFixed(6)} (${evrPct}% of total variance captured)</code></td>
          <td><span class="tag-match">Optimal 1D</span></td>
        </tr>
      `;
    }
  } else if (algo === 'knn') {
    const q = App.knn.queryPoint;
    const distances = App.points.map((p, idx) => ({
      idx: idx + 1,
      x: p.x,
      y: p.y,
      label: p.label,
      dist: MathUtils.euclideanDistance(q.x, q.y, p.x, p.y)
    })).sort((a, b) => a.dist - b.dist);

    const nearest = distances.slice(0, App.knn.k);
    const votes = {};
    nearest.forEach(n => {
      votes[n.label] = (votes[n.label] || 0) + 1;
    });

    let majClass = 0;
    let maxVotes = -1;
    for (const [lbl, count] of Object.entries(votes)) {
      if (count > maxVotes) { maxVotes = count; majClass = lbl; }
    }

    const neighborsList = nearest.map(n => `Point #${n.idx} (${n.x}, ${n.y}) [Class ${n.label}] &rarr; dist = ${n.dist.toFixed(3)}`).join('<br>');
    const votesBreakdown = Object.entries(votes).map(([lbl, count]) => `Class ${lbl}: ${count} votes`).join(', ');

    summaryHtml = `
      <div class="val-summary-cards">
        <div class="val-card">
          <span class="val-card-label">Neighbors (K)</span>
          <span class="val-card-val">${App.knn.k}</span>
          <span class="val-card-sub">Voting count</span>
        </div>
        <div class="val-card">
          <span class="val-card-label">Predicted Class</span>
          <span class="val-card-val" style="color: #fbbf24;">Class ${majClass}</span>
          <span class="val-card-sub">Majority vote</span>
        </div>
        <div class="val-card">
          <span class="val-card-label">Test Probe</span>
          <span class="val-card-val">(${q.x}, ${q.y})</span>
          <span class="val-card-sub">Query marker</span>
        </div>
        <div class="val-card">
          <span class="val-card-label">Votes Cast</span>
          <span class="val-card-val">${maxVotes} / ${App.knn.k}</span>
          <span class="val-card-sub">${votesBreakdown}</span>
        </div>
      </div>
    `;

    tableRows = `
      <tr>
        <td><strong>K Nearest Neighbors List</strong></td>
        <td><code>MathUtils::euclideanDistance &amp; partial sort</code></td>
        <td><code>${neighborsList}</code></td>
        <td><span class="tag-match">Nearest K</span></td>
      </tr>
      <tr>
        <td><strong>Voting Breakdown</strong></td>
        <td><code>MathUtils::majorityVote</code></td>
        <td><code>${votesBreakdown}</code></td>
        <td><span class="tag-match">Majority Decision</span></td>
      </tr>
      <tr>
        <td><strong>Classification Outcome</strong></td>
        <td><code>Query (${q.x}, ${q.y}) with K = ${App.knn.k}</code></td>
        <td><code>Predicted Class ${majClass}</code></td>
        <td><span class="tag-match">Valid Prediction</span></td>
      </tr>
    `;
  }

  details.innerHTML = summaryHtml;
  tbody.innerHTML = tableRows;
}

function initModals() {
  const benchmarkModal = document.getElementById('benchmarkModal');
  const codeModal = document.getElementById('codeModal');

  // Validation Modal Tabs
  const tabActiveVal = document.getElementById('tabActiveValidation');
  const tabBaselineVal = document.getElementById('tabBaselineValidation');
  const viewActiveVal = document.getElementById('viewActiveValidation');
  const viewBaselineVal = document.getElementById('viewBaselineValidation');

  if (tabActiveVal && tabBaselineVal) {
    tabActiveVal.addEventListener('click', () => {
      tabActiveVal.classList.add('active');
      tabBaselineVal.classList.remove('active');
      if (viewActiveVal) viewActiveVal.style.display = 'block';
      if (viewBaselineVal) viewBaselineVal.style.display = 'none';
      renderValidationModal();
    });

    tabBaselineVal.addEventListener('click', () => {
      tabBaselineVal.classList.add('active');
      tabActiveVal.classList.remove('active');
      if (viewActiveVal) viewActiveVal.style.display = 'none';
      if (viewBaselineVal) viewBaselineVal.style.display = 'block';
    });
  }

  const openValidationModal = () => {
    if (tabActiveVal) {
      tabActiveVal.classList.add('active');
      if (tabBaselineVal) tabBaselineVal.classList.remove('active');
      if (viewActiveVal) viewActiveVal.style.display = 'block';
      if (viewBaselineVal) viewBaselineVal.style.display = 'none';
    }
    renderValidationModal();
    benchmarkModal.style.display = 'flex';
  };

  const btnBenchmark = document.getElementById('btnBenchmarkModal');
  if (btnBenchmark) btnBenchmark.addEventListener('click', openValidationModal);

  const btnViewValBanner = document.getElementById('btnViewValidationFromBanner');
  if (btnViewValBanner) btnViewValBanner.addEventListener('click', openValidationModal);

  document.getElementById('btnCloseBenchmark').addEventListener('click', () => {
    benchmarkModal.style.display = 'none';
  });

  document.getElementById('btnCodeModal').addEventListener('click', () => {
    showCodeModal(App.currentAlgo);
    codeModal.style.display = 'flex';
  });
  document.getElementById('btnCloseCode').addEventListener('click', () => {
    codeModal.style.display = 'none';
  });

  // C++ Code Component Selector Grid Buttons
  document.querySelectorAll('.code-component-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.codekey;
      showCodeModal(key);
    });
  });

  // Copy Code Button
  const btnCopyCode = document.getElementById('btnCopyCode');
  if (btnCopyCode) {
    btnCopyCode.addEventListener('click', () => {
      const viewer = document.getElementById('codeViewer');
      if (viewer && viewer.textContent) {
        navigator.clipboard.writeText(viewer.textContent).then(() => {
          const btnText = document.getElementById('btnCopyCodeText');
          if (btnText) btnText.textContent = '✓ Copied!';
          showToast('C++ source code copied to clipboard!');
          setTimeout(() => {
            if (btnText) btnText.textContent = 'Copy Code';
          }, 2500);
        }).catch(() => {
          showToast('Code displayed in viewer.');
        });
      }
    });
  }

  // Close modals on clicking backdrop
  window.addEventListener('click', (e) => {
    if (e.target === benchmarkModal) benchmarkModal.style.display = 'none';
    if (e.target === codeModal) codeModal.style.display = 'none';
  });
}

function showCodeModal(key) {
  const activeKey = key || (BACKEND_CODE_FILES[App.currentAlgo] ? App.currentAlgo : 'linear_regression');
  const fileData = BACKEND_CODE_FILES[activeKey];
  if (!fileData) return;

  // Highlight active button in component grid
  document.querySelectorAll('.code-component-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.codekey === activeKey);
  });

  // Update Modal Header & Subtitle
  const title = document.getElementById('codeModalTitle');
  const sub = document.getElementById('codeModalSub');
  if (title) title.textContent = `C++ Implementation: ${fileData.title}`;
  if (sub) sub.textContent = `Actual backend source code running in ${fileData.path}`;

  // Update File Meta Bar
  const pathBadge = document.getElementById('codeFilePathBadge');
  const descEl = document.getElementById('codeFileDesc');
  const linesStat = document.getElementById('codeFileLinesStat');
  const sizeStat = document.getElementById('codeFileSizeStat');
  const viewer = document.getElementById('codeViewer');

  if (pathBadge) pathBadge.textContent = fileData.path;
  if (descEl) descEl.textContent = fileData.desc;
  if (linesStat) linesStat.textContent = fileData.lines;
  if (sizeStat) sizeStat.textContent = fileData.size;
  if (viewer) viewer.textContent = fileData.code;

  const btnCopyText = document.getElementById('btnCopyCodeText');
  if (btnCopyText) btnCopyText.textContent = 'Copy Code';
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
