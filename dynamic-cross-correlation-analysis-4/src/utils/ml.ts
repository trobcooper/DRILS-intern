/**
 * Machine Learning Regression Models in TypeScript
 * Implements: Linear Regression (OLS via Gradient Descent),
 * Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor.
 */

export interface MLMetrics {
  r2: number;
  mae: number;
  rmse: number;
}

export interface PredictionResult {
  actual: number;
  predicted: number;
  residual: number;
}

export interface FeatureImportance {
  featureName: string;
  importance: number;
}

// Helper: Calculate statistics
export function calculateMetrics(actual: number[], predicted: number[]): MLMetrics {
  const n = actual.length;
  if (n === 0) return { r2: 0, mae: 0, rmse: 0 };

  let sumActual = 0;
  let sumPredicted = 0;
  for (let i = 0; i < n; i++) {
    sumActual += actual[i];
    sumPredicted += predicted[i];
  }
  const meanActual = sumActual / n;

  let sst = 0; // Total sum of squares
  let sse = 0; // Residual sum of squares
  let maeSum = 0;

  for (let i = 0; i < n; i++) {
    const diff = actual[i] - predicted[i];
    sse += diff * diff;
    sst += (actual[i] - meanActual) * (actual[i] - meanActual);
    maeSum += Math.abs(diff);
  }

  const r2 = sst === 0 ? 0 : 1 - sse / sst;
  const mae = maeSum / n;
  const rmse = Math.sqrt(sse / n);

  return { r2, mae, rmse };
}

// 1. LINEAR REGRESSION MODEL
export class LinearRegression {
  private weights: number[] = [];
  private bias: number = 0;

  public train(X: number[][], y: number[], epochs: number = 500, lr: number = 0.01): void {
    const numSamples = X.length;
    if (numSamples === 0) return;
    const numFeatures = X[0].length;

    // Feature scaling (Standardization) to make gradient descent highly stable
    const means: number[] = Array(numFeatures).fill(0);
    const stds: number[] = Array(numFeatures).fill(1);

    for (let j = 0; j < numFeatures; j++) {
      let sum = 0;
      for (let i = 0; i < numSamples; i++) sum += X[i][j];
      means[j] = sum / numSamples;

      let varianceSum = 0;
      for (let i = 0; i < numSamples; i++) {
        const diff = X[i][j] - means[j];
        varianceSum += diff * diff;
      }
      stds[j] = Math.sqrt(varianceSum / numSamples) || 1e-9;
    }

    const scaledX = X.map(row =>
      row.map((val, j) => (val - means[j]) / stds[j])
    );

    this.weights = Array(numFeatures).fill(0).map(() => (Math.random() - 0.5) * 0.1);
    this.bias = 0;

    // Gradient descent
    for (let epoch = 0; epoch < epochs; epoch++) {
      let dw = Array(numFeatures).fill(0);
      let db = 0;

      for (let i = 0; i < numSamples; i++) {
        let pred = this.bias;
        for (let j = 0; j < numFeatures; j++) {
          pred += scaledX[i][j] * this.weights[j];
        }

        const error = pred - y[i];
        db += error;
        for (let j = 0; j < numFeatures; j++) {
          dw[j] += error * scaledX[i][j];
        }
      }

      // Update weights
      this.bias -= (lr * db) / numSamples;
      for (let j = 0; j < numFeatures; j++) {
        this.weights[j] -= (lr * dw[j]) / numSamples;
      }
    }

    // Convert weights back to original unscale factors so we can evaluate raw inputs
    for (let j = 0; j < numFeatures; j++) {
      this.bias -= (this.weights[j] * means[j]) / stds[j];
      this.weights[j] = this.weights[j] / stds[j];
    }
  }

  public predict(X: number[][]): number[] {
    return X.map(row => {
      let val = this.bias;
      for (let j = 0; j < row.length; j++) {
        val += row[j] * (this.weights[j] || 0);
      }
      return val;
    });
  }

  public getFeatureImportances(featureNames: string[]): FeatureImportance[] {
    // Feature importance for linear regression is absolute weight
    const total = this.weights.reduce((sum, w) => sum + Math.abs(w), 0) || 1e-9;
    return this.weights.map((w, idx) => ({
      featureName: featureNames[idx] || `Feature ${idx + 1}`,
      importance: Math.abs(w) / total
    })).sort((a, b) => b.importance - a.importance);
  }
}

// 2. DECISION TREE REGRESSOR BASE
class TreeNode {
  public featureIdx: number = -1;
  public threshold: number = 0;
  public left: TreeNode | null = null;
  public right: TreeNode | null = null;
  public value: number = 0; // Leaf value (average y)
  public isLeaf: boolean = false;
  public gain: number = 0;
}

export class DecisionTreeRegressor {
  private root: TreeNode | null = null;
  private maxDepth: number;
  private minSamplesSplit: number;

  constructor(maxDepth: number = 6, minSamplesSplit: number = 5) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
  }

  public train(X: number[][], y: number[], maxFeaturesCount?: number): void {
    this.root = this.buildTree(X, y, 0, maxFeaturesCount);
  }

  private buildTree(X: number[][], y: number[], depth: number, maxFeaturesCount?: number): TreeNode {
    const node = new TreeNode();
    const numSamples = X.length;

    // Base cases
    if (numSamples === 0) {
      node.isLeaf = true;
      node.value = 0;
      return node;
    }

    const sumY = y.reduce((s, curr) => s + curr, 0);
    const meanY = sumY / numSamples;

    const allSame = y.every(val => Math.abs(val - y[0]) < 1e-9);

    if (depth >= this.maxDepth || numSamples < this.minSamplesSplit || allSame) {
      node.isLeaf = true;
      node.value = meanY;
      return node;
    }

    // Find best split
    const split = this.findBestSplit(X, y, maxFeaturesCount);
    if (!split || split.gain <= 1e-9) {
      node.isLeaf = true;
      node.value = meanY;
      return node;
    }

    node.featureIdx = split.featureIdx;
    node.threshold = split.threshold;
    node.gain = split.gain;

    // Split datasets
    const leftX: number[][] = [];
    const leftY: number[] = [];
    const rightX: number[][] = [];
    const rightY: number[] = [];

    for (let i = 0; i < numSamples; i++) {
      if (X[i][split.featureIdx] <= split.threshold) {
        leftX.push(X[i]);
        leftY.push(y[i]);
      } else {
        rightX.push(X[i]);
        rightY.push(y[i]);
      }
    }

    node.left = this.buildTree(leftX, leftY, depth + 1, maxFeaturesCount);
    node.right = this.buildTree(rightX, rightY, depth + 1, maxFeaturesCount);
    return node;
  }

  private findBestSplit(X: number[][], y: number[], maxFeaturesCount?: number) {
    const numSamples = X.length;
    const numFeatures = X[0].length;

    let bestFeatureIdx = -1;
    let bestThreshold = 0;
    let bestLossReduction = -1;

    const initialSSE = this.calculateSSE(y);

    // Feature Bagging if requested (Random Forest feature subset)
    let featuresToCheck = Array.from({ length: numFeatures }, (_, i) => i);
    if (maxFeaturesCount && maxFeaturesCount < numFeatures) {
      featuresToCheck.sort(() => Math.random() - 0.5);
      featuresToCheck = featuresToCheck.slice(0, maxFeaturesCount);
    }

    for (const fIdx of featuresToCheck) {
      // Find candidate thresholds (e.g., unique sorted values)
      const values = X.map(row => row[fIdx]);
      const uniqueVals = Array.from(new Set(values)).sort((a, b) => a - b);

      // Check midpoints as candidate thresholds
      for (let i = 0; i < uniqueVals.length - 1; i++) {
        const threshold = (uniqueVals[i] + uniqueVals[i + 1]) / 2;

        const leftY: number[] = [];
        const rightY: number[] = [];

        for (let s = 0; s < numSamples; s++) {
          if (X[s][fIdx] <= threshold) {
            leftY.push(y[s]);
          } else {
            rightY.push(y[s]);
          }
        }

        if (leftY.length === 0 || rightY.length === 0) continue;

        const leftSSE = this.calculateSSE(leftY);
        const rightSSE = this.calculateSSE(rightY);
        const splitSSE = leftSSE + rightSSE;
        const lossReduction = initialSSE - splitSSE;

        if (lossReduction > bestLossReduction) {
          bestLossReduction = lossReduction;
          bestFeatureIdx = fIdx;
          bestThreshold = threshold;
        }
      }
    }

    if (bestFeatureIdx === -1) return null;

    return {
      featureIdx: bestFeatureIdx,
      threshold: bestThreshold,
      gain: bestLossReduction
    };
  }

  private calculateSSE(y: number[]): number {
    const len = y.length;
    if (len === 0) return 0;
    const mean = y.reduce((s, curr) => s + curr, 0) / len;
    let sum = 0;
    for (let i = 0; i < len; i++) {
      const diff = y[i] - mean;
      sum += diff * diff;
    }
    return sum;
  }

  public predictRow(row: number[], node: TreeNode | null = this.root): number {
    if (!node) return 0;
    if (node.isLeaf) return node.value;

    if (row[node.featureIdx] <= node.threshold) {
      return this.predictRow(row, node.left);
    } else {
      return this.predictRow(row, node.right);
    }
  }

  public predict(X: number[][]): number[] {
    return X.map(row => this.predictRow(row));
  }

  // Accumulate feature split gains to calculate feature importance
  public accumulateImportance(importances: number[], node: TreeNode | null = this.root): void {
    if (!node || node.isLeaf) return;
    importances[node.featureIdx] = (importances[node.featureIdx] || 0) + node.gain;
    this.accumulateImportance(importances, node.left);
    this.accumulateImportance(importances, node.right);
  }
}

// 3. RANDOM FOREST REGRESSOR
export class RandomForestRegressor {
  private trees: DecisionTreeRegressor[] = [];
  private numTrees: number;
  private maxDepth: number;

  constructor(numTrees: number = 15, maxDepth: number = 6) {
    this.numTrees = numTrees;
    this.maxDepth = maxDepth;
  }

  public train(X: number[][], y: number[]): void {
    this.trees = [];
    const n = X.length;
    if (n === 0) return;
    const numFeatures = X[0].length;
    const maxFeaturesCount = Math.max(1, Math.floor(Math.sqrt(numFeatures)) || 1);

    for (let t = 0; t < this.numTrees; t++) {
      // Bootstrap sampling (bagging) with replacement
      const bootX: number[][] = [];
      const bootY: number[] = [];
      for (let i = 0; i < n; i++) {
        const randIdx = Math.floor(Math.random() * n);
        bootX.push(X[randIdx]);
        bootY.push(y[randIdx]);
      }

      const tree = new DecisionTreeRegressor(this.maxDepth, 3);
      tree.train(bootX, bootY, maxFeaturesCount);
      this.trees.push(tree);
    }
  }

  public predict(X: number[][]): number[] {
    const numSamples = X.length;
    const preds = Array(numSamples).fill(0);

    for (let i = 0; i < numSamples; i++) {
      let sum = 0;
      for (const tree of this.trees) {
        sum += tree.predictRow(X[i]);
      }
      preds[i] = sum / this.trees.length;
    }

    return preds;
  }

  public getFeatureImportances(featureNames: string[], numFeatures: number): FeatureImportance[] {
    const importances = Array(numFeatures).fill(0);
    for (const tree of this.trees) {
      tree.accumulateImportance(importances);
    }

    const total = importances.reduce((s, val) => s + val, 0) || 1e-9;
    return importances.map((imp, idx) => ({
      featureName: featureNames[idx] || `Feature ${idx + 1}`,
      importance: imp / total
    })).sort((a, b) => b.importance - a.importance);
  }
}

// 4. GRADIENT BOOSTING REGRESSOR
export class GradientBoostingRegressor {
  private trees: DecisionTreeRegressor[] = [];
  private numEstimators: number;
  private learningRate: number;
  private initialPrediction: number = 0;
  private maxDepth: number;

  constructor(numEstimators: number = 15, learningRate: number = 0.1, maxDepth: number = 4) {
    this.numEstimators = numEstimators;
    this.learningRate = learningRate;
    this.maxDepth = maxDepth;
  }

  public train(X: number[][], y: number[]): void {
    this.trees = [];
    const n = X.length;
    if (n === 0) return;

    // Initialize prediction with the average of y
    const sumY = y.reduce((s, val) => s + val, 0);
    this.initialPrediction = sumY / n;

    const currentPredictions = Array(n).fill(this.initialPrediction);

    for (let t = 0; t < this.numEstimators; t++) {
      // Calculate pseudo-residuals (residuals = actual - current_predictions)
      const residuals = y.map((val, idx) => val - currentPredictions[idx]);

      const tree = new DecisionTreeRegressor(this.maxDepth, 4);
      tree.train(X, residuals);

      // Update current predictions
      const stepPredictions = tree.predict(X);
      for (let i = 0; i < n; i++) {
        currentPredictions[i] += this.learningRate * stepPredictions[i];
      }

      this.trees.push(tree);
    }
  }

  public predict(X: number[][]): number[] {
    return X.map(row => {
      let val = this.initialPrediction;
      for (const tree of this.trees) {
        val += this.learningRate * tree.predictRow(row);
      }
      return val;
    });
  }

  public getFeatureImportances(featureNames: string[], numFeatures: number): FeatureImportance[] {
    const importances = Array(numFeatures).fill(0);
    for (const tree of this.trees) {
      tree.accumulateImportance(importances);
    }

    const total = importances.reduce((s, val) => s + val, 0) || 1e-9;
    return importances.map((imp, idx) => ({
      featureName: featureNames[idx] || `Feature ${idx + 1}`,
      importance: imp / total
    })).sort((a, b) => b.importance - a.importance);
  }
}
