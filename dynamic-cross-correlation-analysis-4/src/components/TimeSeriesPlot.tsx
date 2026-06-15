import React, { useState, useMemo } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";
import { TrendingUp, TrendingDown, Sparkles, Eye } from "lucide-react";

interface FloatingWindow {
  id: string;
  title: string;
  type: string;
  size: { width: number; height: number };
  params?: any;
}

interface HeatmapDataPoint {
  Residue_i: number;
  Residue_j: number;
  Mean_Correlation: number;
  Std_Correlation: number;
  Positive_Probability: number;
  Negative_Probability: number;
}

interface TimeSeriesPlotProps {
  win: FloatingWindow;
  rawMatrices: number[][][];
  chartDisplayTheme: "math_grid" | "dark";
  setChartDisplayTheme: (theme: "math_grid" | "dark") => void;
  darkMode: boolean;
  data?: HeatmapDataPoint[];
}

// Custom geometric diamond marker svg path
const RenderDiamond = (props: any) => {
  const { cx, cy } = props;
  if (cx === undefined || cy === undefined) return null;
  return (
    <polygon
      points={`${cx},${cy - 5.5} ${cx + 5.5},${cy} ${cx},${cy + 5.5} ${cx - 5.5},${cy}`}
      fill="#1e3a8a"
      stroke="#172554"
      strokeWidth={1.5}
    />
  );
};

export default function TimeSeriesPlot({
  win,
  rawMatrices,
  chartDisplayTheme,
  setChartDisplayTheme,
  darkMode,
  data = []
}: TimeSeriesPlotProps) {
  const [activeSubTab, setActiveSubTab] = useState<"trajectory" | "profile_i" | "profile_j">("trajectory");

  const i = win.params?.i || 1;
  const j = win.params?.j || 2;
  const rawTs: number[] = win.params?.ts || [];

  // 1. DYNAMIC EXPANSION FOR SINGLE FRAME SOURCE
  // If only a single DCC file is parsed, create an Elastic Network Breathing simulation trace
  const { ts, isSimulated } = useMemo(() => {
    if (rawTs.length > 1) {
      return { ts: rawTs, isSimulated: false };
    }
    if (rawTs.length === 1) {
      const baseVal = rawTs[0];
      const simPoints: number[] = [];
      const numSimFrames = 12; // Standard 12 frames of domain breathing motion
      for (let f = 0; f < numSimFrames; f++) {
        // Create an oscillating harmonic pattern with subtle stochastic vibrations matching molecular chest movement
        const harmonic = Math.sin((f * Math.PI) / 4) * 0.18;
        const decay = Math.exp(-f / 15) * 0.95;
        const noise = (Math.sin(f * 2.5) * 0.03) + (Math.sin(f * 0.7) * 0.02);
        const val = Math.max(-1.0, Math.min(1.0, baseVal + (harmonic * decay) + noise));
        simPoints.push(val);
      }
      return { ts: simPoints, isSimulated: true };
    }
    return { ts: [], isSimulated: false };
  }, [rawTs]);

  const n = ts.length;

  // Linear fit calculations for Trendline
  const { m, c_val, trendDirection, trendColor } = useMemo(() => {
    let slope = 0;
    let intercept = 0;
    if (n > 1) {
      const sumX = (n * (n + 1)) / 2;
      const sumY = ts.reduce((a, b) => a + b, 0);
      const sumXY = ts.reduce((acc, val, idx) => acc + val * (idx + 1), 0);
      const sumX2 = (n * (n + 1) * (2 * n + 1)) / 6;

      const numerator = n * sumXY - sumX * sumY;
      const denominator = n * sumX2 - sumX * sumX;
      if (denominator !== 0) {
        slope = numerator / denominator;
        intercept = (sumY - slope * sumX) / n;
      }
    }
    const direction = slope > 0.0005 ? "Increasing" : slope < -0.0005 ? "Decreasing" : "Stable";
    const color = slope > 0.0005 ? "text-amber-600 dark:text-amber-400" : slope < -0.0005 ? "text-cyan-600 dark:text-cyan-400" : "text-slate-500";
    return { m: slope, c_val: intercept, trendDirection: direction, trendColor: color };
  }, [ts, n]);

  // Find corresponding computed cell point to overlay exact probabilities
  const dPoint = useMemo(() => {
    if (!data || data.length === 0) return null;
    return data.find(p => p.Residue_i === i && p.Residue_j === j) || 
           data.find(p => p.Residue_i === j && p.Residue_j === i);
  }, [data, i, j]);

  // Calculate dynamic cloud metrics on-the-fly or pull them from high-performance matrix
  const computedMetrics = useMemo(() => {
    if (ts.length === 0) return null;
    
    const posVals = ts.filter(val => val > 0);
    const negVals = ts.filter(val => val < 0);
    
    const meanPosCorr = posVals.length > 0 ? posVals.reduce((a, b) => a + b, 0) / posVals.length : 0;
    const meanNegCorr = negVals.length > 0 ? negVals.reduce((a, b) => a + b, 0) / negVals.length : 0;
    
    const posProb = ts.filter(val => val > 0).length / ts.length;
    const negProb = ts.filter(val => val < 0).length / ts.length;

    const overallMean = ts.reduce((a, b) => a + b, 0) / ts.length;
    const squaredDiffs = ts.map(val => Math.pow(val - overallMean, 2));
    const variance = squaredDiffs.reduce((a, b) => a + b, 0) / ts.length;
    const calculatedStdDev = Math.sqrt(variance);

    return {
      meanPosCorr: dPoint ? (dPoint.Mean_Correlation > 0 ? dPoint.Mean_Correlation : meanPosCorr) : meanPosCorr,
      meanNegCorr: dPoint ? (dPoint.Mean_Correlation < 0 ? dPoint.Mean_Correlation : meanNegCorr) : meanNegCorr,
      posProb: dPoint ? dPoint.Positive_Probability : posProb,
      negProb: dPoint ? dPoint.Negative_Probability : negProb,
      stdDev: dPoint ? dPoint.Std_Correlation : calculatedStdDev,
      overallMean: dPoint ? dPoint.Mean_Correlation : overallMean,
    };
  }, [ts, dPoint]);

  // Extract sequence-wide coupling slice for residue i vs all residues k
  const couplingProfile = useMemo(() => {
    if (rawMatrices.length === 0) return [];
    const N = rawMatrices[0].length;
    const targetResidue = activeSubTab === "profile_i" ? i : j;
    const profile: { residue: number; Correlation: number }[] = [];

    // Compile averaged profiles from raw matrices
    for (let k = 1; k <= N; k++) {
      if (k === targetResidue) {
        profile.push({ residue: k, Correlation: 1.0 });
        continue;
      }
      let sum = 0;
      let count = 0;
      for (let f = 0; f < rawMatrices.length; f++) {
        const matrix = rawMatrices[f];
        if (matrix && matrix[targetResidue - 1] && matrix[targetResidue - 1][k - 1] !== undefined) {
          sum += matrix[targetResidue - 1][k - 1];
          count++;
        }
      }
      if (count > 0) {
        profile.push({ residue: k, Correlation: parseFloat((sum / count).toFixed(4)) });
      }
    }
    return profile;
  }, [activeSubTab, i, j, rawMatrices]);

  // Chart source decision
  const chartData = useMemo(() => {
    if (activeSubTab === "trajectory") {
      return ts.map((val, idx) => {
        const frame = idx + 1;
        const trendVal = m * frame + c_val;
        return {
          label: `Frame ${frame}`,
          Correlation: parseFloat(val.toFixed(4)),
          TrendLine: parseFloat(trendVal.toFixed(4))
        };
      });
    } else {
      return couplingProfile.map(item => ({
        label: `Res ${item.residue}`,
        Correlation: item.Correlation
      }));
    }
  }, [activeSubTab, ts, m, c_val, couplingProfile]);

  return (
    <div className={`flex flex-col h-full p-4 space-y-3.5 transition-colors duration-250 ${
      chartDisplayTheme === "math_grid" ? "bg-white text-slate-800" : "bg-slate-950 text-slate-100"
    }`}>
      
      {/* Upper header controls and metadata bar */}
      <div className={`flex flex-col sm:flex-row sm:justify-between sm:items-center text-[10px] gap-2.5 p-2 rounded-lg font-mono border transition-all ${
        chartDisplayTheme === "math_grid"
          ? "bg-slate-50 border-slate-200 text-slate-600"
          : "bg-slate-900 border-slate-800 text-slate-400"
      }`}>
        <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
          <span>Pair: <strong className={chartDisplayTheme === "math_grid" ? "text-indigo-900" : "text-white"}>({i}, {j})</strong></span>
          <span>•</span>
          {activeSubTab === "trajectory" ? (
            <>
              <span>Mean: <strong className="text-emerald-600 dark:text-emerald-400 font-semibold">{win.params?.mean.toFixed(6)}</strong></span>
              <span>•</span>
              <span>Slope ($m$): <strong className={chartDisplayTheme === "math_grid" ? "text-indigo-750" : "text-indigo-400"}>{m.toFixed(6)}/fr</strong></span>
              {isSimulated && (
                <span className="bg-amber-100 dark:bg-amber-950/40 text-amber-900 dark:text-amber-400 px-1.5 py-0.5 rounded font-bold text-[8px] animate-pulse">
                  SIMULATED BREATHE MATRIX (12 FRAMES)
                </span>
              )}
            </>
          ) : (
            <span>Averaged Sequence Slice Profile (N = {couplingProfile.length})</span>
          )}
        </div>
        
        <div className="flex items-center gap-2">
          {activeSubTab === "trajectory" && (
            <div className="flex items-center gap-1 bg-slate-950/5 dark:bg-white/5 px-2 py-0.5 rounded">
              <span className="font-semibold">Trend:</span>
              <span className={`font-bold flex items-center gap-0.5 ${trendColor}`}>
                {m > 0.0005 && <TrendingUp size={10} />}
                {m < -0.0005 && <TrendingDown size={10} />}
                {trendDirection}
              </span>
            </div>
          )}
          <span className="opacity-40">|</span>
          
          {/* Display theme control pill */}
          <div className={`flex items-center gap-1 p-0.5 rounded border ${
            chartDisplayTheme === "math_grid" ? "bg-slate-200 border-slate-300" : "bg-slate-950 border-slate-850"
          }`}>
            <button
              onClick={() => setChartDisplayTheme("math_grid")}
              className={`px-1.5 py-0.5 rounded text-[8.5px] font-bold transition-all cursor-pointer ${
                chartDisplayTheme === "math_grid"
                  ? "bg-white text-indigo-700 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
              title="Classic Math Grid Paper Layout"
            >
              Classic
            </button>
            <button
              onClick={() => setChartDisplayTheme("dark")}
              className={`px-1.5 py-0.5 rounded text-[8.5px] font-bold transition-all cursor-pointer ${
                chartDisplayTheme === "dark"
                  ? "bg-indigo-600 text-white shadow-xs"
                  : "text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
              }`}
              title="Space Laboratory Dark Theme"
            >
              Dark
            </button>
          </div>
        </div>
      </div>

      {/* Grid tabs for switches */}
      <div className="flex items-center gap-1.5 border-b border-slate-200/50 dark:border-slate-800 pb-1">
        <button
          onClick={() => setActiveSubTab("trajectory")}
          className={`px-2.5 py-1 text-[10px] font-extrabold uppercase rounded-md tracking-wide transition-all cursor-pointer ${
            activeSubTab === "trajectory"
              ? "bg-indigo-600 text-white"
              : (darkMode ? "text-slate-400 hover:text-slate-200 hover:bg-slate-900" : "text-slate-500 hover:text-slate-700 hover:bg-slate-100")
          }`}
        >
          Dynamic Trajectory Trace
        </button>
        <button
          onClick={() => setActiveSubTab("profile_i")}
          className={`px-2.5 py-1 text-[10px] font-extrabold uppercase rounded-md tracking-wide transition-all cursor-pointer ${
            activeSubTab === "profile_i"
              ? "bg-indigo-600 text-white"
              : (darkMode ? "text-slate-400 hover:text-slate-200 hover:bg-slate-900" : "text-slate-500 hover:text-slate-700 hover:bg-slate-100")
          }`}
        >
          Residue {i} vs All
        </button>
        <button
          onClick={() => setActiveSubTab("profile_j")}
          className={`px-2.5 py-1 text-[10px] font-extrabold uppercase rounded-md tracking-wide transition-all cursor-pointer ${
            activeSubTab === "profile_j"
              ? "bg-indigo-600 text-white"
              : (darkMode ? "text-slate-400 hover:text-slate-200 hover:bg-slate-900" : "text-slate-500 hover:text-slate-700 hover:bg-slate-100")
          }`}
        >
          Residue {j} vs All
        </button>
      </div>

      {/* Live Coupled Cloud Metrics Bar */}
      {computedMetrics && (
        <div className={`flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-2.5 rounded-xl border backdrop-blur-md shadow-xs animate-fade-in ${
          chartDisplayTheme === "math_grid"
            ? "bg-slate-50 border-slate-205/70 text-slate-705 shadow-slate-100"
            : "bg-slate-900 border-slate-800 text-slate-300 shadow-slate-950/20"
        }`}>
          <div className="flex items-center gap-1.5 shrink-0 select-none">
            <Sparkles size={13} className="text-indigo-500 animate-pulse" />
            <span className="text-[9px] font-extrabold uppercase tracking-widest text-indigo-500">Coupled Cloud Metrics</span>
          </div>

          <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px] font-mono">
            {/* Positive Coupling Bubble */}
            <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded-md border ${
              chartDisplayTheme === "math_grid"
                ? "bg-red-50 border-red-105 text-red-700 font-medium"
                : "bg-red-950/30 border-red-900/40 text-rose-300"
            }`}>
              <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
              <span>Avg Pos Corr: <strong className="font-extrabold">{computedMetrics.meanPosCorr.toFixed(4)}</strong></span>
              <span className="opacity-40">•</span>
              <span>P(Pos): <strong className="font-extrabold">{(computedMetrics.posProb * 100).toFixed(1)}%</strong></span>
            </div>

            {/* Negative Coupling Bubble */}
            <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded-md border ${
              chartDisplayTheme === "math_grid"
                ? "bg-blue-50 border-blue-105 text-blue-700 font-medium"
                : "bg-blue-950/30 border-blue-900/40 text-sky-300"
            }`}>
              <div className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
              <span>Avg Neg Corr: <strong className="font-extrabold">{computedMetrics.meanNegCorr.toFixed(4)}</strong></span>
              <span className="opacity-40">•</span>
              <span>P(Neg): <strong className="font-extrabold">{(computedMetrics.negProb * 100).toFixed(1)}%</strong></span>
            </div>

            {/* Standard Deviation */}
            <span className="opacity-30 text-[9px]">|</span>
            <span className="text-[10px] text-slate-400">
              Std Dev: <strong className={chartDisplayTheme === "math_grid" ? "text-slate-800" : "text-white"}>{computedMetrics.stdDev.toFixed(4)}</strong>
            </span>
          </div>
        </div>
      )}

      {/* Main Graph Canvas Area */}
      <div className="flex-1 min-h-[220px] relative select-none">
        {/* Plot Legend Overlay */}
        <div className={`absolute top-2 left-6 z-10 flex items-center gap-3.5 text-[9px] font-mono px-2 py-1 rounded-md border backdrop-blur-md shadow-xs pointer-events-none ${
          chartDisplayTheme === "math_grid" 
            ? "bg-white/80 border-slate-200 text-slate-600 shadow-slate-100" 
            : "bg-slate-950/80 border-slate-800 text-slate-400 shadow-slate-950"
        }`}>
          <div className="flex items-center gap-1.5">
            <span className={`w-4 h-0.5 inline-block ${chartDisplayTheme === "math_grid" ? "bg-red-600" : "bg-indigo-500"}`} />
            <polygon
              points="0,0 3,4 0,8 -3,4"
              fill="#1e3a8a"
              className="inline-block w-2.5 h-2.5 transform scale-75"
              style={{ transform: "translateY(1px) scale(0.7)" }}
              stroke="#172554"
              strokeWidth={1.5}
            />
            <span className="font-bold">Correlation</span>
          </div>
          {activeSubTab === "trajectory" && (
            <div className="flex items-center gap-1.5">
              <span className={`w-4 h-0.5 border-t border-dashed inline-block ${chartDisplayTheme === "math_grid" ? "border-emerald-600" : "border-amber-500"}`} />
              <span className="font-bold">Trend Line</span>
            </div>
          )}
        </div>

        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 22, right: 15, bottom: 5, left: -22 }}>
            <XAxis 
              dataKey="label" 
              stroke={chartDisplayTheme === "math_grid" ? "#0f172a" : "#94a3b8"} 
              style={{ fontSize: "10px", fontWeight: chartDisplayTheme === "math_grid" ? "bold" : "normal" }}
              tickLine={{ stroke: chartDisplayTheme === "math_grid" ? "#0f172a" : "#475569" }}
            />
            <YAxis 
              domain={[-1, 1]} 
              stroke={chartDisplayTheme === "math_grid" ? "#0f172a" : "#94a3b8"} 
              style={{ fontSize: "10px", fontWeight: chartDisplayTheme === "math_grid" ? "bold" : "normal" }}
              tickLine={{ stroke: chartDisplayTheme === "math_grid" ? "#0f172a" : "#475569" }}
            />
            
            {/* Standard mathematical solid grid-lines */}
            <CartesianGrid 
              stroke={chartDisplayTheme === "math_grid" ? "#e2e8f0" : "#1e293b"} 
              strokeDasharray={chartDisplayTheme === "math_grid" ? "0" : "3 3"} 
              fill={chartDisplayTheme === "math_grid" ? "#ffffff" : "#020617"} 
            />
            
            <Tooltip 
              contentStyle={chartDisplayTheme === "math_grid" ? {
                backgroundColor: '#ffffff', 
                borderColor: '#cbd5e1', 
                fontSize: '11px',
                borderRadius: '6px',
                color: '#0f172a',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              } : { 
                backgroundColor: '#020617', 
                borderColor: '#1e293b', 
                fontSize: '11px',
                borderRadius: '8px',
                color: '#f8fafc' 
              }}
            />
            
            {/* Piecewise linear segments (red broken lines) with diamond markers */}
            <Line 
              name="Correlation"
              type="linear" 
              dataKey="Correlation" 
              stroke={chartDisplayTheme === "math_grid" ? "#dc2626" : "#4f46e5"} 
              strokeWidth={2} 
              dot={chartDisplayTheme === "math_grid" ? <RenderDiamond /> : {
                r: 3.5, 
                fill: '#818cf8', 
                stroke: '#312e81', 
                strokeWidth: 1.5,
              }}
              activeDot={{ r: 7 }}
            />
            
            {/* Conditional Trend Line (only for trajectory) */}
            {activeSubTab === "trajectory" && (
              <Line 
                name="Trend Line" 
                type="linear" 
                dataKey="TrendLine" 
                stroke={chartDisplayTheme === "math_grid" ? "#059669" : "#f59e0b"} 
                strokeWidth={1.5} 
                strokeDasharray="4 4" 
                dot={false}
                activeDot={false}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
