import React, { useState, useEffect, useMemo, useRef } from "react";
import { 
  FolderOpen, Play, Download, LayoutDashboard, Binary, 
  HelpCircle, Settings, Sun, Moon, Sparkles, Cpu, Maximize,
  Activity, Calendar, Layers, TableProperties, HelpCircle as HelpIcon,
  Minimize, Scaling, Trash2, Sliders, AlertCircle, TrendingUp, TrendingDown
} from "lucide-react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";

import WindowFrame from "./components/WindowFrame";
import InteractiveHeatmap from "./components/InteractiveHeatmap";
import AIPredictionPanel from "./components/AIPredictionPanel";
import TimeSeriesPlot from "./components/TimeSeriesPlot";
import DynamicDomainsPanel from "./components/DynamicDomainsPanel";

// High-fidelity biophysical sample generator representing a protein with 100 residues
function generateDummyDCCData(numFiles: number = 20, residueCount: number = 100): number[][][] {
  const data: number[][][] = [];
  
  for (let f = 0; f < numFiles; f++) {
    const matrix: number[][] = [];
    const phaseShift = Math.sin((f * Math.PI) / 4) * 0.12; // Domain swing period
    
    for (let i = 0; i < residueCount; i++) {
      matrix[i] = [];
      for (let j = 0; j < residueCount; j++) {
        if (i === j) {
          matrix[i][j] = 1.0;
          continue;
        }
        
        let val = 0.0;
        const dist = Math.abs(i - j);
        
        // Alpha Helix turns (i to i+4) are highly rigid & positive
        if (dist === 4) {
          val += 0.52 + Math.random() * 0.15;
        } else if (dist === 3 || dist === 5) {
          val += 0.28 + Math.random() * 0.1;
        }
        
        // Sequence proximity mechanical connection exponential decay
        val += 0.75 / (dist * 0.12 + 1.0);
        
        // Dynamic Domain breathing hinges
        // Domain A residues 15-45, Domain B residues 55-90
        const inDomainA_i = i >= 15 && i <= 45;
        const inDomainA_j = j >= 15 && j <= 45;
        const inDomainB_i = i >= 55 && i <= 90;
        const inDomainB_j = j >= 55 && j <= 90;
        
        if (inDomainA_i && inDomainA_j) {
          val += 0.25 + phaseShift; 
        } else if (inDomainB_i && inDomainB_j) {
          val += 0.28 - phaseShift;
        } else if ((inDomainA_i && inDomainB_j) || (inDomainB_i && inDomainA_j)) {
          // Out of phase domain breathing negative correlations
          val -= 0.48 + phaseShift * 0.6; 
        }
        
        // Stochastic residue vibrations noise
        val += (Math.random() - 0.5) * 0.18;
        
        // Clamp to physics boundaries [-1.0, 1.0]
        matrix[i][j] = Math.max(-1.0, Math.min(1.0, val));
      }
    }
    data.push(matrix);
  }
  return data;
}

interface HeatmapDataPoint {
  Residue_i: number;
  Residue_j: number;
  Mean_Correlation: number;
  Std_Correlation: number;
  Positive_Probability: number;
  Negative_Probability: number;
}

interface FloatingWindow {
  id: string;
  title: string;
  type: "dashboard_correlation" | "dashboard_probability" | "time_series" | "interactive_correlation" | "interactive_probability" | "single_plot";
  params?: any; // e.g. { i, j, ts, mode, threshold, plot_type }
  position: { x: number; y: number };
  size: { width: number; height: number };
}

export default function App() {
  const [viewState, setViewState] = useState<"welcome" | "workbench">("welcome");
  const [darkMode, setDarkMode] = useState<boolean>(true);

  // Status and logs progress trackers
  const [statusText, setStatusText] = useState<string>("System initialized. Ready.");
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisProgress, setAnalysisProgress] = useState<number>(0);

  // Analysis Parameters (States match Python exactly)
  const [i_min, setIMin] = useState<number>(1);
  const [i_max, setIMax] = useState<number>(100);
  const [j_min, setJMin] = useState<number>(1);
  const [j_max, setJMax] = useState<number>(100);
  const [res_dist, setResDist] = useState<number>(2);

  const [analysis_mode, setAnalysisMode] = useState<"Probability" | "Correlation">("Probability");

  // Basic (range-based) thresholds
  const [pmin, setPMin] = useState<number>(0.6);
  const [pmax, setPMax] = useState<number>(0.9);
  const [cmin, setCMin] = useState<number>(0.5);
  const [cmax, setCMax] = useState<number>(1.0);

  // Advanced (cutoff-based) thresholds
  const [p_pos_cut, setPPosCut] = useState<number>(0.6);
  const [p_neg_cut, setPNegCut] = useState<number>(0.6);
  const [c_pos_cut, setCPosCut] = useState<number>(0.9);
  const [c_neg_cut, setCNegCut] = useState<number>(-0.5);

  // Simulated internal disk files
  const [numSampleFiles, setNumSampleFiles] = useState<number>(20);
  const [rawMatrices, setRawMatrices] = useState<number[][][]>([]);
  const [loadedFilesName, setLoadedFilesName] = useState<string>("Simulated protein trajectories (Default)");

  // Correlation analysis outputs
  const [df_all, setDfAll] = useState<HeatmapDataPoint[]>([]);

  // Main UI Visualizer Tabs
  const [activeTab, setActiveTab] = useState<"dashboard" | "interactive_corr" | "interactive_prob" | "prediction" | "domains">("dashboard");

  // Drag-and-Drop multi-window state manager
  const [openWindows, setOpenWindows] = useState<FloatingWindow[]>([]);
  const [windowFocusOrder, setWindowFocusOrder] = useState<string[]>([]);
  const [chartDisplayTheme, setChartDisplayTheme] = useState<"math_grid" | "dark">("math_grid");

  // Seed default dummy data on mount
  useEffect(() => {
    const defaultData = generateDummyDCCData(20, 100);
    setRawMatrices(defaultData);
  }, []);

  // Auto-run analysis when rawMatrices are initialized/loaded on mount
  useEffect(() => {
    if (rawMatrices.length > 0 && df_all.length === 0) {
      runAnalysis({ iMin: i_min, iMax: i_max, jMin: j_min, jMax: j_max });
    }
  }, [rawMatrices, df_all]);

  // Multi-file drag upload handler
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setStatusText(`Staging ${files.length} coordinate trace files...`);
    setAnalysisProgress(10);

    const matricesList: number[][][] = [];
    let filesParsed = 0;

    for (let fIdx = 0; fIdx < files.length; fIdx++) {
      const file = files[fIdx];
      const reader = new FileReader();

      reader.onload = (event) => {
        const text = event.target?.result as string;
        const lines = text.trim().split(/\r?\n/);
        const fileMatrix: number[][] = [];

        for (let i = 0; i < lines.length; i++) {
          const rowVals = lines[i].trim().split(/\s+/).map(val => parseFloat(val));
          if (rowVals.length > 0 && !rowVals.some(isNaN)) {
            fileMatrix.push(rowVals);
          }
        }

        if (fileMatrix.length > 0) {
          matricesList.push(fileMatrix);
        }

        filesParsed++;
        setAnalysisProgress(Math.min(95, Math.round((filesParsed / files.length) * 100)));

        if (filesParsed === files.length) {
          if (matricesList.length === 0) {
            setStatusText("Error parsing uploaded files. Check formatting.");
            setAnalysisProgress(0);
            return;
          }

          // Use residue matrix length of parsed item
          const N = matricesList[0].length;
          setRawMatrices(matricesList);
          setIMin(1);
          setIMax(N);
          setJMin(1);
          setJMax(N);
          setNumSampleFiles(matricesList.length);
          setLoadedFilesName(`${files.length} custom DCC text files loaded`);
          setStatusText(`Successfully compiled ${matricesList.length} trajectory files. Matrix resolution N = ${N}. Running computations...`);
          setAnalysisProgress(100);
          setTimeout(() => {
            setAnalysisProgress(0);
            runAnalysis({ iMin: 1, iMax: N, jMin: 1, jMax: N });
          }, 1000);
        }
      };

      reader.readAsText(file);
    }
  };

  // Run Calculations (Matching Python analysis pipeline exactly)
  const runAnalysis = (customBounds?: { iMin: number; iMax: number; jMin: number; jMax: number }) => {
    if (rawMatrices.length === 0) {
      setStatusText("Error: No trace matrices loaded.");
      return;
    }

    setIsAnalyzing(true);
    setStatusText("Analyzing trajectory covariance arrays...");
    setAnalysisProgress(20);

    setTimeout(() => {
      try {
        const numFiles = rawMatrices.length;
        const rows: HeatmapDataPoint[] = [];

        const activeIMin = customBounds ? customBounds.iMin : i_min;
        const activeIMax = customBounds ? customBounds.iMax : i_max;
        const activeJMin = customBounds ? customBounds.jMin : j_min;
        const activeJMax = customBounds ? customBounds.jMax : j_max;

        // Dual loops matching original sequence calculation:
        // for i in range(self.i_min, self.i_max + 1)
        //   for j in range(i + self.res_dist, self.j_max + 1)
        for (let i = activeIMin; i <= activeIMax; i++) {
          for (let j = i + res_dist; j <= activeJMax; j++) {
            
            // Get correlation values across files (represent frames)
            const vals: number[] = [];
            for (let f = 0; f < numFiles; f++) {
              const matrix = rawMatrices[f];
              // 1-based index in GUI map, boundary check bounds
              if (matrix && matrix[i - 1] && matrix[i - 1][j - 1] !== undefined) {
                vals.push(matrix[i - 1][j - 1]);
              }
            }

            if (vals.length === 0) continue;

            // Mean
            const sum = vals.reduce((a, b) => a + b, 0);
            const meanVal = sum / vals.length;

            // Standard deviation
            const varSum = vals.reduce((accum, val) => accum + Math.pow(val - meanVal, 2), 0);
            const stdVal = Math.sqrt(varSum / vals.length) || 0;

            // Positive probability count
            const posCount = vals.filter(v => v > 0).length;
            const posProb = posCount / vals.length;

            // Negative probability count
            const negCount = vals.filter(v => v < 0).length;
            const negProb = negCount / vals.length;

            rows.push({
              Residue_i: i,
              Residue_j: j,
              Mean_Correlation: meanVal,
              Std_Correlation: stdVal,
              Positive_Probability: posProb,
              Negative_Probability: negProb
            });
          }
        }

        // Sort: rows.sort(key=lambda x: (x[0], x[1]))
        rows.sort((a, b) => {
          if (a.Residue_i !== b.Residue_i) {
            return a.Residue_i - b.Residue_i;
          }
          return a.Residue_j - b.Residue_j;
        });

        if (rows.length === 0) {
          throw new Error("No residue selections satisfied current filters.");
        }

        setDfAll(rows);
        setStatusText(`Completed simulation computations. Generated ${rows.length} dynamic row items.`);
        setAnalysisProgress(100);
      } catch (err: any) {
        setStatusText(`Analysis failed: ${err.message || err}`);
        setAnalysisProgress(0);
      } finally {
        setIsAnalyzing(false);
        setTimeout(() => setAnalysisProgress(0), 1500);
      }
    }, 150);
  };

  // Drag and Drop multi-window spawn helpers
  const focusWindow = (id: string) => {
    setWindowFocusOrder(prev => {
      const idx = prev.indexOf(id);
      if (idx !== -1) {
        const newOrder = [...prev];
        newOrder.splice(idx, 1);
        return [...newOrder, id];
      }
      return [...prev, id];
    });
  };

  const spawnWindow = (title: string, type: FloatingWindow["type"], params?: any, w = 620, h = 480) => {
    const id = `${type}-${Date.now()}`;
    const count = openWindows.length;
    const windowOffset = 30;
    
    // Cascading starting positions
    const x = Math.min(window.innerWidth - w, 140 + count * windowOffset);
    const y = Math.min(window.innerHeight - h, 140 + count * windowOffset);

    const newWin: FloatingWindow = {
      id,
      title,
      type,
      params,
      position: { x, y },
      size: { width: w, height: h }
    };

    setOpenWindows(prev => [...prev, newWin]);
    setWindowFocusOrder(prev => [...prev, id]);
  };

  const closeWindow = (id: string) => {
    setOpenWindows(prev => prev.filter(w => w.id !== id));
    setWindowFocusOrder(prev => prev.filter(winId => winId !== id));
  };

  // CSV Exporter helpers (Matching original CSV downloads exactly)
  const handleCSVDownload = (type: "positive" | "negative" | "combined") => {
    if (df_all.length === 0) {
      setStatusText("Error: Perform correlation computations first.");
      return;
    }

    let filteredRows = [...df_all];

    if (analysis_mode === "Probability") {
      // Case 1: BASIC (Range-based filtering matching original code)
      if (type === "positive") {
        filteredRows = df_all.filter(row => 
          row.Mean_Correlation >= cmin && 
          row.Mean_Correlation <= cmax &&
          row.Positive_Probability >= pmin &&
          row.Positive_Probability <= pmax
        );
      } else if (type === "negative") {
        filteredRows = df_all.filter(row => 
          row.Mean_Correlation <= -cmin && 
          row.Mean_Correlation >= -cmax &&
          row.Negative_Probability >= pmin &&
          row.Negative_Probability <= pmax
        );
      } else {
        // Combined
        filteredRows = df_all.filter(row => 
          (row.Mean_Correlation >= cmin && row.Mean_Correlation <= cmax && row.Positive_Probability >= pmin && row.Positive_Probability <= pmax) ||
          (row.Mean_Correlation <= -cmin && row.Mean_Correlation >= -cmax && row.Negative_Probability >= pmin && row.Negative_Probability <= pmax)
        );
      }
    } else {
      // Case 2: ADVANCED (Cutoff-based filtering)
      if (type === "positive") {
        filteredRows = df_all.filter(row => 
          row.Mean_Correlation >= c_pos_cut && 
          row.Positive_Probability >= p_pos_cut
        );
      } else if (type === "negative") {
        filteredRows = df_all.filter(row => 
          row.Mean_Correlation <= c_neg_cut && 
          row.Negative_Probability >= p_neg_cut
        );
      } else {
        // Combined
        filteredRows = df_all.filter(row => 
          (row.Mean_Correlation >= c_pos_cut && row.Positive_Probability >= p_pos_cut) ||
          (row.Mean_Correlation <= c_neg_cut && row.Negative_Probability >= p_neg_cut)
        );
      }
    }

    if (filteredRows.length === 0) {
      setStatusText("Download failed: No coordinate rows satisfied active threshold parameters.");
      return;
    }

    const headers = ["Residue_i", "Residue_j", "Mean_Correlation", "Std_Correlation", "Positive_Probability", "Negative_Probability"];
    const csvContent = [
      headers.join(","),
      ...filteredRows.map(row => [
        row.Residue_i,
        row.Residue_j,
        row.Mean_Correlation.toFixed(6),
        row.Std_Correlation.toFixed(6),
        row.Positive_Probability.toFixed(6),
        row.Negative_Probability.toFixed(6)
      ].join(","))
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${type}_correlation_filtered.csv`;
    link.click();
    URL.revokeObjectURL(url);
    setStatusText(`Successfully exported ${filteredRows.length} rows to ${type}_correlation_filtered.csv.`);
  };

  // Standard and double clicks on cells spawn or update standard floating plot panel of dynamic frames with trend lines
  const handleCellClickAndTimeSeries = (i: number, j: number) => {
    // Collect historical sequence from frames
    const ts: number[] = [];
    rawMatrices.forEach(m => {
      if (m && m[i - 1] && m[i - 1][j - 1] !== undefined) {
        ts.push(m[i - 1][j - 1]);
      }
    });

    if (ts.length === 0) return;

    // Search if there is already an open time_series window to do a smooth, real-time reactive update in-place
    const existingWin = openWindows.find(w => w.type === "time_series");
    if (existingWin) {
      setOpenWindows(prev => prev.map(w => {
        if (w.id === existingWin.id) {
          return {
            ...w,
            title: `Time Series Correlation (${i}, ${j})`,
            params: { i, j, ts, mean: ts.reduce((a, b) => a + b, 0) / ts.length }
          };
        }
        return w;
      }));
      // Spark focus on the window
      focusWindow(existingWin.id);
    } else {
      spawnWindow(
        `Time Series Correlation (${i}, ${j})`,
        "time_series",
        { i, j, ts, mean: ts.reduce((a, b) => a + b, 0) / ts.length },
        550,
        350
      );
    }
  };

  return (
    <div className={`min-h-screen flex flex-col font-sans transition-all duration-300 ${
      darkMode ? "bg-slate-950 text-slate-150" : "bg-slate-50 text-slate-900"
    }`}>
      
      {/* 1. INTRO SCREEN OVERLAY */}
      {viewState === "welcome" && (
        <div className={`flex-1 flex flex-col items-center justify-center p-6 text-center min-h-screen transition-colors duration-300 ${
          darkMode ? "bg-slate-950" : "bg-slate-50"
        }`}>
          <div className={`max-w-2xl p-10 rounded-2xl border space-y-6 shadow-sm transition-all duration-300 ${
            darkMode ? "bg-slate-900/40 border-slate-800" : "bg-white border-slate-200"
          }`}>
            <div className="inline-flex p-3 rounded-xl bg-indigo-600/10 border border-indigo-500/20 text-indigo-600 mb-2">
              <Activity size={36} className="animate-pulse" />
            </div>
            <h1 className={`text-4xl sm:text-5xl font-extrabold tracking-tight ${
              darkMode ? "text-white" : "text-slate-850"
            }`}>
              Residue-X <span className="text-indigo-600">Predictor Pro</span>
            </h1>
            <p className={`text-sm sm:text-base max-w-lg mx-auto ${
              darkMode ? "text-slate-400" : "text-slate-600"
            }`}>
              Track co-varying motions and structural coupling boundaries across biomolecular MD simulation ensembles.
            </p>
            <div className="pt-4 flex flex-col items-center">
              <button
                onClick={() => setViewState("workbench")}
                className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-sm px-8 py-3.5 rounded-lg shadow-md hover:shadow-indigo-500/10 transition-all hover:scale-[1.01] active:scale-[0.99] cursor-pointer"
              >
                Launch Analysis Workstation
              </button>
              <span className={`text-[10px] font-mono mt-3 ${
                darkMode ? "text-slate-500" : "text-slate-400"
              }`}>Ready. Matplotlib responsive GUI replacement thread.</span>
            </div>
          </div>
        </div>
      )}

      {/* 2. WORKBENCH SIMULATION LAYOUT */}
      {viewState === "workbench" && (
        <div className="flex-1 flex flex-col min-h-0 min-w-0 transition-all">
          
          {/* Header toolbar stats */}
          <header className={`h-14 border-b flex flex-wrap items-center justify-between px-6 shrink-0 z-40 transition-colors duration-300 ${
            darkMode ? "bg-slate-900 border-slate-850" : "bg-white border-slate-200 shadow-sm"
          }`}>
            <div className="flex items-center gap-3.5">
              <div className="bg-indigo-600 p-1.5 rounded-lg text-white">
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.4" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h1 className={`text-base font-bold tracking-tight ${
                darkMode ? "text-white" : "text-slate-850"
              }`}>
                Residue-X <span className="text-indigo-600">Predictor Pro</span>
              </h1>
              <span className={`text-[10px] uppercase px-1.5 py-0.5 rounded font-bold tracking-widest ${
                darkMode ? "bg-slate-800 text-indigo-400" : "bg-slate-100 text-indigo-700 border border-slate-200"
              }`}>v2.1</span>
            </div>

            {/* Quick dashboard status */}
            <div className="flex items-center gap-5 text-xs">
              <div className="hidden sm:flex items-center gap-2">
                <span className={darkMode ? "text-slate-400" : "text-slate-500"}>Ensemble Traces:</span>
                <span className={`font-bold transition-all ${darkMode ? "text-indigo-400" : "text-slate-800"}`}>{rawMatrices.length} files</span>
              </div>
              <div className="hidden sm:flex items-center gap-2">
                <span className={darkMode ? "text-slate-400" : "text-slate-500"}>Computed Range:</span>
                <span className={`font-bold transition-all ${darkMode ? "text-indigo-400" : "text-slate-800"}`}>{df_all.length || "0"} pairs</span>
              </div>

              {/* Theme & Workspace resets */}
              <div className="flex items-center gap-3">
                <div className={`flex rounded-md p-0.5 border ${
                  darkMode ? "bg-slate-950 border-slate-800" : "bg-slate-100 border-slate-200"
                }`}>
                  <button
                    onClick={() => setDarkMode(false)}
                    className={`px-3 py-1 text-[10px] font-bold rounded transition-all cursor-pointer ${
                      !darkMode 
                        ? "bg-white text-slate-850 shadow-sm" 
                        : "text-slate-400 hover:text-slate-100"
                    }`}
                  >
                    Light
                  </button>
                  <button
                    onClick={() => setDarkMode(true)}
                    className={`px-3 py-1 text-[10px] font-bold rounded transition-all cursor-pointer ${
                      darkMode 
                        ? "bg-slate-800 text-white shadow-xs" 
                        : "text-slate-500 hover:text-slate-800"
                    }`}
                  >
                    Dark
                  </button>
                </div>

                <button
                  onClick={() => {
                    setViewState("welcome");
                    setOpenWindows([]);
                  }}
                  className={`px-2.5 py-1 rounded text-xs transition-all border font-medium cursor-pointer ${
                    darkMode 
                      ? "bg-slate-900 border-slate-800 hover:bg-slate-800 text-slate-400" 
                      : "bg-white border-slate-200 hover:bg-slate-100 text-slate-700 shadow-xs"
                  }`}
                >
                  Exit welcome
                </button>
              </div>
            </div>
          </header>

          <div className="flex-1 flex max-h-[calc(100vh-56px)] min-h-0 flex-col md:flex-row relative">
            
            {/* Left Control Panel Siderail */}
            <aside className="w-full md:w-[360px] flex-shrink-0 border-r p-5 overflow-y-auto flex flex-col justify-between scrollbar-thin transition-colors duration-300 bg-slate-900 border-slate-800 text-slate-300">
              <div className="space-y-4">
                
                {/* File source drag input */}
                <div className="border rounded-lg p-3.5 space-y-2 bg-slate-950/60 border-slate-800">
                  <span className="text-[10px] font-bold uppercase text-slate-400 tracking-wider">Simulation Logs</span>
                  <div className="flex items-stretch gap-2">
                    <input
                      type="text"
                      className="flex-1 text-xs bg-transparent border-none outline-none font-mono py-1 truncate text-slate-400"
                      value={loadedFilesName}
                      disabled
                    />
                    <label className="text-[11px] font-bold bg-indigo-600 hover:bg-indigo-700 text-white rounded-md px-3 py-1 flex items-center justify-center cursor-pointer transition-all duration-150 active:scale-[0.98]">
                      Upload DCC
                      <input
                        type="file"
                        multiple
                        accept=".txt"
                        className="hidden"
                        onChange={handleFileUpload}
                      />
                    </label>
                  </div>
                </div>

                {/* Residue Selection inputs matching Tkinter */}
                <div className="space-y-2">
                  <span className="text-[10px] font-bold uppercase text-indigo-400 tracking-wider block">Residue Selection bounds</span>
                  <div className="grid grid-cols-2 gap-3.5">
                    <div className="space-y-1">
                      <label className="text-[10px] text-slate-400 font-medium">Residue i range</label>
                      <div className="flex items-center gap-1.5">
                        <input
                          type="number"
                          value={i_min}
                          onChange={(e) => setIMin(Math.max(1, parseInt(e.target.value) || 1))}
                          className="w-full bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded px-2 py-1.5 text-xs text-center font-mono focus:outline-none transition-colors"
                        />
                        <span className="text-slate-500 text-xs">to</span>
                        <input
                          type="number"
                          value={i_max}
                          onChange={(e) => setIMax(Math.max(1, parseInt(e.target.value) || 1))}
                          className="w-full bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded px-2 py-1.5 text-xs text-center font-mono focus:outline-none transition-colors"
                        />
                      </div>
                    </div>
                    <div className="space-y-1">
                      <label className="text-[10px] text-slate-400 font-medium">Residue j range</label>
                      <div className="flex items-center gap-1.5">
                        <input
                          type="number"
                          value={j_min}
                          onChange={(e) => setJMin(Math.max(1, parseInt(e.target.value) || 1))}
                          className="w-full bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded px-2 py-1.5 text-xs text-center font-mono focus:outline-none transition-colors"
                        />
                        <span className="text-slate-500 text-xs">to</span>
                        <input
                          type="number"
                          value={j_max}
                          onChange={(e) => setJMax(Math.max(1, parseInt(e.target.value) || 1))}
                          className="w-full bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded px-2 py-1.5 text-xs text-center font-mono focus:outline-none transition-colors"
                        />
                      </div>
                    </div>
                  </div>
                  <div className="space-y-1">
                    <label className="text-[10px] text-slate-400 font-medium block">Sequence separation distance |j − i| ≥</label>
                    <input
                      type="number"
                      value={res_dist}
                      onChange={(e) => setResDist(Math.max(0, parseInt(e.target.value) || 0))}
                      className="w-full bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded px-2.5 py-1.5 text-xs font-mono focus:outline-none transition-colors"
                    />
                  </div>
                </div>

                {/* Analysis Radiogroup toggler */}
                <div className="space-y-2 pt-1">
                  <span className="text-[10px] font-bold uppercase text-slate-400 tracking-wider block">Analytical Workflow Mode</span>
                  <div className="grid grid-cols-1 gap-2">
                    <button
                      onClick={() => setAnalysisMode("Probability")}
                      className={`text-left p-2.5 rounded-lg border text-xs font-semibold cursor-pointer transition-all ${
                        analysis_mode === "Probability"
                          ? "bg-slate-950 border-indigo-500 text-indigo-400 shadow-sm"
                          : "bg-slate-950/30 border-slate-800 hover:border-slate-700 text-slate-400"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span>Case 1: Probability-driven</span>
                        <span className={`w-2 h-2 rounded-full ${analysis_mode === "Probability" ? "bg-indigo-500" : "bg-transparent border border-slate-700"}`} />
                      </div>
                      <p className="text-[9px] text-slate-500 mt-1 font-normal font-sans">Basic co-varying residue probability thresholds.</p>
                    </button>
                    <button
                      onClick={() => setAnalysisMode("Correlation")}
                      className={`text-left p-2.5 rounded-lg border text-xs font-semibold cursor-pointer transition-all ${
                        analysis_mode === "Correlation"
                          ? "bg-slate-950 border-indigo-500 text-indigo-400 shadow-sm"
                          : "bg-slate-950/30 border-slate-800 hover:border-slate-700 text-slate-400"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span>Case 2: Correlation-driven</span>
                        <span className={`w-2 h-2 rounded-full ${analysis_mode === "Correlation" ? "bg-indigo-500" : "bg-transparent border border-slate-700"}`} />
                      </div>
                      <p className="text-[9px] text-slate-500 mt-1 font-normal font-sans">Advanced co-efficient cutoff filtration boundaries.</p>
                    </button>
                  </div>
                </div>

                {/* Case 1: Probability parameters */}
                {analysis_mode === "Probability" && (
                  <div className="bg-slate-950/50 p-3.5 rounded-lg border border-slate-800 space-y-2.5">
                    <span className="text-[10px] font-bold uppercase text-indigo-400 tracking-wider block">Probability Range (Pmin - Pmax)</span>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="space-y-0.5">
                        <label className="text-[9px] text-slate-400">Min bound</label>
                        <input
                          type="number"
                          step="0.05"
                          value={pmin}
                          onChange={(e) => setPMin(parseFloat(e.target.value) || 0)}
                          className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs font-mono focus:outline-none focus:border-indigo-500 transition-colors"
                        />
                      </div>
                      <div className="space-y-0.5">
                        <label className="text-[9px] text-slate-400">Max bound</label>
                        <input
                          type="number"
                          step="0.05"
                          value={pmax}
                          onChange={(e) => setPMax(parseFloat(e.target.value) || 0)}
                          className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs font-mono focus:outline-none focus:border-indigo-500 transition-colors"
                        />
                      </div>
                    </div>
                    <span className="text-[10px] font-bold uppercase text-indigo-400 tracking-wider block pt-1.5">Correlation bounds |C| (Cmin - Cmax)</span>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="space-y-0.5">
                        <label className="text-[9px] text-slate-400">Min Correlation</label>
                        <input
                          type="number"
                          step="0.05"
                          value={cmin}
                          onChange={(e) => setCMin(parseFloat(e.target.value) || 0)}
                          className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs font-mono focus:outline-none focus:border-indigo-500 transition-colors"
                        />
                      </div>
                      <div className="space-y-0.5">
                        <label className="text-[9px] text-slate-400">Max Correlation</label>
                        <input
                          type="number"
                          step="0.05"
                          value={cmax}
                          onChange={(e) => setCMax(parseFloat(e.target.value) || 0)}
                          className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs font-mono focus:outline-none focus:border-indigo-500 transition-colors"
                        />
                      </div>
                    </div>
                  </div>
                )}

                {/* Case 2: Advanced cutoff inputs */}
                {analysis_mode === "Correlation" && (
                  <div className="bg-slate-950/50 p-3.5 rounded-lg border border-slate-800 space-y-2.5">
                    <span className="text-[10px] font-bold uppercase text-indigo-400 tracking-wider block mb-1">Advanced Co-efficient Cutoffs</span>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="space-y-0.5">
                        <label className="text-[9px] text-slate-500 font-bold uppercase">Pos Prob &ge;</label>
                        <input
                          type="number"
                          step="0.05"
                          value={p_pos_cut}
                          onChange={(e) => setPPosCut(parseFloat(e.target.value) || 0)}
                          className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs font-mono focus:outline-none focus:border-indigo-500 transition-colors"
                        />
                      </div>
                      <div className="space-y-0.5">
                        <label className="text-[9px] text-slate-500 font-bold uppercase">Neg Prob &ge;</label>
                        <input
                          type="number"
                          step="0.05"
                          value={p_neg_cut}
                          onChange={(e) => setPNegCut(parseFloat(e.target.value) || 0)}
                          className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs font-mono focus:outline-none focus:border-indigo-500 transition-colors"
                        />
                      </div>
                      <div className="space-y-0.5">
                        <label className="text-[9px] text-slate-500 font-bold uppercase">Pos Corr &ge;</label>
                        <input
                          type="number"
                          step="0.05"
                          value={c_pos_cut}
                          onChange={(e) => setCPosCut(parseFloat(e.target.value) || 0)}
                          className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs font-mono focus:outline-none focus:border-indigo-500 transition-colors"
                        />
                      </div>
                      <div className="space-y-0.5">
                        <label className="text-[9px] text-slate-500 font-bold uppercase">Neg Corr &le;</label>
                        <input
                          type="number"
                          step="0.05"
                          value={c_neg_cut}
                          onChange={(e) => setCNegCut(parseFloat(e.target.value) || 0)}
                          className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs font-mono focus:outline-none focus:border-indigo-500 transition-colors"
                        />
                      </div>
                    </div>
                  </div>
                )}

                {/* Primary execution action */}
                <button
                  onClick={runAnalysis}
                  disabled={isAnalyzing}
                  className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs py-3 rounded-lg shadow-md hover:shadow-indigo-500/10 transition-all flex items-center justify-center gap-2 cursor-pointer duration-150 active:scale-[0.98]"
                >
                  <Sliders size={13} />
                  {isAnalyzing ? "ANALYZING TRAJECTORY VECTORS..." : "RUN COVARIANCE COMPUTATION"}
                </button>

                {/* Python original standalone Toplevel plot window popouts */}
                <div className="space-y-2 pt-2">
                  <span className="text-[10px] font-bold uppercase text-slate-400 tracking-wider block">Spawn GUI Desktop Plots</span>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    <button
                      onClick={() => spawnWindow("Correlation Dashboard View", "dashboard_correlation", null, 660, 520)}
                      disabled={df_all.length === 0}
                      className="text-[10px] font-semibold py-2 px-3 border rounded-lg bg-slate-950/40 border-slate-800 hover:border-indigo-500/50 hover:bg-slate-950 text-slate-350 hover:text-indigo-400 transition-all cursor-pointer disabled:opacity-20 duration-150"
                    >
                      Correlation Dashboard
                    </button>
                    <button
                      onClick={() => spawnWindow("Probability Dashboard View", "dashboard_probability", null, 660, 520)}
                      disabled={df_all.length === 0}
                      className="text-[10px] font-semibold py-2 px-3 border rounded-lg bg-slate-950/40 border-slate-800 hover:border-indigo-500/50 hover:bg-slate-950 text-slate-350 hover:text-indigo-400 transition-all cursor-pointer disabled:opacity-20 duration-150"
                    >
                      Probability Dashboard
                    </button>
                    <button
                      onClick={() => spawnWindow("Interactive Correlation Studio", "interactive_correlation", null, 680, 560)}
                      disabled={df_all.length === 0}
                      className="text-[10px] font-semibold py-2 px-3 border rounded-lg bg-slate-950/40 border-slate-800 hover:border-indigo-500/50 hover:bg-slate-950 text-slate-350 hover:text-indigo-400 transition-all cursor-pointer disabled:opacity-20 duration-150"
                    >
                      Interactive Correlation
                    </button>
                    <button
                      onClick={() => spawnWindow("Interactive Probability Studio", "interactive_probability", null, 680, 560)}
                      disabled={df_all.length === 0}
                      className="text-[10px] font-semibold py-2 px-3 border rounded-lg bg-slate-950/40 border-slate-800 hover:border-indigo-500/50 hover:bg-slate-950 text-slate-350 hover:text-indigo-400 transition-all cursor-pointer disabled:opacity-20 duration-150"
                    >
                      Interactive Probability
                    </button>
                  </div>
                </div>

              </div>

              {/* Data downloads module frame matching Python */}
              <div className="space-y-2.5 pt-4 border-t border-slate-800/60">
                <span className="text-[10px] font-bold uppercase text-slate-400 tracking-wider block">Download Matrices</span>
                <div className="flex flex-col gap-2">
                  <button
                    onClick={() => handleCSVDownload("combined")}
                    disabled={df_all.length === 0}
                    className="w-full text-left bg-emerald-950/20 hover:bg-emerald-950/40 text-emerald-400 border border-emerald-900/40 font-semibold text-[11px] py-2 px-3.5 rounded-lg flex items-center justify-between transition-colors cursor-pointer duration-150 disabled:opacity-20"
                  >
                    <span>Download All correlated (Filtered)</span>
                    <Download size={13} />
                  </button>
                  <button
                    onClick={() => handleCSVDownload("positive")}
                    disabled={df_all.length === 0}
                    className="w-full text-left bg-emerald-950/20 hover:bg-emerald-950/40 text-emerald-400 border border-emerald-900/40 font-semibold text-[11px] py-2 px-3.5 rounded-lg flex items-center justify-between transition-colors cursor-pointer duration-150 disabled:opacity-20"
                  >
                    <span>Download Positive correlated</span>
                    <Download size={13} />
                  </button>
                  <button
                    onClick={() => handleCSVDownload("negative")}
                    disabled={df_all.length === 0}
                    className="w-full text-left bg-indigo-950/20 hover:bg-indigo-950/40 text-indigo-450 border border-indigo-900/40 font-semibold text-[11px] py-2 px-3.5 rounded-lg flex items-center justify-between transition-colors cursor-pointer duration-150 disabled:opacity-20"
                  >
                    <span>Download Negative correlated</span>
                    <Download size={13} />
                  </button>
                </div>
              </div>

            </aside>

            {/* Right Pane: Visualizers and models tabs workspaces */}
            <main className={`flex-1 flex flex-col min-w-0 transition-colors duration-300 ${
              darkMode ? "bg-slate-950" : "bg-slate-100"
            }`}>
              
              {/* Tab Selector */}
              <div className={`flex items-center px-6 border-b gap-1 select-none z-30 transition-colors duration-300 ${
                darkMode ? "bg-slate-900 border-slate-850" : "bg-white border-slate-200 shadow-xs"
              }`}>
                {[
                  { id: "dashboard", label: "Dynamic Dashboard", icon: LayoutDashboard },
                  { id: "interactive_corr", label: "Interactive Correlation", icon: Binary },
                  { id: "interactive_prob", label: "Interactive Probability", icon: Layers },
                  { id: "prediction", label: "AI Prediction", icon: Cpu },
                  { id: "domains", label: "Dynamic Domains & Hubs", icon: TableProperties }
                ].map(tab => {
                  const Icon = tab.icon;
                  const isActive = activeTab === tab.id;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id as any)}
                      className={`flex items-center gap-2 py-3.5 px-4 text-xs font-semibold border-b-2 tracking-wide transition-all cursor-pointer ${
                        isActive
                          ? (darkMode ? "border-indigo-500 text-indigo-400" : "border-indigo-600 text-indigo-600")
                          : (darkMode ? "border-transparent text-slate-450 hover:text-slate-200" : "border-transparent text-slate-500 hover:text-slate-800")
                      }`}
                    >
                      <Icon size={14} className={isActive ? (darkMode ? "text-indigo-400" : "text-indigo-600") : "text-slate-400"} />
                      <span>{tab.label}</span>
                    </button>
                  );
                })}
              </div>

              {/* Workstation viewspaces */}
              <div className="flex-1 min-h-0 min-w-0 flex flex-col overflow-auto font-sans relative">
                
                {/* A. Dynamic Dashboard layout */}
                {activeTab === "dashboard" && (
                  <div className="p-4 space-y-4 h-full overflow-auto">
                    {df_all.length > 0 ? (
                      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                        
                        {/* Positive Correlation Chart */}
                        <div className={`border rounded-xl p-3 flex flex-col h-[340px] transition-all duration-200 ${
                          darkMode ? "bg-slate-900 border-slate-800/80 text-slate-100" : "bg-white border-slate-200 shadow-sm text-slate-800"
                        }`}>
                          <span className={`text-[10px] uppercase font-extrabold tracking-wider mb-2 ${darkMode ? "text-slate-400" : "text-slate-500"}`}>Positive Correlation Plots</span>
                          <div className="flex-1 min-h-0">
                            <InteractiveHeatmap
                              data={df_all}
                              iMin={i_min}
                              iMax={i_max}
                              jMin={j_min}
                              jMax={j_max}
                              mode="correlation"
                              subType="positive"
                              pMin={pmin}
                              pMax={pmax}
                              cMin={cmin}
                              cMax={cmax}
                              pPosCut={p_pos_cut}
                              pNegCut={p_neg_cut}
                              cPosCut={c_pos_cut}
                              cNegCut={c_neg_cut}
                              analysisMode={analysis_mode}
                              onCellDoubleClick={handleCellClickAndTimeSeries}
                              darkMode={darkMode}
                            />
                          </div>
                        </div>

                        {/* Negative Correlation Chart */}
                        <div className={`border rounded-xl p-3 flex flex-col h-[340px] transition-all duration-200 ${
                          darkMode ? "bg-slate-900 border-slate-800/80 text-slate-100" : "bg-white border-slate-200 shadow-sm text-slate-800"
                        }`}>
                          <span className={`text-[10px] uppercase font-extrabold tracking-wider mb-2 ${darkMode ? "text-slate-400" : "text-slate-500"}`}>Negative Correlation Plots</span>
                          <div className="flex-1 min-h-0">
                            <InteractiveHeatmap
                              data={df_all}
                              iMin={i_min}
                              iMax={i_max}
                              jMin={j_min}
                              jMax={j_max}
                              mode="correlation"
                              subType="negative"
                              pMin={pmin}
                              pMax={pmax}
                              cMin={cmin}
                              cMax={cmax}
                              pPosCut={p_pos_cut}
                              pNegCut={p_neg_cut}
                              cPosCut={c_pos_cut}
                              cNegCut={c_neg_cut}
                              analysisMode={analysis_mode}
                              onCellDoubleClick={handleCellClickAndTimeSeries}
                              darkMode={darkMode}
                            />
                          </div>
                        </div>

                        {/* Combined Correlation Chart */}
                        <div className={`border rounded-xl p-3 flex flex-col h-[340px] transition-all duration-200 ${
                          darkMode ? "bg-slate-900 border-slate-800/80 text-slate-100" : "bg-white border-slate-200 shadow-sm text-slate-800"
                        }`}>
                          <span className={`text-[10px] uppercase font-extrabold tracking-wider mb-2 ${darkMode ? "text-slate-400" : "text-slate-500"}`}>Combined Correlation Plot (+Red, −Blue)</span>
                          <div className="flex-1 min-h-0">
                            <InteractiveHeatmap
                              data={df_all}
                              iMin={i_min}
                              iMax={i_max}
                              jMin={j_min}
                              jMax={j_max}
                              mode="correlation"
                              subType="combined"
                              pMin={pmin}
                              pMax={pmax}
                              cMin={cmin}
                              cMax={cmax}
                              pPosCut={p_pos_cut}
                              pNegCut={p_neg_cut}
                              cPosCut={c_pos_cut}
                              cNegCut={c_neg_cut}
                              analysisMode={analysis_mode}
                              onCellDoubleClick={handleCellClickAndTimeSeries}
                              darkMode={darkMode}
                            />
                          </div>
                        </div>

                      </div>
                    ) : (
                      <div className="h-full flex flex-col items-center justify-center p-12 text-center text-slate-500 border-2 border-dashed border-slate-900 m-4 rounded-xl">
                        <Sliders size={32} className="text-slate-800 mb-3 animate-pulse" />
                        <h4 className="text-xs font-semibold mb-1 text-slate-400">Calculations Staged</h4>
                        <p className="text-[11px] text-slate-500 max-w-sm">
                          Configure numerical bounds on the left control dock and click "RUN COVARIANCE COMPUTATION" to generate interactive dynamics plots.
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {/* B. Interactive Correlation and probability tabs */}
                {activeTab === "interactive_corr" && (
                  <div className="p-4 space-y-4 h-full flex flex-col">
                    {df_all.length > 0 ? (
                      <div className={`flex-1 min-h-0 border rounded-xl p-4 flex flex-col transition-all duration-200 ${
                        darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200 shadow-sm"
                      }`}>
                        <div className="flex justify-between items-center mb-3">
                          <h3 className={`text-xs font-extrabold uppercase tracking-wide ${darkMode ? "text-slate-300" : "text-slate-800"}`}>Composite Correlation heatmaps</h3>
                          <span className={`text-[10px] font-mono ${darkMode ? "text-slate-500" : "text-slate-405"}`}>Dbl-click cell coordinates to open frames time-series trace</span>
                        </div>
                        <div className="flex-1 min-h-0">
                          <InteractiveHeatmap
                            data={df_all}
                            iMin={i_min}
                            iMax={i_max}
                            jMin={j_min}
                            jMax={j_max}
                            mode="correlation"
                            subType="combined"
                            pMin={pmin}
                            pMax={pmax}
                            cMin={cmin}
                            cMax={cmax}
                            pPosCut={p_pos_cut}
                            pNegCut={p_neg_cut}
                            cPosCut={c_pos_cut}
                            cNegCut={c_neg_cut}
                            analysisMode={analysis_mode}
                            onCellDoubleClick={handleCellClickAndTimeSeries}
                            darkMode={darkMode}
                          />
                        </div>
                      </div>
                    ) : (
                      <div className="flex-1 flex flex-col items-center justify-center text-slate-500">
                        Run primary calculations to load interactive correlation canvas.
                      </div>
                    )}
                  </div>
                )}

                {activeTab === "interactive_prob" && (
                  <div className="p-4 space-y-4 h-full flex flex-col">
                    {df_all.length > 0 ? (
                      <div className={`flex-1 min-h-0 border rounded-xl p-4 flex flex-col transition-all duration-200 ${
                        darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200 shadow-sm"
                      }`}>
                        <div className="flex justify-between items-center mb-3">
                          <h3 className={`text-xs font-extrabold uppercase tracking-wide ${darkMode ? "text-slate-300" : "text-slate-800"}`}>Composite Probability heatmaps</h3>
                          <span className={`text-[10px] font-mono ${darkMode ? "text-slate-500" : "text-slate-450"}`}>Dbl-click coordinates to open time-series</span>
                        </div>
                        <div className="flex-1 min-h-0">
                          <InteractiveHeatmap
                            data={df_all}
                            iMin={i_min}
                            iMax={i_max}
                            jMin={j_min}
                            jMax={j_max}
                            mode="probability"
                            subType="combined"
                            pMin={pmin}
                            pMax={pmax}
                            cMin={cmin}
                            cMax={cmax}
                            pPosCut={p_pos_cut}
                            pNegCut={p_neg_cut}
                            cPosCut={c_pos_cut}
                            cNegCut={c_neg_cut}
                            analysisMode={analysis_mode}
                            onCellDoubleClick={handleCellClickAndTimeSeries}
                            darkMode={darkMode}
                          />
                        </div>
                      </div>
                    ) : (
                      <div className="flex-1 flex flex-col items-center justify-center text-slate-500">
                        Run primary calculations to load interactive probability layout.
                      </div>
                    )}
                  </div>
                )}

                {/* C. ML prediction panel */}
                {activeTab === "prediction" && (
                  <AIPredictionPanel darkMode={darkMode} dccData={df_all} />
                )}

                {/* E. Dynamic Domains partition panel (outstanding functionality without detailed outside data) */}
                {activeTab === "domains" && (
                  <DynamicDomainsPanel dccData={df_all} darkMode={darkMode} />
                )}

              </div>

              {/* Status bar panel matching Tkinter requirements */}
              <footer className={`px-4 py-1.5 border-t text-[10px] font-mono flex items-center justify-between z-30 ${
                darkMode ? "bg-slate-950 border-slate-850 text-slate-500" : "bg-slate-100 border-slate-250 text-slate-600"
              }`}>
                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                  <span>Workbench State: <strong>{statusText}</strong></span>
                </div>
                <div>
                  <span>Threads: Node 22 / React 19</span>
                </div>
              </footer>

            </main>

          </div>

          {/* 3. FLATING COMPILER DESKTOP OVERLAY WINDOWS */}
          {openWindows.map(win => {
            const zIndexIdx = windowFocusOrder.indexOf(win.id);
            const zIndexVal = zIndexIdx !== -1 ? 50 + zIndexIdx : 50;

            return (
              <WindowFrame
                key={win.id}
                id={win.id}
                title={win.title}
                defaultSize={win.size}
                onClose={() => closeWindow(win.id)}
                onFocus={() => focusWindow(win.id)}
                activeZIndex={zIndexVal}
              >
                {/* Dynamic floating layout compilers */}
                
                {/* 3.1 Time series coordinate frames plot */}
                {win.type === "time_series" && (
                  <TimeSeriesPlot
                    win={win}
                    rawMatrices={rawMatrices}
                    chartDisplayTheme={chartDisplayTheme}
                    setChartDisplayTheme={setChartDisplayTheme}
                    darkMode={darkMode}
                    data={df_all}
                  />
                )}

                {/* 3.2 Dynamic heatmaps in multi window views */}
                {win.type === "dashboard_correlation" && (
                  <div className="flex flex-col h-full bg-slate-950 p-3 overflow-y-auto space-y-3.5 scrollbar-thin">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="bg-slate-900 border border-slate-800 p-2 h-[220px] rounded flex flex-col">
                        <span className="text-[9px] uppercase tracking-wider font-bold text-slate-400 mb-1">Row Positive Correlation</span>
                        <div className="flex-1 min-h-0">
                          <InteractiveHeatmap
                            data={df_all}
                            iMin={i_min}
                            iMax={i_max}
                            jMin={j_min}
                            jMax={j_max}
                            mode="correlation"
                            subType="positive"
                            pMin={pmin}
                            pMax={pmax}
                            cMin={cmin}
                            cMax={cmax}
                            pPosCut={p_pos_cut}
                            pNegCut={p_neg_cut}
                            cPosCut={c_pos_cut}
                            cNegCut={c_neg_cut}
                            analysisMode={analysis_mode}
                            onCellDoubleClick={handleCellClickAndTimeSeries}
                          />
                        </div>
                      </div>
                      <div className="bg-slate-900 border border-slate-800 p-2 h-[220px] rounded flex flex-col">
                        <span className="text-[9px] uppercase tracking-wider font-bold text-slate-400 mb-1">Row Negative Correlation</span>
                        <div className="flex-1 min-h-0">
                          <InteractiveHeatmap
                            data={df_all}
                            iMin={i_min}
                            iMax={i_max}
                            jMin={j_min}
                            jMax={j_max}
                            mode="correlation"
                            subType="negative"
                            pMin={pmin}
                            pMax={pmax}
                            cMin={cmin}
                            cMax={cmax}
                            pPosCut={p_pos_cut}
                            pNegCut={p_neg_cut}
                            cPosCut={c_pos_cut}
                            cNegCut={c_neg_cut}
                            analysisMode={analysis_mode}
                            onCellDoubleClick={handleCellClickAndTimeSeries}
                          />
                        </div>
                      </div>
                    </div>
                    <div className="bg-slate-900 border border-slate-800 p-2 h-[220px] rounded flex flex-col">
                      <span className="text-[9px] uppercase tracking-wider font-bold text-slate-400 mb-1">Composite Combined Matrix</span>
                      <div className="flex-1 min-h-0">
                        <InteractiveHeatmap
                          data={df_all}
                          iMin={i_min}
                          iMax={i_max}
                          jMin={j_min}
                          jMax={j_max}
                          mode="correlation"
                          subType="combined"
                          pMin={pmin}
                          pMax={pmax}
                          cMin={cmin}
                          cMax={cmax}
                          pPosCut={p_pos_cut}
                          pNegCut={p_neg_cut}
                          cPosCut={c_pos_cut}
                          cNegCut={c_neg_cut}
                          analysisMode={analysis_mode}
                          onCellDoubleClick={handleCellClickAndTimeSeries}
                        />
                      </div>
                    </div>
                  </div>
                )}

                {win.type === "dashboard_probability" && (
                  <div className="flex flex-col h-full bg-slate-950 p-3 overflow-y-auto space-y-3.5 scrollbar-thin">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div className="bg-slate-900 border border-slate-800 p-2 h-[220px] rounded flex flex-col">
                        <span className="text-[9px] uppercase tracking-wider font-bold text-slate-400 mb-1">Row Positive Probability</span>
                        <div className="flex-1 min-h-0">
                          <InteractiveHeatmap
                            data={df_all}
                            iMin={i_min}
                            iMax={i_max}
                            jMin={j_min}
                            jMax={j_max}
                            mode="probability"
                            subType="positive"
                            pMin={pmin}
                            pMax={pmax}
                            cMin={cmin}
                            cMax={cmax}
                            pPosCut={p_pos_cut}
                            pNegCut={p_neg_cut}
                            cPosCut={c_pos_cut}
                            cNegCut={c_neg_cut}
                            analysisMode={analysis_mode}
                            onCellDoubleClick={handleCellClickAndTimeSeries}
                          />
                        </div>
                      </div>
                      <div className="bg-slate-900 border border-slate-800 p-2 h-[220px] rounded flex flex-col">
                        <span className="text-[9px] uppercase tracking-wider font-bold text-slate-400 mb-1">Row Negative Probability</span>
                        <div className="flex-1 min-h-0">
                          <InteractiveHeatmap
                            data={df_all}
                            iMin={i_min}
                            iMax={i_max}
                            jMin={j_min}
                            jMax={j_max}
                            mode="probability"
                            subType="negative"
                            pMin={pmin}
                            pMax={pmax}
                            cMin={cmin}
                            cMax={cmax}
                            pPosCut={p_pos_cut}
                            pNegCut={p_neg_cut}
                            cPosCut={c_pos_cut}
                            cNegCut={c_neg_cut}
                            analysisMode={analysis_mode}
                            onCellDoubleClick={handleCellClickAndTimeSeries}
                          />
                        </div>
                      </div>
                    </div>
                    <div className="bg-slate-900 border border-slate-800 p-2 h-[220px] rounded flex flex-col">
                      <span className="text-[9px] uppercase tracking-wider font-bold text-slate-400 mb-1">Composite Combined Probability</span>
                      <div className="flex-1 min-h-0">
                        <InteractiveHeatmap
                          data={df_all}
                          iMin={i_min}
                          iMax={i_max}
                          jMin={j_min}
                          jMax={j_max}
                          mode="probability"
                          subType="combined"
                          pMin={pmin}
                          pMax={pmax}
                          cMin={cmin}
                          cMax={cmax}
                          pPosCut={p_pos_cut}
                          pNegCut={p_neg_cut}
                          cPosCut={c_pos_cut}
                          cNegCut={c_neg_cut}
                          analysisMode={analysis_mode}
                          onCellDoubleClick={handleCellClickAndTimeSeries}
                        />
                      </div>
                    </div>
                  </div>
                )}

                {win.type === "interactive_correlation" && (
                  <div className="w-full h-full p-4 flex flex-col">
                    <div className="flex-1 min-h-0">
                      <InteractiveHeatmap
                        data={df_all}
                        iMin={i_min}
                        iMax={i_max}
                        jMin={j_min}
                        jMax={j_max}
                        mode="correlation"
                        subType="combined"
                        pMin={pmin}
                        pMax={pmax}
                        cMin={cmin}
                        cMax={cmax}
                        pPosCut={p_pos_cut}
                        pNegCut={p_neg_cut}
                        cPosCut={c_pos_cut}
                        cNegCut={c_neg_cut}
                        analysisMode={analysis_mode}
                        onCellDoubleClick={handleCellClickAndTimeSeries}
                      />
                    </div>
                  </div>
                )}

                {win.type === "interactive_probability" && (
                  <div className="w-full h-full p-4 flex flex-col">
                    <div className="flex-1 min-h-0">
                      <InteractiveHeatmap
                        data={df_all}
                        iMin={i_min}
                        iMax={i_max}
                        jMin={j_min}
                        jMax={j_max}
                        mode="probability"
                        subType="combined"
                        pMin={pmin}
                        pMax={pmax}
                        cMin={cmin}
                        cMax={cmax}
                        pPosCut={p_pos_cut}
                        pNegCut={p_neg_cut}
                        cPosCut={c_pos_cut}
                        cNegCut={c_neg_cut}
                        analysisMode={analysis_mode}
                        onCellDoubleClick={handleCellClickAndTimeSeries}
                      />
                    </div>
                  </div>
                )}

              </WindowFrame>
            );
          })}

          {/* 4. RUNNING PROGRESS OVERLAY PANEL (Matches Progressbar request) */}
          {isAnalyzing && (
            <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm flex flex-col items-center justify-center p-6 z-[9999]">
              <div className="bg-slate-900 border border-slate-800 p-6 rounded-lg shadow-2xl max-w-sm w-full space-y-4 text-center">
                <div className="relative w-10 h-10 mx-auto">
                  <div className="absolute inset-0 rounded-full border-2 border-indigo-500/20" />
                  <div className="absolute inset-0 rounded-full border-2 border-indigo-500 border-t-transparent animate-spin" />
                </div>
                <div className="space-y-1">
                  <h4 className="text-sm font-bold text-slate-100">Calculating Correlation Covariance</h4>
                  <p className="text-[11px] text-slate-400">Processing residue trajectory arrays in memory thread...</p>
                </div>
                {/* Visual Progress Bar indicator */}
                <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
                  <div 
                    className="bg-indigo-500 h-full rounded transition-all duration-300"
                    style={{ width: `${analysisProgress}%` }}
                  />
                </div>
                <span className="text-[10px] text-indigo-400 font-mono font-bold">{analysisProgress}%</span>
              </div>
            </div>
          )}

        </div>
      )}

    </div>
  );
}
