import React, { useState, useMemo, useEffect } from "react";
import { Upload, Cpu, Play, BarChart2, Activity, ListCollapse, Layers } from "lucide-react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell
} from "recharts";
import {
  LinearRegression,
  RandomForestRegressor,
  GradientBoostingRegressor,
  calculateMetrics,
  MLMetrics,
  PredictionResult,
  FeatureImportance
} from "../utils/ml";

// Generate extremely realistic sample biomolecular dynamics dataset for instant walkthrough
const generateSampleCSV = (): { headers: string[]; data: Record<string, number>[] } => {
  const headers = [
    "Residue_Distance",
    "RMSF_i",
    "RMSF_j",
    "Solvent_Accessibility",
    "Hydrogen_Bonds",
    "Electrostatic_Energy",
    "Mean_Correlation"
  ];

  const data: Record<string, number>[] = [];
  for (let k = 0; k < 150; k++) {
    const dist = Math.floor(Math.random() * 80) + 2;
    const rmsf_i = Math.random() * 3.5 + 0.5;
    const rmsf_j = Math.random() * 3.5 + 0.5;
    const solvent = Math.random() * 80 + 5;
    const hbonds = dist < 6 ? Math.floor(Math.random() * 4) + 1 : (Math.random() > 0.8 ? 1 : 0);
    const electro = -(Math.random() * 15) - (dist < 5 ? 20 : 0);
    
    // Core physical dynamic formula
    const baseCorr = 1.0 / (dist * 0.15 + 1.0) * (1.0 + hbonds * 0.25) + (Math.random() - 0.5) * 0.2;
    const meanCorrelation = Math.max(-1, Math.min(1, baseCorr));

    data.push({
      Residue_Distance: dist,
      RMSF_i: rmsf_i,
      RMSF_j: rmsf_j,
      Solvent_Accessibility: solvent,
      Hydrogen_Bonds: hbonds,
      Electrostatic_Energy: electro,
      Mean_Correlation: meanCorrelation
    });
  }

  return { headers, data };
};

// Compile-on-the-fly DCC mathematical profile features without needing hydrogen bonds, solvent accessibility, etc.
const generateDCCFeatures = (data: any[]): { headers: string[]; data: Record<string, number>[] } => {
  if (!data || data.length === 0) return { headers: [], data: [] };

  const resSet = new Set<number>();
  data.forEach(item => {
    resSet.add(item.Residue_i);
    resSet.add(item.Residue_j);
  });
  const residues = Array.from(resSet).sort((a, b) => a - b);
  const n = residues.length;

  const meanCorrMap: Record<number, number[]> = {};
  const stdCorrMap: Record<number, number[]> = {};

  residues.forEach(r => {
    meanCorrMap[r] = [];
    stdCorrMap[r] = [];
  });

  data.forEach(item => {
    const r_i = item.Residue_i;
    const r_j = item.Residue_j;
    meanCorrMap[r_i].push(item.Mean_Correlation);
    meanCorrMap[r_j].push(item.Mean_Correlation);
    stdCorrMap[r_i].push(item.Std_Correlation);
    stdCorrMap[r_j].push(item.Std_Correlation);
  });

  const hubness: Record<number, number> = {};
  const flexibility: Record<number, number> = {};

  residues.forEach(r => {
    const listMean = meanCorrMap[r] || [];
    const listStd = stdCorrMap[r] || [];
    
    const absSum = listMean.reduce((sum, val) => sum + Math.abs(val), 0);
    hubness[r] = listMean.length > 0 ? (absSum / listMean.length) : 0;

    const stdSum = listStd.reduce((sum, val) => sum + val, 0);
    flexibility[r] = listStd.length > 0 ? (stdSum / listStd.length) : 0;
  });

  const featuresList: Record<string, number>[] = data.map(item => {
    const i = item.Residue_i;
    const j = item.Residue_j;
    const dist = Math.abs(i - j);
    const profileSim = 1.0 - Math.abs(hubness[i] - hubness[j]);

    return {
      Sequence_Separation: dist,
      Residue_i_Hubness: parseFloat(hubness[i].toFixed(4)),
      Residue_j_Hubness: parseFloat(hubness[j].toFixed(4)),
      Residue_i_Flexibility: parseFloat(flexibility[i].toFixed(4)),
      Residue_j_Flexibility: parseFloat(flexibility[j].toFixed(4)),
      Profile_Similarity: parseFloat(profileSim.toFixed(4)),
      Mean_Correlation: parseFloat(item.Mean_Correlation.toFixed(4))
    };
  });

  const headers = [
    "Sequence_Separation",
    "Residue_i_Hubness",
    "Residue_j_Hubness",
    "Residue_i_Flexibility",
    "Residue_j_Flexibility",
    "Profile_Similarity",
    "Mean_Correlation"
  ];

  return { headers, data: featuresList };
};

interface AIPredictionPanelProps {
  darkMode?: boolean;
  dccData?: any[];
}

export default function AIPredictionPanel({ darkMode = true, dccData = [] }: AIPredictionPanelProps) {
  const sample = useMemo(() => generateSampleCSV(), []);
  
  const [predictionMode, setPredictionMode] = useState<"dcc_on_the_fly" | "csv_upload">(
    dccData && dccData.length > 0 ? "dcc_on_the_fly" : "csv_upload"
  );

  const [csvHeaders, setCsvHeaders] = useState<string[]>(sample.headers);
  const [csvData, setCsvData] = useState<Record<string, number>[]>(sample.data);
  const [csvFileName, setCsvFileName] = useState<string>("protein_dynamics_sample.csv (Loaded)");
  const [statusText, setStatusText] = useState<string>("Ready to train ML prediction models.");

  // ML configuration states
  const [targetColumn, setTargetColumn] = useState<string>("Mean_Correlation");
  const [inputFeatures, setInputFeatures] = useState<string[]>([
    "Residue_Distance",
    "RMSF_i",
    "RMSF_j",
    "Solvent_Accessibility",
    "Hydrogen_Bonds",
    "Electrostatic_Energy"
  ]);

  // Synchronize predictions data mode
  useEffect(() => {
    if (predictionMode === "dcc_on_the_fly" && dccData && dccData.length > 0) {
      const generated = generateDCCFeatures(dccData);
      setCsvHeaders(generated.headers);
      setCsvData(generated.data);
      setTargetColumn("Mean_Correlation");
      setInputFeatures(generated.headers.filter(h => h !== "Mean_Correlation"));
      setCsvFileName(`dynamic_dcc_features (${dccData.length} pairs)`);
      setStatusText("Synthesized math topology features from active DCC matrices.");
      
      setMetrics(null);
      setPredictionsList([]);
      setFeatureImportances([]);
      setTablePage(0);
    } else if (predictionMode === "csv_upload") {
      setCsvHeaders(sample.headers);
      setCsvData(sample.data);
      setTargetColumn("Mean_Correlation");
      setInputFeatures([
        "Residue_Distance",
        "RMSF_i",
        "RMSF_j",
        "Solvent_Accessibility",
        "Hydrogen_Bonds",
        "Electrostatic_Energy"
      ]);
      setCsvFileName("protein_dynamics_sample.csv (Restored)");
      setStatusText("Loaded standard biophysics feature set from template CSV.");
      
      setMetrics(null);
      setPredictionsList([]);
      setFeatureImportances([]);
      setTablePage(0);
    }
  }, [predictionMode, dccData]);

  const [trainRatio, setTrainRatio] = useState<number>(0.8);
  const [modelType, setModelType] = useState<"linear" | "rf" | "gb">("rf");

  // Model parameters (Decision Tree, RF forest size...)
  const [numEstimators, setNumEstimators] = useState<number>(20);
  const [maxDepth, setMaxDepth] = useState<number>(5);

  // Training outputs
  const [metrics, setMetrics] = useState<MLMetrics | null>(null);
  const [predictionsList, setPredictionsList] = useState<PredictionResult[]>([]);
  const [featureImportances, setFeatureImportances] = useState<FeatureImportance[]>([]);
  const [isTraining, setIsTraining] = useState<boolean>(false);

  // Pagination for table
  const [tablePage, setTablePage] = useState<number>(0);
  const pageSize = 10;

  // Custom CSV parser handling commas & basic spaces
  const handleCSVUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setCsvFileName(file.name);
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      const lines = text.trim().split(/\r?\n/);
      if (lines.length < 2) {
        setStatusText("Error: Empty or invalid CSV file.");
        return;
      }

      const headers = lines[0].split(",").map(h => h.trim().replace(/^["']|["']$/g, ""));
      const parsedData: Record<string, number>[] = [];

      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(",").map(v => v.trim());
        if (values.length !== headers.length) continue;

        const row: Record<string, number> = {};
        let valid = true;
        for (let j = 0; j < headers.length; j++) {
          const num = parseFloat(values[j]);
          if (isNaN(num)) {
            valid = false;
            break;
          }
          row[headers[j]] = num;
        }
        if (valid) {
          parsedData.push(row);
        }
      }

      if (parsedData.length === 0) {
        setStatusText("Error: Could not parse any valid numerical rows.");
        return;
      }

      setCsvHeaders(headers);
      setCsvData(parsedData);
      
      // Select first header as target and all other headers as features by default
      setTargetColumn(headers[headers.length - 1] || headers[0]);
      setInputFeatures(headers.slice(0, -1));
      setTablePage(0);
      setMetrics(null);
      setPredictionsList([]);
      setFeatureImportances([]);
      setStatusText(`Successfully uploaded ${file.name} (${parsedData.length} rows loaded)`);
    };
    reader.readAsText(file);
  };

  // Feature tickboxes toggler
  const handleFeatureToggle = (feature: string) => {
    if (inputFeatures.includes(feature)) {
      if (inputFeatures.length > 1) {
        setInputFeatures(inputFeatures.filter(f => f !== feature));
      }
    } else {
      setInputFeatures([...inputFeatures, feature]);
    }
  };

  // ML model execution pipeline
  const runModelTraining = () => {
    if (csvData.length === 0) return;
    setIsTraining(true);
    setStatusText("Preparing train/test split datasets...");

    // Timeout allows DOM loading spinners to redraw smoothly
    setTimeout(() => {
      try {
        const n = csvData.length;
        // Create randomized index mapping for true Split
        const indices = Array.from({ length: n }, (_, i) => i).sort(() => Math.random() - 0.5);
        const splitIdx = Math.floor(n * trainRatio);

        const trainIndices = indices.slice(0, splitIdx);
        const testIndices = indices.slice(splitIdx);

        if (trainIndices.length === 0 || testIndices.length === 0) {
          throw new Error("Insufficient data points for splitting. Adjust your partition ratio.");
        }

        // Extrapolate double dimensions matrix input
        const extractX = (inds: number[]): number[][] => {
          return inds.map(idx => {
            const rowObj = csvData[idx];
            return inputFeatures.map(f => rowObj[f] || 0);
          });
        };

        const extractY = (inds: number[]): number[] => {
          return inds.map(idx => csvData[idx][targetColumn] || 0);
        };

        const X_train = extractX(trainIndices);
        const y_train = extractY(trainIndices);
        const X_test = extractX(testIndices);
        const y_test = extractY(testIndices);

        let testPreds: number[] = [];
        let importances: FeatureImportance[] = [];

        if (modelType === "linear") {
          setStatusText("Training Linear Regression (Standardized GD)...");
          const model = new LinearRegression();
          model.train(X_train, y_train, 600, 0.015);
          testPreds = model.predict(X_test);
          importances = model.getFeatureImportances(inputFeatures);
        } else if (modelType === "rf") {
          setStatusText(`Training Random Forest Regressor (${numEstimators} trees)...`);
          const model = new RandomForestRegressor(numEstimators, maxDepth);
          model.train(X_train, y_train);
          testPreds = model.predict(X_test);
          importances = model.getFeatureImportances(inputFeatures, inputFeatures.length);
        } else if (modelType === "gb") {
          setStatusText(`Training Gradient Boosting Regressor (${numEstimators} updates)...`);
          const model = new GradientBoostingRegressor(numEstimators, 0.1, maxDepth);
          model.train(X_train, y_train);
          testPreds = model.predict(X_test);
          importances = model.getFeatureImportances(inputFeatures, inputFeatures.length);
        }

        // Calculate precision statistics
        const computedMetrics = calculateMetrics(y_test, testPreds);
        setMetrics(computedMetrics);

        // Build prediction list with residuals
        const predResults: PredictionResult[] = y_test.map((act, i) => ({
          actual: act,
          predicted: testPreds[i],
          residual: act - testPreds[i]
        }));
        setPredictionsList(predResults);

        // Sort feature importances
        setFeatureImportances(importances);
        setStatusText("Model optimization and testing successfully completed!");
      } catch (err: any) {
        setStatusText(`Training error: ${err.message || err}`);
      } finally {
        setIsTraining(false);
      }
    }, 150);
  };

  // Recharts actual versus predicted scatter chart series
  const scatterData = useMemo(() => {
    return predictionsList.map((item, idx) => ({
      id: idx,
      actual: parseFloat(item.actual.toFixed(4)),
      predicted: parseFloat(item.predicted.toFixed(4))
    }));
  }, [predictionsList]);

  // Ideal fit line for scatter plot overlays
  const diagonalLine = useMemo(() => {
    if (predictionsList.length === 0) return [];
    const values = predictionsList.flatMap(p => [p.actual, p.predicted]);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    return [
      { actual: minVal, predicted: minVal },
      { actual: maxVal, predicted: maxVal }
    ];
  }, [predictionsList]);

  // Chart data for Feature Importances
  const importanceChartData = useMemo(() => {
    return featureImportances.map(f => ({
      Feature: f.featureName,
      "Importance (%)": parseFloat((f.importance * 100).toFixed(2))
    }));
  }, [featureImportances]);

  // Color selection for individual importance bars
  const colors = ["#6366f1", "#4f46e5", "#4338ca", "#3730a3", "#312e81"];

  // Prediction table data length
  const totalPages = Math.ceil(predictionsList.length / pageSize);
  const paginatedPredictions = useMemo(() => {
    const start = tablePage * pageSize;
    return predictionsList.slice(start, start + pageSize);
  }, [predictionsList, tablePage]);

  return (
    <div className={`flex flex-col h-full font-sans p-5 space-y-4 overflow-auto scrollbar-thin transition-colors duration-200 ${
      darkMode ? "bg-slate-950 text-slate-100" : "bg-slate-50 text-slate-900"
    }`}>
      
      {/* Prediction Mode Selector */}
      <div className={`p-4 rounded-xl border flex flex-col md:flex-row items-start md:items-center justify-between gap-4 transition-all ${
        darkMode ? "bg-slate-900 border-slate-800 text-slate-100" : "bg-white border-slate-200 shadow-sm"
      }`}>
        <div className="space-y-1 bg-transparent">
          <div className="flex items-center gap-2">
            <Cpu size={16} className="text-indigo-400" />
            <h3 className="text-sm font-bold">Predictive Machine Learning Workflow</h3>
          </div>
          <p className={`text-xs ${darkMode ? "text-slate-400" : "text-slate-505"}`}>
            Choose whether to perform on-the-fly training using mathematically derived features from active DCC files, or load a custom external physical props CSV.
          </p>
        </div>

        <div className="flex rounded-lg border border-slate-205 dark:border-slate-800 p-0.5 bg-slate-100 dark:bg-slate-950 font-mono text-[10px] uppercase tracking-wide">
          <button
            onClick={() => setPredictionMode("dcc_on_the_fly")}
            disabled={!dccData || dccData.length === 0}
            className={`px-3 py-1.5 rounded-md font-bold cursor-pointer transition-all flex items-center gap-1.5 disabled:opacity-30 ${
              predictionMode === "dcc_on_the_fly"
                ? "bg-indigo-600 text-white font-extrabold shadow-sm"
                : (darkMode ? "text-slate-400 hover:bg-slate-900" : "text-slate-600 hover:bg-white")
            }`}
            title={(!dccData || dccData.length === 0) ? "Please run calculations in the workspace first" : "Train on current DCC files"}
          >
            <Layers size={11} />
            <span>Active DCC Matrix features</span>
          </button>
          <button
            onClick={() => setPredictionMode("csv_upload")}
            className={`px-3 py-1.5 rounded-md font-bold cursor-pointer transition-all flex items-center gap-1.5 ${
              predictionMode === "csv_upload"
                ? "bg-indigo-600 text-white font-extrabold shadow-sm"
                : (darkMode ? "text-slate-400 hover:bg-slate-900" : "text-slate-600 hover:bg-white")
            }`}
          >
            <Upload size={11} />
            <span>Structural CSV Mode</span>
          </button>
        </div>
      </div>

      {predictionMode === "csv_upload" ? (
        /* CSV upload banner */
        <div className={`border p-4 flex flex-wrap items-center justify-between gap-4 rounded-xl transition-all duration-300 ${
          darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200 shadow-sm"
        }`}>
          <div className="flex items-center gap-3">
            <div className={`w-9 h-9 rounded-md flex items-center justify-center ${
              darkMode ? "bg-indigo-500/10 text-indigo-400" : "bg-indigo-50 text-indigo-650"
            }`}>
              <Upload size={18} />
            </div>
            <div>
              <h3 className={`text-sm font-bold ${darkMode ? "text-slate-100" : "text-slate-800"}`}>Upload Target Dataset</h3>
              <p className={`text-xs ${darkMode ? "text-slate-400" : "text-slate-505"}`}>{csvFileName} | {csvData.length} records</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <label className="bg-indigo-600 hover:bg-indigo-700 text-white text-xs px-4 py-2.5 rounded-lg font-bold cursor-pointer transition-all shadow-sm active:scale-[0.98]">
              Browse CSV
              <input
                type="file"
                accept=".csv"
                onChange={handleCSVUpload}
                className="hidden"
              />
            </label>
            <button 
              onClick={() => {
                const res = generateSampleCSV();
                setCsvHeaders(res.headers);
                setCsvData(res.data);
                setCsvFileName("protein_dynamics_sample.csv (Restored)");
                setMetrics(null);
                setPredictionsList([]);
                setFeatureImportances([]);
                setTablePage(0);
                setStatusText("Restored synthetic biomolecular dynamic simulation dataset.");
              }}
              className={`text-xs px-3.5 py-2.5 rounded-lg border font-bold transition-all cursor-pointer ${
                darkMode 
                  ? "text-slate-300 border-slate-700 hover:border-slate-600 hover:bg-slate-800" 
                  : "text-slate-600 border-slate-200 hover:border-slate-300 hover:bg-slate-100"
              }`}
            >
              Reset to Sample
            </button>
          </div>
        </div>
      ) : (
        /* Dynamic compiled banner */
        <div className={`border p-4 flex flex-wrap items-center justify-between gap-4 rounded-xl transition-all duration-300 ${
          darkMode ? "bg-gradient-to-r from-emerald-950/20 via-slate-905 to-slate-900 border-emerald-900/40" : "bg-white border-slate-202 shadow-sm"
        }`}>
          <div className="flex items-center gap-3">
            <div className={`w-9 h-9 rounded-md flex items-center justify-center ${
              darkMode ? "bg-emerald-500/10 text-emerald-400" : "bg-emerald-50 text-emerald-700"
            }`}>
              <Cpu size={18} />
            </div>
            <div>
              <h3 className={`text-sm font-bold ${darkMode ? "text-slate-100" : "text-slate-800"}`}>Active DCC Mathematical Features Compiled</h3>
              <p className={`text-xs opacity-80 ${darkMode ? "text-slate-300" : "text-slate-505"}`}>
                {csvFileName} | {csvData.length} pairs. Detailed physicochemical metrics like <strong>Hydrogen Bonds & Electrostatics are removed</strong> to prevent physical data dependencies.
              </p>
            </div>
          </div>
          <span className={`text-[10px] uppercase font-mono font-bold border rounded px-2.5 py-1.5 select-none ${
            darkMode ? "bg-emerald-500/10 border-emerald-500/25 text-emerald-400" : "bg-emerald-50 border-emerald-200 text-emerald-800"
          }`}>
            Live Syncing
          </span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left Side: Columns selection & hyperparameters config */}
        <div className={`lg:col-span-4 border rounded-xl p-5 space-y-4 flex flex-col justify-between transition-all duration-300 ${
          darkMode ? "bg-slate-900 border-slate-800 text-slate-300" : "bg-white border-slate-200 shadow-sm text-slate-700"
        }`}>
          <div className="space-y-4">
            <div className={`flex items-center gap-2 border-b pb-3 ${darkMode ? "border-slate-800" : "border-slate-150"}`}>
              <Cpu size={16} className="text-indigo-500" />
              <h2 className={`text-xs uppercase font-extrabold tracking-wider ${darkMode ? "text-slate-305" : "text-slate-800"}`}>Configure Model</h2>
            </div>

            {/* Target Column Select */}
            <div className="space-y-1.5">
              <label className={`text-xs font-semibold ${darkMode ? "text-slate-405" : "text-slate-600"}`}>Target Variable (y)</label>
              <select
                value={targetColumn}
                onChange={(e) => {
                  setTargetColumn(e.target.value);
                  // Safeguard: remove target column from inputs
                  setInputFeatures(csvHeaders.filter(h => h !== e.target.value));
                }}
                className={`w-full border rounded-lg px-3 py-2 text-xs focus:outline-none transition-all ${
                  darkMode 
                    ? "bg-slate-950 border-slate-800 text-slate-200 focus:border-indigo-500" 
                    : "bg-slate-50 border-slate-200 text-slate-800 focus:border-indigo-600"
                }`}
              >
                {csvHeaders.map(h => (
                  <option key={h} className={darkMode ? "bg-slate-950" : "bg-white"} value={h}>{h}</option>
                ))}
              </select>
            </div>

            {/* Features multicheck */}
            <div className="space-y-1.5">
              <label className={`text-xs font-semibold block mb-1 ${darkMode ? "text-slate-405" : "text-slate-600"}`}>Input Features (X)</label>
              <div className={`border rounded-lg p-3 max-h-[160px] overflow-auto flex flex-col gap-2 scrollbar-thin ${
                darkMode ? "bg-slate-950 border-slate-800" : "bg-slate-50 border-slate-200"
              }`}>
                {csvHeaders.map(h => {
                  const isTarget = h === targetColumn;
                  const isChecked = inputFeatures.includes(h);
                  return (
                    <label 
                      key={h} 
                      className={`flex items-center gap-2.5 text-xs py-0.5 select-none ${isTarget ? "opacity-30 cursor-not-allowed" : "cursor-pointer"}`}
                    >
                      <input
                        type="checkbox"
                        checked={isChecked && !isTarget}
                        disabled={isTarget}
                        onChange={() => handleFeatureToggle(h)}
                        className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-505"
                      />
                      <span className={isChecked && !isTarget ? (darkMode ? "text-indigo-400 font-bold" : "text-indigo-600 font-bold") : (darkMode ? "text-slate-400" : "text-slate-600")}>
                        {h} {isTarget && "(Target)"}
                      </span>
                    </label>
                  );
                })}
              </div>
            </div>

            {/* Train Split Ratio slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs font-semibold">
                <span className={darkMode ? "text-slate-400" : "text-slate-600"}>Train/Test Split ratio</span>
                <span className={darkMode ? "text-slate-200" : "text-slate-800"}>{(trainRatio * 100).toFixed(0)}% / {((1 - trainRatio) * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="0.9"
                step="0.05"
                value={trainRatio}
                onChange={(e) => setTrainRatio(parseFloat(e.target.value))}
                className="w-full h-1.5 rounded-lg appearance-none cursor-pointer accent-indigo-500 bg-slate-300 dark:bg-slate-800"
              />
            </div>

            {/* ML Models type check */}
            <div className="space-y-1.5">
              <label className={`text-xs font-semibold block mb-1 ${darkMode ? "text-slate-450" : "text-slate-600"}`}>Model Family</label>
              <div className="grid grid-cols-1 gap-1.5">
                {[
                  { id: "linear", label: "Linear Regression" },
                  { id: "rf", label: "Random Forest Regressor" },
                  { id: "gb", label: "Gradient Boosting Regressor" }
                ].map(m => (
                  <label 
                    key={m.id} 
                    className={`flex items-center justify-between border rounded-lg p-2.5 text-xs font-semibold cursor-pointer transition-colors ${
                      modelType === m.id 
                        ? (darkMode ? "bg-indigo-900/20 border-indigo-500 text-indigo-300" : "bg-indigo-50 border-indigo-600 text-indigo-700")
                        : (darkMode ? "bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700" : "bg-slate-50 border-slate-200 text-slate-600 hover:border-slate-350")
                    }`}
                  >
                    <span>{m.label}</span>
                    <input
                      type="radio"
                      name="model-family"
                      checked={modelType === m.id}
                      onChange={() => setModelType(m.id as any)}
                      className="text-indigo-600 focus:ring-indigo-550 border-slate-8 00"
                    />
                  </label>
                ))}
              </div>
            </div>

            {/* Optional RF/GB Hyperparameters */}
            {modelType !== "linear" && (
              <div className={`grid grid-cols-2 gap-3 p-3 border rounded-lg ${
                darkMode ? "bg-slate-950/60 border-slate-800" : "bg-slate-50 border-slate-200"
              }`}>
                <div className="space-y-1">
                  <label className="text-[9px] text-slate-500 uppercase font-extrabold block">Estimators (Trees)</label>
                  <select
                    value={numEstimators}
                    onChange={(e) => setNumEstimators(parseInt(e.target.value))}
                    className={`w-full border rounded px-2 py-1 text-xs focus:outline-none ${
                        darkMode ? "bg-slate-900 border-slate-800 text-slate-200" : "bg-white border-slate-200 text-slate-800"
                    }`}
                  >
                    <option value="5">5 trees</option>
                    <option value="15">15 trees</option>
                    <option value="30">30 trees</option>
                    <option value="55">55 trees</option>
                  </select>
                </div>
                <div className="space-y-1">
                  <label className="text-[9px] text-slate-500 uppercase font-extrabold block">Max Depth</label>
                  <select
                    value={maxDepth}
                    onChange={(e) => setMaxDepth(parseInt(e.target.value))}
                    className={`w-full border rounded px-2 py-1 text-xs focus:outline-none ${
                        darkMode ? "bg-slate-900 border-slate-800 text-slate-200" : "bg-white border-slate-200 text-slate-800"
                    }`}
                  >
                    <option value="3">3 layers</option>
                    <option value="5">5 layers</option>
                    <option value="8">8 layers</option>
                  </select>
                </div>
              </div>
            )}
          </div>

          <button
            onClick={runModelTraining}
            disabled={isTraining}
            className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs py-3 rounded-lg mt-4 shadow-md transition-all flex items-center justify-center gap-2 cursor-pointer duration-150 active:scale-[0.98]"
          >
            <Play size={13} fill="currentColor" />
            {isTraining ? "COMPUTING OPTIMIZATIONS..." : "TRAIN & TEST MODEL"}
          </button>
        </div>

        {/* Right Side: Charts & performance evaluation outputs */}
        <div className="lg:col-span-8 flex flex-col space-y-4">
          
          {/* Status logs */}
          <div className={`p-3 rounded-lg text-xs flex items-center justify-between border transition-colors ${
            darkMode ? "bg-slate-900 border-slate-800 text-slate-300" : "bg-white border-slate-200 text-slate-705 shadow-xs"
          }`}>
            <div className="flex items-center gap-2">
              <Activity size={13} className="text-emerald-500 animate-pulse" />
              <span>Status: <strong className={darkMode ? "text-slate-100" : "text-slate-800"}>{statusText}</strong></span>
            </div>
            {metrics && (
              <span className={`text-[10px] px-2 py-0.5 rounded border font-extrabold uppercase tracking-widest ${
                darkMode ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" : "bg-emerald-50 text-emerald-700 border-emerald-200"
              }`}>Verified</span>
            )}
          </div>

          {/* Core Accuracy metrics cards */}
          {metrics && (
            <div className="grid grid-cols-3 gap-3">
              <div className={`border rounded-xl p-4 text-center transition-all ${
                darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200 shadow-xs"
              }`}>
                <span className={`text-[9px] uppercase tracking-wider font-extrabold block mb-0.5 ${darkMode ? "text-slate-400" : "text-slate-500"}`}>R² Coefficient</span>
                <p className="text-xl font-bold text-indigo-600 dark:text-indigo-400 font-mono">
                  {metrics.r2.toFixed(4)}
                </p>
              </div>
              <div className={`border rounded-xl p-4 text-center transition-all ${
                darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200 shadow-xs"
              }`}>
                <span className={`text-[9px] uppercase tracking-wider font-extrabold block mb-0.5 ${darkMode ? "text-slate-400" : "text-slate-500"}`}>MAE (Mean Abs)</span>
                <p className="text-xl font-bold text-emerald-600 dark:text-emerald-400 font-mono">
                  {metrics.mae.toFixed(4)}
                </p>
              </div>
              <div className={`border rounded-xl p-4 text-center transition-all ${
                darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200 shadow-xs"
              }`}>
                <span className={`text-[9px] uppercase tracking-wider font-extrabold block mb-0.5 ${darkMode ? "text-slate-400" : "text-slate-500"}`}>RMSE (Root MSE)</span>
                <p className="text-xl font-bold text-rose-600 dark:text-rose-450 font-mono">
                  {metrics.rmse.toFixed(4)}
                </p>
              </div>
            </div>
          )}

          {/* Graphics layouts panel */}
          {predictionsList.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              
              {/* Actual vs Predicted Plot */}
              <div className={`border rounded-xl p-4 flex flex-col h-[320px] transition-all ${
                darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200 shadow-xs"
              }`}>
                <h3 className={`text-xs font-bold uppercase tracking-wide mb-3 flex items-center gap-1.5 ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
                  <span>Actual vs Predicted Scatter</span>
                </h3>
                <div className="flex-1 min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: -20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? "#2a3341" : "#f1f5f9"} />
                      <XAxis type="number" dataKey="actual" name="Actual" stroke={darkMode ? "#94a3b8" : "#64748b"} style={{ fontSize: "10px" }} />
                      <YAxis type="number" dataKey="predicted" name="Predicted" stroke={darkMode ? "#94a3b8" : "#64748b"} style={{ fontSize: "10px" }} />
                      <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                      <Scatter name="Test Points" data={scatterData} fill="#6366f1" stroke={darkMode ? "#312e81" : "#e0e7ff"} strokeWidth={1} />
                      <Scatter name="Ideal Fit" data={diagonalLine} line stroke="#ef4444" strokeWidth={1.5} shape="none" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Feature Importance Plot */}
              <div className={`border rounded-xl p-4 flex flex-col h-[320px] transition-all ${
                darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200 shadow-xs"
              }`}>
                <h3 className={`text-xs font-bold uppercase tracking-wide mb-3 flex items-center gap-1.5 ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
                  <BarChart2 size={13} className="text-indigo-500" />
                  <span>Feature Importance (%)</span>
                </h3>
                <div className="flex-1 min-h-0">
                  {featureImportances.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={importanceChartData} layout="vertical" margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? "#2a3341" : "#f1f5f9"} />
                        <XAxis type="number" stroke={darkMode ? "#94a3b8" : "#64748b"} style={{ fontSize: "10px" }} />
                        <YAxis type="category" dataKey="Feature" stroke={darkMode ? "#94a3b8" : "#64748b"} width={90} style={{ fontSize: "10px", fontFamily: "monospace" }} />
                        <Tooltip />
                        <Bar dataKey="Importance (%)" fill="#6366f1" radius={[0, 4, 4, 0]}>
                          {importanceChartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex items-center justify-center h-full text-xs text-slate-500">
                      Feature importance calculation unsupported for single feature sets.
                    </div>
                  )}
                </div>
              </div>

              {/* Prediction Table */}
              <div className={`md:col-span-2 border rounded-xl p-5 flex flex-col select-all transition-all ${
                darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200 shadow-xs"
              }`}>
                <div className={`flex items-center justify-between border-b pb-3 mb-4 ${darkMode ? "border-slate-800" : "border-slate-150"}`}>
                  <h3 className={`text-xs font-bold uppercase tracking-wide flex items-center gap-1.5 ${darkMode ? "text-slate-300" : "text-slate-700"}`}>
                    <ListCollapse size={13} className="text-slate-400" />
                    <span>Prediction Output Matrix (Test split results)</span>
                  </h3>
                  <span className={`text-[10px] font-mono ${darkMode ? "text-slate-500" : "text-slate-400"}`}>Row {tablePage * pageSize + 1} - {Math.min((tablePage + 1) * pageSize, predictionsList.length)} of {predictionsList.length}</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className={`border-b text-[11px] font-mono ${darkMode ? "border-slate-800 text-slate-500" : "border-slate-150 text-slate-400"}`}>
                        <th className="py-2.5 px-3">Test Node ID</th>
                        <th className="py-2.5 px-3 text-right">Actual Value</th>
                        <th className="py-2.5 px-3 text-right">Predicted Value</th>
                        <th className="py-2.5 px-3 text-right">Residual Error</th>
                        <th className="py-2.5 px-3 text-right">Absolute Error</th>
                      </tr>
                    </thead>
                    <tbody className={`divide-y font-mono ${
                      darkMode ? "divide-slate-850 text-slate-300" : "divide-slate-100 text-slate-700"
                    }`}>
                      {paginatedPredictions.map((pred, i) => {
                        const rowIdx = tablePage * pageSize + i;
                        const ae = Math.abs(pred.residual);
                        return (
                          <tr key={rowIdx} className={darkMode ? "hover:bg-slate-850/40" : "hover:bg-slate-50"}>
                            <td className="py-2.5 px-3 text-slate-500 font-bold">#{rowIdx + 1}</td>
                            <td className="py-2.5 px-3 text-right">{pred.actual.toFixed(6)}</td>
                            <td className="py-2.5 px-3 text-right text-indigo-600 dark:text-indigo-400 font-semibold">{pred.predicted.toFixed(6)}</td>
                            <td className={`py-2.5 px-3 text-right ${pred.residual >= 0 ? "text-emerald-500" : "text-rose-400"}`}>
                              {pred.residual >= 0 ? "+" : ""}{pred.residual.toFixed(6)}
                            </td>
                            <td className="py-2.5 px-3 text-right opacity-80">{ae.toFixed(6)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Pagination footer controls */}
                {totalPages > 1 && (
                  <div className={`flex items-center justify-end gap-1.5 mt-4 pt-3 border-t ${
                    darkMode ? "border-slate-850" : "border-slate-150"
                  }`}>
                    <button
                      onClick={() => setTablePage(p => Math.max(0, p - 1))}
                      disabled={tablePage === 0}
                      className={`text-xs font-semibold px-3 py-1.5 rounded-lg border transition-all cursor-pointer ${
                        darkMode 
                          ? "text-slate-300 border-slate-800 hover:bg-slate-850 hover:border-slate-700 disabled:opacity-20" 
                          : "text-slate-600 border-slate-200 hover:bg-slate-50 hover:border-slate-300 disabled:opacity-20"
                      }`}
                    >
                      Previous
                    </button>
                    <span className="text-[11px] text-slate-500 px-2">Page {tablePage + 1} of {totalPages}</span>
                    <button
                      onClick={() => setTablePage(p => Math.min(totalPages - 1, p + 1))}
                      disabled={tablePage === totalPages - 1}
                      className={`text-xs font-semibold px-3 py-1.5 rounded-lg border transition-all cursor-pointer ${
                        darkMode 
                          ? "text-slate-300 border-slate-800 hover:bg-slate-850 hover:border-slate-700 disabled:opacity-20" 
                          : "text-slate-600 border-slate-200 hover:bg-slate-50 hover:border-slate-300 disabled:opacity-20"
                      }`}
                    >
                      Next
                    </button>
                  </div>
                )}
              </div>

            </div>
          ) : (
            <div className={`flex flex-col items-center justify-center border-2 border-dashed rounded-xl p-16 text-center ${
              darkMode ? "border-slate-800 text-slate-500" : "border-slate-250 text-slate-400 bg-white shadow-xs"
            }`}>
              <Cpu size={32} className={`mb-3 animate-pulse ${darkMode ? "text-slate-700" : "text-indigo-200"}`} />
              <p className={`text-xs font-bold mb-1 ${darkMode ? "text-slate-400" : "text-slate-700"}`}>Model Untrained</p>
              <p className={`text-[11px] max-w-sm ${darkMode ? "text-slate-500" : "text-slate-400"}`}>
                Select features and click "Train & Test Model". The pipeline splits loaded records into a test group to calculate R², MAE, RMSE, and graph feature scaling.
              </p>
            </div>
          )}

        </div>
      </div>

    </div>
  );
}
