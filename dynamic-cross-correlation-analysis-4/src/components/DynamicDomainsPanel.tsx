import React, { useState, useMemo } from "react";
import { Layers, Activity, Info, ShieldCheck, Sliders, Zap, Anchor } from "lucide-react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from "recharts";

interface HeatmapDataPoint {
  Residue_i: number;
  Residue_j: number;
  Mean_Correlation: number;
  Std_Correlation: number;
  Positive_Probability: number;
  Negative_Probability: number;
}

interface DynamicDomainsPanelProps {
  dccData: HeatmapDataPoint[];
  darkMode?: boolean;
}

// Lightweight K-means clustering implementation
function kmeans(data: number[][], k: number, maxIterations: number = 30): number[] {
  const n = data.length;
  if (n === 0) return [];
  const d = data[0].length;

  // Initialize centroids using K-means++ style spread
  let centroids: number[][] = [];
  const usedIndices = new Set<number>();
  
  // Pick first seed centroid randomly
  let idx = Math.floor(Math.random() * n);
  usedIndices.add(idx);
  centroids.push([...data[idx]]);

  const distSq = (a: number[], b: number[]) => {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i];
      sum += diff * diff;
    }
    return sum;
  };

  // Find other seeds based on squared distance to nearest centroid
  while (centroids.length < k && usedIndices.size < n) {
    let maxDist = -1;
    let nextIdx = -1;
    for (let i = 0; i < n; i++) {
      if (usedIndices.has(i)) continue;
      let minDistToC = Infinity;
      for (const cent of centroids) {
        const d_sq = distSq(data[i], cent);
        if (d_sq < minDistToC) {
          minDistToC = d_sq;
        }
      }
      if (minDistToC > maxDist) {
        maxDist = minDistToC;
        nextIdx = i;
      }
    }
    if (nextIdx !== -1) {
      usedIndices.add(nextIdx);
      centroids.push([...data[nextIdx]]);
    } else {
      break;
    }
  }

  // Fallback if we have fewer elements than k
  while (centroids.length < k) {
    const fallbackIdx = Math.floor(Math.random() * n);
    centroids.push(data[fallbackIdx] ? [...data[fallbackIdx]] : new Array(d).fill(0));
  }

  let assignments = new Array(n).fill(-1);
  let changed = true;
  let iteration = 0;

  while (changed && iteration < maxIterations) {
    changed = false;
    iteration++;

    // Assign points to nearest centroid
    for (let i = 0; i < n; i++) {
      let minD = Infinity;
      let minK = 0;
      for (let j = 0; j < k; j++) {
        const d_sq = distSq(data[i], centroids[j]);
        if (d_sq < minD) {
          minD = d_sq;
          minK = j;
        }
      }
      if (assignments[i] !== minK) {
        assignments[i] = minK;
        changed = true;
      }
    }

    // Recalculate centroids
    const counts = new Array(k).fill(0);
    const sums = Array.from({ length: k }, () => new Array(d).fill(0));

    for (let i = 0; i < n; i++) {
      const c = assignments[i];
      counts[c]++;
      for (let j = 0; j < d; j++) {
        sums[c][j] += data[i][j];
      }
    }

    for (let j = 0; j < k; j++) {
      if (counts[j] > 0) {
        for (let jDim = 0; jDim < d; jDim++) {
          centroids[j][jDim] = sums[j][jDim] / counts[j];
        }
      }
    }
  }

  return assignments;
}

// continuous range formatting: [1, 2, 3, 5, 8, 9] -> "1-3, 5, 8-9"
function formatResidueRanges(residues: number[]): string {
  if (residues.length === 0) return "None";
  const sorted = [...residues].sort((a, b) => a - b);
  const ranges: string[] = [];
  let rStart = sorted[0];
  let rPrev = sorted[0];

  for (let idx = 1; idx < sorted.length; idx++) {
    const rCurr = sorted[idx];
    if (rCurr === rPrev + 1) {
      rPrev = rCurr;
    } else {
      ranges.push(rStart === rPrev ? `${rStart}` : `${rStart}-${rPrev}`);
      rStart = rCurr;
      rPrev = rCurr;
    }
  }
  ranges.push(rStart === rPrev ? `${rStart}` : `${rStart}-${rPrev}`);
  return ranges.join(", ");
}

export default function DynamicDomainsPanel({ dccData = [], darkMode = true }: DynamicDomainsPanelProps) {
  const [numClusters, setNumClusters] = useState<number>(3);
  const [activeAnalysisMode, setActiveAnalysisMode] = useState<"domains" | "centrality">("domains");

  // Complete analysis calculated dynamically from the active DCC matrix
  const analysis = useMemo(() => {
    if (!dccData || dccData.length === 0) return null;

    // 1. Find all participating unique residues
    const resSet = new Set<number>();
    dccData.forEach(p => {
      resSet.add(p.Residue_i);
      resSet.add(p.Residue_j);
    });
    const residuesList = Array.from(resSet).sort((a, b) => a - b);
    const n = residuesList.length;

    if (n < 4) return null;

    // Map residue index to position in matrix
    const resToIndex = new Map<number, number>();
    residuesList.forEach((r, idx) => resToIndex.set(r, idx));

    // Initialize 2D DCC Matrix
    const dccMatrix = Array.from({ length: n }, () => new Array(n).fill(1.0));
    const stdMatrix = Array.from({ length: n }, () => new Array(n).fill(0.0));

    dccData.forEach(p => {
      const idx_i = resToIndex.get(p.Residue_i);
      const idx_j = resToIndex.get(p.Residue_j);
      if (idx_i !== undefined && idx_j !== undefined) {
        dccMatrix[idx_i][idx_j] = p.Mean_Correlation;
        dccMatrix[idx_j][idx_i] = p.Mean_Correlation;
        stdMatrix[idx_i][idx_j] = p.Std_Correlation;
        stdMatrix[idx_j][idx_i] = p.Std_Correlation;
      }
    });

    // 2. Perform K-means clustering on similarity matrices 
    // Shift correlation [-1, +1] to [0, 1] for stable vector similarity space
    const simVectors = dccMatrix.map(row => row.map(v => (v + 1.0) / 2.0));
    const clusterAssignments = kmeans(simVectors, numClusters, 40);

    // Group residues by clusters
    const clusters: Record<number, number[]> = {};
    for (let c = 0; c < numClusters; c++) {
      clusters[c] = [];
    }
    clusterAssignments.forEach((clusterIdx, idx) => {
      const actualRes = residuesList[idx];
      if (clusters[clusterIdx]) {
        clusters[clusterIdx].push(actualRes);
      } else {
        clusters[clusterIdx] = [actualRes];
      }
    });

    // 3. Compute residue-level topology properties: Hubness Centrality & Toggle Hinge index
    const residueProfiles = residuesList.map((res, idx) => {
      const rowCorr = dccMatrix[idx];
      const rowStd = stdMatrix[idx];

      // Hubness = average absolute cross correlation with all other residues
      const totalAbsCorr = rowCorr.reduce((sum, v) => sum + Math.abs(v), 0) - 1.0; // Subtract self-correlation (1.0)
      const hubnessScore = totalAbsCorr / (n - 1);

      // Hinge index = average standard deviation (fluctuation) / (abs correlation + 0.1)
      // High standard deviation combined with near-zero mean correlation indicates a shifting boundary toggle point!
      let totalHingeScore = 0;
      for (let k = 0; k < n; k++) {
        if (k === idx) continue;
        totalHingeScore += rowStd[k] / (Math.abs(rowCorr[k]) + 0.1);
      }
      const hingeScore = totalHingeScore / (n - 1);

      return {
        residue: res,
        cluster: clusterAssignments[idx],
        hubness: hubnessScore,
        hinge: hingeScore,
      };
    });

    // Sort residues to find Top Hubs
    const topHubs = [...residueProfiles]
      .sort((a, b) => b.hubness - a.hubness)
      .slice(0, 8);

    // Sort residues to find Top Hinges
    const topHinges = [...residueProfiles]
      .sort((a, b) => b.hinge - a.hinge)
      .slice(0, 8);

    // Dynamic Domain Profiles
    const domainProfiles = Object.keys(clusters).map(cKey => {
      const groupIdx = parseInt(cKey, 10);
      const members = clusters[groupIdx] || [];
      
      // Calculate average interaction strength within this domain
      let internalSum = 0;
      let internalCount = 0;
      members.forEach(r => {
        const iIdx = resToIndex.get(r);
        members.forEach(r2 => {
          if (r === r2) return;
          const jIdx = resToIndex.get(r2);
          if (iIdx !== undefined && jIdx !== undefined) {
            internalSum += Math.abs(dccMatrix[iIdx][jIdx]);
            internalCount++;
          }
        });
      });
      const coherence = internalCount > 0 ? (internalSum / internalCount) : 1.0;

      // Identify most central "Key Organizer hub" in this domain
      const sortedMembers = [...members].sort((r1, r2) => {
        const h1 = residueProfiles.find(item => item.residue === r1)?.hubness || 0;
        const h2 = residueProfiles.find(item => item.residue === r2)?.hubness || 0;
        return h2 - h1;
      });

      return {
        id: groupIdx,
        residueCount: members.length,
        membersSorted: members.sort((a, b) => a - b),
        formattedRanges: formatResidueRanges(members),
        organizerHub: sortedMembers[0] || -1,
        coherenceScore: coherence,
      };
    });

    return {
      residuesList,
      clusterAssignments,
      domainProfiles,
      topHubs,
      topHinges,
      residueProfiles,
    };
  }, [dccData, numClusters]);

  // Design aesthetics domain palettes
  const clusterColors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6"];
  const clusterBgColors = [
    "bg-red-500/10 text-red-400 border-red-500/20",
    "bg-blue-500/10 text-blue-400 border-blue-500/20",
    "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    "bg-amber-500/10 text-amber-400 border-amber-500/20",
    "bg-purple-500/10 text-purple-400 border-purple-500/20",
    "bg-pink-500/10 text-pink-400 border-pink-500/20",
    "bg-teal-500/10 text-teal-400 border-teal-500/20",
  ];

  const clusterBgColorsLight = [
    "bg-red-50 text-red-700 border-red-200",
    "bg-blue-55 text-blue-700 border-blue-200",
    "bg-emerald-50 text-emerald-700 border-emerald-200",
    "bg-amber-50 hover:bg-amber-100 text-amber-800 border-amber-200",
    "bg-purple-50 text-purple-700 border-purple-200",
    "bg-pink-50 text-pink-700 border-pink-200",
    "bg-teal-50 text-teal-700 border-teal-200",
  ];

  const chartData = useMemo(() => {
    if (!analysis) return [];
    return analysis.domainProfiles.map(p => ({
      name: `Domain ${p.id + 1}`,
      "Residue Count": p.residueCount,
      "Coherence Score": parseFloat(p.coherenceScore.toFixed(3)),
    }));
  }, [analysis]);

  if (!dccData || dccData.length === 0) {
    return (
      <div className={`flex flex-col items-center justify-center p-16 text-center select-none h-full border ${
        darkMode ? "bg-slate-950 border-slate-900 text-slate-500" : "bg-slate-50 border-slate-200 text-slate-450 shadow-sm"
      }`}>
        <Layers size={45} className="text-indigo-400/30 animate-bounce mb-4" />
        <h3 className={`text-sm font-bold mb-1 ${darkMode ? "text-slate-300" : "text-slate-800"}`}>
          Dynamic Domain Identification Idle
        </h3>
        <p className="text-xs max-w-sm leading-normal">
          Please upload your dynamic cross-correlation (DCC) simulation files and run the primary calculations in the left control panel first to activate domain partitioning.
        </p>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className={`p-10 text-center text-xs ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
        Insufficient residue matrices loaded for dynamic partition clustering (minimum 4 residues required).
      </div>
    );
  }

  return (
    <div className={`flex flex-col h-full font-sans p-5 space-y-4 overflow-auto scrollbar-thin transition-colors duration-200 ${
      darkMode ? "bg-slate-950 text-slate-100" : "bg-slate-50 text-slate-900"
    }`}>
      {/* Header and Controller banner */}
      <div className={`border p-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 rounded-xl transition-all ${
        darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200 shadow-sm"
      }`}>
        <div className="flex items-center gap-3">
          <div className={`w-9 h-9 rounded-md flex items-center justify-center ${
            darkMode ? "bg-indigo-500/10 text-indigo-400" : "bg-indigo-50 text-indigo-650"
          }`}>
            <Layers size={18} />
          </div>
          <div>
            <h3 className={`text-sm font-bold ${darkMode ? "text-slate-100" : "text-slate-800"}`}>
              Dynamic Functional Domain Identification
            </h3>
            <p className={`text-xs ${darkMode ? "text-slate-400" : "text-slate-500"}`}>
              Partitioning residues into mechanically coordinated spatial domains using similarity profiles computed directly from DCC matrices.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4 border-t pt-3 md:pt-0 md:border-t-0 border-slate-800/50 w-full md:w-auto justify-between md:justify-end">
          <div className="flex items-center gap-2">
            <Sliders size={14} className="text-slate-550" />
            <span className={`text-xs ${darkMode ? "text-slate-400" : "text-slate-600"}`}>Domains (K):</span>
            <input
              type="range"
              min="2"
              max="7"
              value={numClusters}
              onChange={(e) => setNumClusters(parseInt(e.target.value, 10))}
              className="w-24 h-1.5 rounded-lg appearance-none cursor-pointer accent-indigo-500 bg-slate-300 dark:bg-slate-805"
            />
            <span className="text-xs font-bold font-mono px-1.5 py-0.5 rounded border border-slate-800 text-indigo-400 bg-slate-950">
              {numClusters}
            </span>
          </div>
          
          <div className="flex rounded-lg border border-slate-200 dark:border-slate-800 p-0.5 bg-slate-100 dark:bg-slate-950 font-mono text-[10px] uppercase tracking-wide">
            <button
              onClick={() => setActiveAnalysisMode("domains")}
              className={`px-3 py-1.5 rounded-md font-bold cursor-pointer transition-colors ${
                activeAnalysisMode === "domains"
                  ? "bg-indigo-600 text-white"
                  : (darkMode ? "text-slate-405 hover:bg-slate-900" : "text-slate-600 hover:bg-white")
              }`}
            >
              Domains
            </button>
            <button
              onClick={() => setActiveAnalysisMode("centrality")}
              className={`px-3 py-1.5 rounded-md font-bold cursor-pointer transition-colors ${
                activeAnalysisMode === "centrality"
                  ? "bg-indigo-600 text-white"
                  : (darkMode ? "text-slate-405 hover:bg-slate-900" : "text-slate-600 hover:bg-white")
              }`}
            >
              Hubs & Hinges
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        
        {/* VIEW A: DOMAIN MODULAR ANALYSIS */}
        {activeAnalysisMode === "domains" && (
          <>
            {/* Primary sequence partition visual bar */}
            <div className={`col-span-12 border rounded-xl p-5 space-y-3.5 transition-all ${
              darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200 shadow-sm"
            }`}>
              <div className="flex items-center justify-between border-b pb-2 dark:border-slate-850">
                <h4 className="text-xs font-extrabold uppercase tracking-widest flex items-center gap-1.5">
                  <Activity size={14} className="text-indigo-400" />
                  <span>Primary Sequence Dynamic Partition Ribbon</span>
                </h4>
                <span className="text-[10px] select-none text-slate-500 font-mono">
                  Colored by discovered mechanical domains (Residue 1 to {analysis.residuesList[analysis.residuesList.length - 1]})
                </span>
              </div>
              
              {/* Full horizontal sequence ribbon */}
              <div className="flex items-stretch h-8 rounded-lg overflow-hidden border border-slate-350 dark:border-slate-800 font-mono text-[10px] relative">
                {analysis.residuesList.map((res, idx) => {
                  const assignmentIdx = analysis.clusterAssignments[idx];
                  const col = clusterColors[assignmentIdx % clusterColors.length];
                  return (
                    <div
                      key={idx}
                      style={{ backgroundColor: col, flexGrow: 1 }}
                      className="group relative flex items-center justify-center border-r border-black/5 last:border-r-0 cursor-default transition-all hover:scale-y-110"
                      title={`Residue ${res} | Domain ${assignmentIdx + 1}`}
                    >
                      {analysis.residuesList.length <= 40 ? res : null}
                      
                      {/* Detailed tooltip on hover */}
                      <span className="absolute bottom-9 hidden group-hover:flex flex-col bg-slate-950 border border-slate-800 p-2 rounded text-white text-[9px] w-24 tracking-wide shadow-xl text-center pointer-events-none z-50">
                        <strong>Residue {res}</strong>
                        <span className="text-slate-400 mt-0.5">Domain {assignmentIdx + 1}</span>
                      </span>
                    </div>
                  );
                })}
              </div>

              {/* Legend scale */}
              <div className="flex flex-wrap items-center gap-x-5 py-1 text-[10px] font-bold">
                {analysis.domainProfiles.map(p => (
                  <div key={p.id} className="flex items-center gap-1.5 select-none">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: clusterColors[p.id % clusterColors.length] }} />
                    <span className={darkMode ? "text-slate-400" : "text-slate-600"}>
                      Domain {p.id + 1} ({p.residueCount} res)
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* List breakdown of discovered communities */}
            <div className="lg:col-span-7 space-y-3.5">
              <div className="flex items-center justify-between border-b pb-1 dark:border-slate-900 border-slate-205">
                <span className={`text-[11px] font-extrabold uppercase tracking-wide flex items-center gap-1.5 ${darkMode ? "text-slate-400" : "text-slate-700"}`}>
                  <ShieldCheck size={14} className="text-emerald-500" />
                  <span>Dynamic Domain Directory</span>
                </span>
              </div>

              {analysis.domainProfiles.map(p => {
                const accentCol = clusterColors[p.id % clusterColors.length];
                const bgStyle = darkMode 
                  ? clusterBgColors[p.id % clusterBgColors.length]
                  : clusterBgColorsLight[p.id % clusterBgColorsLight.length];

                return (
                  <div
                    key={p.id}
                    className={`border rounded-xl p-4.5 space-y-3.5 transition-all hover:translate-x-1 duration-200 border-l-4 ${
                      darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200 shadow-xs"
                    }`}
                    style={{ borderLeftColor: accentCol }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className={`text-[10px] font-extrabold px-2 py-0.5 rounded-md border font-mono ${bgStyle}`}>
                          DOMAIN {p.id + 1}
                        </span>
                        <span className={`text-[10px] font-mono font-bold ${darkMode ? "text-slate-500" : "text-slate-400"}`}>
                          ({p.residueCount} residues)
                        </span>
                      </div>
                      <div className="flex gap-2 text-[10px] font-mono">
                        <span className={darkMode ? "text-slate-500" : "text-slate-400"}>Coherence:</span>
                        <strong className="text-indigo-400">{parseFloat((p.coherenceScore * 100).toFixed(1))}%</strong>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs leading-relaxed">
                      <div className="space-y-1">
                        <span className="text-[9px] text-slate-500 font-extrabold uppercase tracking-wide block">
                          Sequence Portions
                        </span>
                        <p className={`font-mono leading-normal select-all bg-slate-950/40 p-2 rounded-lg border border-slate-200/5 ${darkMode ? "text-slate-300" : "text-slate-700 bg-slate-50 border-slate-100"}`}>
                          {p.formattedRanges}
                        </p>
                      </div>
                      <div className="space-y-1.5 flex flex-col justify-center">
                        <div className="flex items-center justify-between text-[11px]">
                          <span className={`${darkMode ? "text-slate-400" : "text-slate-500"} font-semibold flex items-center gap-1`}>
                            <Anchor size={11} className="text-slate-500" />
                            Organizer Hub:
                          </span>
                          <span className="font-mono font-bold text-indigo-400 bg-slate-950/60 px-2 py-0.5 rounded border border-slate-800/20">
                            Residue {p.organizerHub}
                          </span>
                        </div>
                        <p className={`text-[10px] leading-normal ${darkMode ? "text-slate-500" : "text-slate-450"}`}>
                          Matches the residue inside Domain {p.id + 1} with the absolute highest dynamic coupling centrality score.
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Sizes of Discovered Domains Recharts bar */}
            <div className={`lg:col-span-5 border rounded-xl p-5 flex flex-col h-[380px] transition-all ${
              darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-205 shadow-xs"
            }`}>
              <h3 className={`text-xs font-bold uppercase tracking-wide mb-3 flex items-center gap-1.5 ${darkMode ? "text-slate-400" : "text-slate-650"}`}>
                <Info size={13} className="text-slate-400" />
                <span>Domain Cohesiveness Analysis</span>
              </h3>
              
              <div className="flex-1 min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 10, left: -25, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? "#2a3341" : "#f1f5f9"} />
                    <XAxis dataKey="name" stroke={darkMode ? "#94a3b8" : "#64748b"} style={{ fontSize: "10px", fontWeight: "bold" }} />
                    <YAxis stroke={darkMode ? "#94a3b8" : "#64748b"} style={{ fontSize: "10px" }} />
                    <Tooltip contentStyle={darkMode ? { backgroundColor: "#0f172a", borderColor: "#1e293b" } : undefined} />
                    <Bar dataKey="Residue Count" fill="#6366f1" radius={[4, 4, 0, 0]}>
                      {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={clusterColors[index % clusterColors.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="border-t border-dashed border-slate-800/40 dark:border-slate-800/80 pt-3 mt-3 text-[10px] leading-relaxed flex items-start gap-1.5">
                <Info size={11} className="text-indigo-400 flex-shrink-0 mt-0.5" />
                <span className={darkMode ? "text-slate-500" : "text-slate-450"}>
                  <strong>Coherence Score:</strong> Represents the average correlation coefficient of intra-domain residue node-pairs. High coherence values indicate structural components that translate as a synchronized, rigid structural unit.
                </span>
              </div>
            </div>
          </>
        )}

        {/* VIEW B: HUBNESS CENTRALITY & TOGGLE HINGE POINTS */}
        {activeAnalysisMode === "centrality" && (
          <div className="col-span-12 grid grid-cols-1 md:grid-cols-2 gap-4">
            
            {/* Table 1: Top Coupling Hubs */}
            <div className={`border rounded-xl p-5 flex flex-col transition-all ${
              darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200 shadow-sm"
            }`}>
              <div className="border-b pb-3 mb-4 flex items-center justify-between">
                <h4 className="text-xs font-bold uppercase tracking-wide flex items-center gap-1.5 text-indigo-505 dark:text-indigo-400">
                  <Zap size={13} className="text-indigo-505" />
                  <span>Dynamic Organization Hubs (Integrators)</span>
                </h4>
                <span className="text-[9px] text-slate-500 font-mono">Most dynamically coupled</span>
              </div>

              <div className="overflow-x-auto flex-1">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className={`border-b text-[10px] font-mono ${darkMode ? "border-slate-800 text-slate-500" : "border-slate-150 text-slate-400"}`}>
                      <th className="py-2 px-3">Residue ID</th>
                      <th className="py-2 px-3">Domain</th>
                      <th className="py-2 px-3 text-right">Hub Power Centrality</th>
                      <th className="py-2 px-3 text-right">Integrative Role</th>
                    </tr>
                  </thead>
                  <tbody className={`divide-y font-mono ${darkMode ? "divide-slate-850 text-slate-300" : "divide-slate-100 text-slate-700"}`}>
                    {analysis.topHubs.map((node, i) => {
                      const colorStyle = darkMode 
                        ? clusterBgColors[node.cluster % clusterBgColors.length]
                        : clusterBgColorsLight[node.cluster % clusterBgColorsLight.length];

                      return (
                        <tr key={i} className={darkMode ? "hover:bg-slate-850/40" : "hover:bg-slate-50"}>
                          <td className="py-2.5 px-3 font-bold text-slate-200 dark:text-slate-100 italic">#{node.residue}</td>
                          <td className="py-2.5 px-3">
                            <span className={`text-[9px] px-2 py-0.5 rounded font-bold border ${colorStyle}`}>
                              DOM {node.cluster + 1}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-right text-indigo-455 dark:text-indigo-400 font-bold font-mono">
                            {node.hubness.toFixed(4)}
                          </td>
                          <td className="py-2.5 px-3 text-right text-[10px] opacity-75 font-sans">
                            {i === 0 ? "Global Allosteric Master" : i < 3 ? "Core Dom Integrator" : "Backbone Anchor"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <div className="border-t border-dashed dark:border-slate-850 pt-3.5 mt-3.5 text-[10px] leading-normal flex items-start gap-1.5">
                <Info size={11} className="text-indigo-400 flex-shrink-0 mt-0.5" />
                <span className={darkMode ? "text-slate-500" : "text-slate-450"}>
                  <strong>Dynamic Hubs</strong> interact strongly with a high proportion of the remaining residues. These are usually structural core elements (e.g. rigid hydrophobic burial cores) that disperse structural signals and are highly vulnerable to allosteric mutagenesis.
                </span>
              </div>
            </div>

            {/* Table 2: Top Hinge Point Toggle junction residues */}
            <div className={`border rounded-xl p-5 flex flex-col transition-all ${
              darkMode ? "bg-slate-900 border-slate-800" : "bg-white border-slate-200 shadow-sm"
            }`}>
              <div className="border-b pb-3 mb-4 flex items-center justify-between">
                <h4 className="text-xs font-bold uppercase tracking-wide flex items-center gap-1.5 text-amber-605 dark:text-amber-400">
                  <Activity size={13} className="text-amber-500" />
                  <span>Conformational Hinge-Junctions (Toggle Sites)</span>
                </h4>
                <span className="text-[9px] text-slate-500 font-mono">High variance toggle points</span>
              </div>

              <div className="overflow-x-auto flex-1">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className={`border-b text-[10px] font-mono ${darkMode ? "border-slate-800 text-slate-500" : "border-slate-150 text-slate-400"}`}>
                      <th className="py-2 px-3">Residue ID</th>
                      <th className="py-2 px-3">Domain</th>
                      <th className="py-2 px-3 text-right">Bistability Toggle Index</th>
                      <th className="py-2 px-3 text-right">Physical Role</th>
                    </tr>
                  </thead>
                  <tbody className={`divide-y font-mono ${darkMode ? "divide-slate-850 text-slate-300" : "divide-slate-100 text-slate-700"}`}>
                    {analysis.topHinges.map((node, i) => {
                      const colorStyle = darkMode 
                        ? clusterBgColors[node.cluster % clusterBgColors.length]
                        : clusterBgColorsLight[node.cluster % clusterBgColorsLight.length];

                      return (
                        <tr key={i} className={darkMode ? "hover:bg-slate-850/40" : "hover:bg-slate-50"}>
                          <td className="py-2.5 px-3 font-bold text-slate-200 dark:text-slate-100 italic">#{node.residue}</td>
                          <td className="py-2.5 px-3">
                            <span className={`text-[9px] px-2 py-0.5 rounded font-bold border ${colorStyle}`}>
                              DOM {node.cluster + 1}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-right text-amber-505 dark:text-amber-400 font-bold font-mono">
                            {node.hinge.toFixed(3)}
                          </td>
                          <td className="py-2.5 px-3 text-right text-[10px] opacity-75 font-sans">
                            {i === 0 ? "Primary Elastic Pivot" : i < 3 ? "Mechanical Joint" : "Flexible Portal Loop"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <div className="border-t border-dashed dark:border-slate-850 pt-3.5 mt-3.5 text-[10px] leading-normal flex items-start gap-1.5">
                <Info size={11} className="text-amber-400 flex-shrink-0 mt-0.5" />
                <span className={darkMode ? "text-slate-500" : "text-slate-450"}>
                  <strong>Conformational Hinges</strong> possess low absolute average correlation but exceptionally high transient correlation variance (standard deviation) across conformations. These act as passive elastic pivot junctions coordinating large-scale domain breathing motions.
                </span>
              </div>
            </div>

          </div>
        )}

      </div>
    </div>
  );
}
