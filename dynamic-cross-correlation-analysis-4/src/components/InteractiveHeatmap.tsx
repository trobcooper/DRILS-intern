import React, { useRef, useEffect, useState } from "react";
import { ZoomIn, ZoomOut, Maximize2, RotateCcw, Download } from "lucide-react";

interface HeatmapDataPoint {
  Residue_i: number;
  Residue_j: number;
  Mean_Correlation: number;
  Std_Correlation: number;
  Positive_Probability: number;
  Negative_Probability: number;
}

interface InteractiveHeatmapProps {
  data: HeatmapDataPoint[];
  iMin: number;
  iMax: number;
  jMin: number;
  jMax: number;
  mode: "correlation" | "probability";
  subType: "positive" | "negative" | "combined";
  pMin?: number;
  pMax?: number;
  cMin?: number;
  cMax?: number;
  pPosCut?: number;
  pNegCut?: number;
  cPosCut?: number;
  cNegCut?: number;
  onCellDoubleClick: (i: number, j: number) => void;
  analysisMode: "Probability" | "Correlation";
  darkMode?: boolean;
}

export default function InteractiveHeatmap({
  data,
  iMin,
  iMax,
  jMin,
  jMax,
  mode,
  subType,
  pMin = 0.6,
  pMax = 0.9,
  cMin = 0.5,
  cMax = 1.0,
  pPosCut = 0.6,
  pNegCut = 0.6,
  cPosCut = 0.9,
  cNegCut = -0.5,
  onCellDoubleClick,
  analysisMode,
  darkMode = true,
}: InteractiveHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // View range boundaries (support sub-pixel zoom/pan limits)
  const [viewIMin, setViewIMin] = useState<number>(iMin);
  const [viewIMax, setViewIMax] = useState<number>(iMax + 1);
  const [viewJMin, setViewJMin] = useState<number>(jMin);
  const [viewJMax, setViewJMax] = useState<number>(jMax + 1);

  // States for panning and single-click detection
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const dragStartRef = useRef<{ 
    x: number; 
    y: number; 
    viewIMin: number; 
    viewIMax: number; 
    viewJMin: number; 
    viewJMax: number;
    startX: number;
    startY: number;
    time: number;
  } | null>(null);

  // Re-initialize viewport when dataset bounds change
  useEffect(() => {
    setViewIMin(iMin);
    setViewIMax(iMax + 1);
    setViewJMin(jMin);
    setViewJMax(jMax + 1);
  }, [iMin, iMax, jMin, jMax]);

  const [hoveredCell, setHoveredCell] = useState<{
    i: number;
    j: number;
    point: HeatmapDataPoint;
    x: number;
    y: number;
  } | null>(null);

  // Resize handling
  const [dimensions, setDimensions] = useState({ width: 500, height: 400 });

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setDimensions({
          width: Math.max(150, width),
          height: Math.max(150, height),
        });
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Set lookup map for extreme performance
  const [matrixLookup, setMatrixLookup] = useState<Map<string, HeatmapDataPoint>>(new Map());

  useEffect(() => {
    const map = new Map<string, HeatmapDataPoint>();
    for (const point of data) {
      map.set(`${point.Residue_i},${point.Residue_j}`, point);
    }
    setMatrixLookup(map);
  }, [data]);

  // Red White Blue color mapper
  const getColor = (val: number, type: "correlation" | "probability") => {
    if (type === "correlation") {
      if (subType === "positive") {
        // Red scale (0 to 1)
        const intensity = Math.max(0, Math.min(1, val));
        return `rgba(239, 68, 68, ${intensity})`; // Tailwind Red-500
      } else if (subType === "negative") {
        // Blue scale (-1 to 0 or absolute in correlation)
        const intensity = Math.max(0, Math.min(1, Math.abs(val)));
        return `rgba(59, 130, 246, ${intensity})`; // Tailwind Blue-500
      } else {
        // Combined Red-Blue scale (-1 to 1)
        if (val > 0) {
          const intensity = Math.max(0, Math.min(1, val));
          return `rgba(239, 68, 68, ${intensity})`;
        } else if (val < 0) {
          const intensity = Math.max(0, Math.min(1, Math.abs(val)));
          return `rgba(59, 130, 246, ${intensity})`;
        }
        return darkMode ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.03)";
      }
    } else {
      // Probability rendering
      if (subType === "positive") {
        const intensity = Math.max(0, Math.min(1, val));
        return `rgba(244, 63, 94, ${intensity})`; // Rose
      } else if (subType === "negative") {
        const intensity = Math.max(0, Math.min(1, val));
        return `rgba(168, 85, 247, ${intensity})`; // Purple
      } else {
        // Combined Probability (Positive Red, Negative Blue proxy)
        if (val > 0) {
          return `rgba(244, 63, 94, ${val})`;
        } else if (val < 0) {
          return `rgba(168, 85, 247, ${Math.abs(val)})`;
        }
        return darkMode ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.03)";
      }
    }
  };

  // Draw heatmap matrix on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Scale canvas for high DPI (Retina) screens
    const dpr = window.devicePixelRatio || 1;
    canvas.width = dimensions.width * dpr;
    canvas.height = dimensions.height * dpr;
    ctx.scale(dpr, dpr);

    // Padding parameters for labels & ticks
    const paddingLeft = 52;
    const paddingBottom = 44;
    const paddingTop = 26;
    const paddingRight = 85;

    const plotWidth = dimensions.width - paddingLeft - paddingRight;
    const plotHeight = dimensions.height - paddingTop - paddingBottom;

    // Clear canvas
    const canvasBg = darkMode ? "#0f172a" : "#fafafa";
    const borderStroke = darkMode ? "#334155" : "#cbd5e1";
    const labelColor = darkMode ? "#cbd5e1" : "#1e293b";
    const textColor = darkMode ? "#94a3b8" : "#475569";
    const tickColor = darkMode ? "#475569" : "#cbd5e1";

    ctx.fillStyle = canvasBg;
    ctx.fillRect(0, 0, dimensions.width, dimensions.height);

    if (data.length === 0) {
      ctx.fillStyle = "#94a3b8";
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("No data loaded. Change selection criteria & run analysis.", dimensions.width / 2, dimensions.height / 2);
      return;
    }

    const colRange = viewJMax - viewJMin;
    const rowRange = viewIMax - viewIMin;

    // Render cells inside viewport
    const startR = Math.max(iMin, Math.floor(viewIMin - 1));
    const endR = Math.min(iMax, Math.ceil(viewIMax + 1));
    const startC = Math.max(jMin, Math.floor(viewJMin - 1));
    const endC = Math.min(jMax, Math.ceil(viewJMax + 1));

    for (let r = startR; r <= endR; r++) {
      const iVal = r; // Row index (Residue i)
      for (let c = startC; c <= endC; c++) {
        const jVal = c; // Col index (Residue j)

        const point = matrixLookup.get(`${iVal},${jVal}`);
        if (!point) continue;

        // Apply masking algorithms matching math in python
        let showCell = false;
        let cellColor = "rgba(255,255,255,0.02)";

        const mCorr = point.Mean_Correlation;
        const pPos = point.Positive_Probability;
        const pNeg = point.Negative_Probability;

        if (analysisMode === "Probability") {
          // Case 1: Probability driven BASIC
          const isPosFilter = pPos >= pMin && pPos <= pMax && mCorr >= cMin && mCorr <= cMax;
          const isNegFilter = pNeg >= pMin && pNeg <= pMax && mCorr <= -cMin && mCorr >= -cMax;

          if (mode === "correlation") {
            if (subType === "positive" && isPosFilter) {
              showCell = true;
              cellColor = getColor(mCorr, "correlation");
            } else if (subType === "negative" && isNegFilter) {
              showCell = true;
              cellColor = getColor(mCorr, "correlation");
            } else if (subType === "combined") {
              if (isPosFilter) {
                showCell = true;
                cellColor = getColor(mCorr, "correlation");
              } else if (isNegFilter) {
                showCell = true;
                cellColor = getColor(mCorr, "correlation");
              }
            }
          } else {
            // mode = probability
            if (subType === "positive" && isPosFilter) {
              showCell = true;
              cellColor = getColor(pPos, "probability");
            } else if (subType === "negative" && isNegFilter) {
              showCell = true;
              cellColor = getColor(pNeg, "probability");
            } else if (subType === "combined") {
              if (isPosFilter) {
                showCell = true;
                cellColor = getColor(pPos, "probability");
              } else if (isNegFilter) {
                showCell = true;
                cellColor = getColor(-pNeg, "probability"); // negative proxy
              }
            }
          }
        } else {
          // Case 2: Advanced (cutoff-based) masks
          const isPosFilter = mCorr >= cPosCut && pPos >= pPosCut;
          const isNegFilter = mCorr <= cNegCut && pNeg >= pNegCut;

          if (mode === "correlation") {
            if (subType === "positive" && isPosFilter) {
              showCell = true;
              cellColor = getColor(mCorr, "correlation");
            } else if (subType === "negative" && isNegFilter) {
              showCell = true;
              cellColor = getColor(mCorr, "correlation");
            } else if (subType === "combined") {
              if (isPosFilter) {
                showCell = true;
                cellColor = getColor(mCorr, "correlation");
              } else if (isNegFilter) {
                showCell = true;
                cellColor = getColor(mCorr, "correlation");
              }
            }
          } else {
            // mode = probability
            if (subType === "positive" && isPosFilter) {
              showCell = true;
              cellColor = getColor(pPos, "probability");
            } else if (subType === "negative" && isNegFilter) {
              showCell = true;
              cellColor = getColor(pNeg, "probability");
            } else if (subType === "combined") {
              if (isPosFilter) {
                showCell = true;
                cellColor = getColor(pPos, "probability");
              } else if (isNegFilter) {
                showCell = true;
                cellColor = getColor(-pNeg, "probability");
              }
            }
          }
        }

        if (showCell) {
          ctx.fillStyle = cellColor;
          
          // Subpixel boundaries scaled to coordinates
          const x1 = paddingLeft + ((jVal - viewJMin) / colRange) * plotWidth;
          const x2 = paddingLeft + ((jVal + 1 - viewJMin) / colRange) * plotWidth;
          const y1 = paddingTop + plotHeight - ((iVal + 1 - viewIMin) / rowRange) * plotHeight;
          const y2 = paddingTop + plotHeight - ((iVal - viewIMin) / rowRange) * plotHeight;

          // Clip parameters beautifully to inside the plot boundary
          const drawX = Math.max(paddingLeft, x1);
          const drawY = Math.max(paddingTop, y1);
          const drawW = Math.min(paddingLeft + plotWidth, x2) - drawX;
          const drawH = Math.min(paddingTop + plotHeight, y2) - drawY;

          if (drawW > 0.05 && drawH > 0.05) {
            ctx.fillRect(drawX, drawY, drawW + 0.35, drawH + 0.35); // Slight subpixel overlap to prevent thin grid spacing gap
          }
        }
      }
    }

    // Draw Axes Boarders
    ctx.strokeStyle = borderStroke;
    ctx.lineWidth = 1;
    ctx.strokeRect(paddingLeft, paddingTop, plotWidth, plotHeight);

    // Save context state for clipping ticks
    ctx.save();

    // X Axis Ticks (Residue j) - select sparse ticks for legibility based on viewport zoom
    ctx.fillStyle = textColor;
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";

    const viewCols = viewJMax - viewJMin;
    const xTickStep = Math.max(1, Math.round(viewCols / 8));
    const startTickC = Math.ceil(viewJMin);
    const endTickC = Math.floor(viewJMax);

    for (let c = startTickC; c < endTickC; c++) {
      if ((c - startTickC) % xTickStep !== 0) continue;

      const idxStr = c.toString();
      const x = paddingLeft + ((c - viewJMin) / viewCols) * plotWidth + (plotWidth / viewCols) / 2;

      if (x >= paddingLeft && x <= paddingLeft + plotWidth) {
        ctx.fillText(idxStr, x, paddingTop + plotHeight + 6);
        
        // Draw tiny ticks
        ctx.beginPath();
        ctx.moveTo(x, paddingTop + plotHeight);
        ctx.lineTo(x, paddingTop + plotHeight + 4);
        ctx.strokeStyle = tickColor;
        ctx.stroke();
      }
    }

    // Y Axis Ticks (Residue i)
    ctx.fillStyle = textColor;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    const viewRows = viewIMax - viewIMin;
    const yTickStep = Math.max(1, Math.round(viewRows / 8));
    const startTickR = Math.ceil(viewIMin);
    const endTickR = Math.floor(viewIMax);

    for (let r = startTickR; r < endTickR; r++) {
      if ((r - startTickR) % yTickStep !== 0) continue;

      const idxStr = r.toString();
      const y = paddingTop + plotHeight - ((r - viewIMin) / viewRows) * plotHeight - (plotHeight / viewRows) / 2;

      if (y >= paddingTop && y <= paddingTop + plotHeight) {
        ctx.fillText(idxStr, paddingLeft - 6, y);

        // Draw tiny ticks
        ctx.beginPath();
        ctx.moveTo(paddingLeft, y);
        ctx.lineTo(paddingLeft - 4, y);
        ctx.strokeStyle = tickColor;
        ctx.stroke();
      }
    }

    ctx.restore();

    // Labels
    ctx.fillStyle = labelColor;
    ctx.font = "bold 11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Residue j", paddingLeft + plotWidth / 2, paddingTop + plotHeight + 26);

    // Rotated label for Y-axis
    ctx.save();
    ctx.translate(14, paddingTop + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Residue i", 0, 0);
    ctx.restore();

    // ============== RIGHT-SIDE COLOR SCALE BAR (BIG LEGEND) ==============
    if (data.length > 0) {
      const barX = dimensions.width - 66;
      const barWidth = 14;
      
      // 1. Create Linear Gradient for the vertical color bar
      const grad = ctx.createLinearGradient(0, paddingTop, 0, paddingTop + plotHeight);
      
      if (mode === "correlation") {
        if (subType === "positive") {
          grad.addColorStop(0, "rgba(239, 68, 68, 1)");   // Top (+1.0)
          grad.addColorStop(1, "rgba(239, 68, 68, 0.05)"); // Bottom (0.00)
        } else if (subType === "negative") {
          grad.addColorStop(0, "rgba(59, 130, 246, 0.05)"); // Top (0.00)
          grad.addColorStop(1, "rgba(59, 130, 246, 1)");    // Bottom (-1.0)
        } else {
          // Combined
          grad.addColorStop(0, "rgba(239, 68, 68, 1)");   // Top (+1)
          grad.addColorStop(0.5, darkMode ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.04)"); // Mid (0)
          grad.addColorStop(1, "rgba(59, 130, 246, 1)");  // Bottom (-1)
        }
      } else {
        // Probability Mode
        if (subType === "positive") {
          grad.addColorStop(0, "rgba(244, 63, 94, 1)");   // Top (1.0)
          grad.addColorStop(1, "rgba(244, 63, 94, 0.05)"); // Bottom (0.0)
        } else if (subType === "negative") {
          grad.addColorStop(0, "rgba(168, 85, 247, 1)");   // Top (1.0)
          grad.addColorStop(1, "rgba(168, 85, 247, 0.05)"); // Bottom (0.0)
        } else {
          // Combined
          grad.addColorStop(0, "rgba(244, 63, 94, 1)");   // Top (Pos 1.0)
          grad.addColorStop(0.5, darkMode ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.04)"); // Mid (0.0)
          grad.addColorStop(1, "rgba(168, 85, 247, 1)");  // Bottom (Neg 1.0)
        }
      }
      
      // 2. Draw the vertical gradient bar
      ctx.fillStyle = grad;
      ctx.fillRect(barX, paddingTop, barWidth, plotHeight);
      
      // Draw outer border for the bar
      ctx.strokeStyle = borderStroke;
      ctx.lineWidth = 1;
      ctx.strokeRect(barX, paddingTop, barWidth, plotHeight);
      
      // 3. Determine labels & draw ticks/values next to the color bar
      ctx.fillStyle = labelColor;
      ctx.font = "bold 9px monospace";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      
      let topLabel = "";
      let midLabel = "";
      let botLabel = "";
      
      if (mode === "correlation") {
        if (subType === "positive") {
          topLabel = "+1.0";
          midLabel = "+0.5";
          botLabel = " 0.0";
        } else if (subType === "negative") {
          topLabel = " 0.0";
          midLabel = "-0.5";
          botLabel = "-1.0";
        } else {
          topLabel = "+1.0";
          midLabel = " 0.0";
          botLabel = "-1.0";
        }
      } else {
        // Probability Mode
        if (subType === "positive") {
          topLabel = "1.0 P";
          midLabel = "0.5 P";
          botLabel = "0.0 P";
        } else if (subType === "negative") {
          topLabel = "1.0 P";
          midLabel = "0.5 P";
          botLabel = "0.0 P";
        } else {
          topLabel = "Pos 1";
          midLabel = "0.0 P";
          botLabel = "Neg 1";
        }
      }
      
      // Draw ticks and text
      const textX = barX + barWidth + 6;
      
      // Top Ticks & Text
      ctx.beginPath();
      ctx.moveTo(barX, paddingTop);
      ctx.lineTo(barX + barWidth + 3, paddingTop);
      ctx.strokeStyle = borderStroke;
      ctx.stroke();
      ctx.fillText(topLabel, textX, paddingTop);
      
      // Middle Ticks & Text
      ctx.beginPath();
      ctx.moveTo(barX, paddingTop + plotHeight / 2);
      ctx.lineTo(barX + barWidth + 3, paddingTop + plotHeight / 2);
      ctx.strokeStyle = borderStroke;
      ctx.stroke();
      ctx.fillText(midLabel, textX, paddingTop + plotHeight / 2);
      
      // Bottom Ticks & Text
      ctx.beginPath();
      ctx.moveTo(barX, paddingTop + plotHeight);
      ctx.lineTo(barX + barWidth + 3, paddingTop + plotHeight);
      ctx.strokeStyle = borderStroke;
      ctx.stroke();
      ctx.fillText(botLabel, textX, paddingTop + plotHeight);
      
      // Vertical label/header above the bar
      ctx.fillStyle = textColor;
      ctx.font = "bold 8px sans-serif";
      ctx.textAlign = "center";
      const titleStr = mode === "correlation" ? "CORR" : "PROB";
      ctx.fillText(titleStr, barX + barWidth / 2, paddingTop - 8);
    }
  }, [dimensions, data, iMin, iMax, jMin, jMax, viewIMin, viewIMax, viewJMin, viewJMax, mode, subType, pMin, pMax, cMin, cMax, pPosCut, pNegCut, cPosCut, cNegCut, analysisMode, matrixLookup, darkMode]);

  // Handle native scroll wheel zoom
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleWheelNative = (e: WheelEvent) => {
      e.preventDefault(); // Prevent standard scroll of page

      const rect = canvas.getBoundingClientRect();
      const rawX = e.clientX - rect.left;
      const rawY = e.clientY - rect.top;

      const paddingLeft = 52;
      const paddingBottom = 44;
      const paddingTop = 26;
      const paddingRight = 85;

      const plotWidth = dimensions.width - paddingLeft - paddingRight;
      const plotHeight = dimensions.height - paddingTop - paddingBottom;

      // Only zoom if scrolling is inside plot coordinates
      if (
        rawX < paddingLeft ||
        rawX > dimensions.width - paddingRight ||
        rawY < paddingTop ||
        rawY > dimensions.height - paddingBottom
      ) {
        return;
      }

      const colRange = viewJMax - viewJMin;
      const rowRange = viewIMax - viewIMin;

      // Extract float coordinates of the cursor focus
      const fx = (rawX - paddingLeft) / plotWidth;
      const fy = (paddingTop + plotHeight - rawY) / plotHeight;

      const cursorJ = viewJMin + fx * colRange;
      const cursorI = viewIMin + fy * rowRange;

      // Deltas: zoom in on scroll up (-deltaY), zoom out on down (+deltaY)
      const zoomFactor = e.deltaY < 0 ? 0.82 : 1.18;

      let newColRange = colRange * zoomFactor;
      let newRowRange = rowRange * zoomFactor;

      const maxColRange = iMax - iMin + 1;
      const maxRowRange = jMax - jMin + 1;

      // Set clamp boundaries (e.g., minimum zoom level is 3 residues)
      if (newColRange < 3) newColRange = 3;
      if (newColRange > maxColRange) newColRange = maxColRange;

      if (newRowRange < 3) newRowRange = 3;
      if (newRowRange > maxRowRange) newRowRange = maxRowRange;

      // Derive zoomed boundary positions keeping cursor under same point
      let newJMin = cursorJ - fx * newColRange;
      let newJMax = newJMin + newColRange;

      let newIMin = cursorI - fy * newRowRange;
      let newIMax = newIMin + newRowRange;

      // Clamp coordinates inside database matrix margins
      if (newJMin < jMin) {
        newJMin = jMin;
        newJMax = jMin + newColRange;
      }
      if (newJMax > jMax + 1) {
        newJMax = jMax + 1;
        newJMin = newJMax - newColRange;
      }

      if (newIMin < iMin) {
        newIMin = iMin;
        newIMax = iMin + newRowRange;
      }
      if (newIMax > iMax + 1) {
        newIMax = iMax + 1;
        newIMin = newIMax - newRowRange;
      }

      setViewJMin(newJMin);
      setViewJMax(newJMax);
      setViewIMin(newIMin);
      setViewIMax(newIMax);
    };

    canvas.addEventListener("wheel", handleWheelNative, { passive: false });
    return () => {
      canvas.removeEventListener("wheel", handleWheelNative);
    };
  }, [viewIMin, viewIMax, viewJMin, viewJMax, iMin, iMax, jMin, jMax, dimensions]);

  // Window-level Panning state mouse handlers
  useEffect(() => {
    if (!isDragging) return;

    const handleWindowMouseMove = (e: MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas || !dragStartRef.current) return;

      const rect = canvas.getBoundingClientRect();
      const rawX = e.clientX - rect.left;
      const rawY = e.clientY - rect.top;

      const paddingLeft = 52;
      const paddingBottom = 44;
      const paddingTop = 26;
      const paddingRight = 85;

      const plotWidth = dimensions.width - paddingLeft - paddingRight;
      const plotHeight = dimensions.height - paddingTop - paddingBottom;

      const dx = rawX - dragStartRef.current.x;
      const dy = rawY - dragStartRef.current.y;

      const colRange = dragStartRef.current.viewJMax - dragStartRef.current.viewJMin;
      const rowRange = dragStartRef.current.viewIMax - dragStartRef.current.viewIMin;

      const dResidueJ = (dx / plotWidth) * colRange;
      const dResidueI = (dy / plotHeight) * rowRange;

      let newJMin = dragStartRef.current.viewJMin - dResidueJ;
      let newJMax = dragStartRef.current.viewJMax - dResidueJ;

      let newIMin = dragStartRef.current.viewIMin + dResidueI;
      let newIMax = dragStartRef.current.viewIMax + dResidueI;

      // Handle lock limits
      if (newJMin < jMin) {
        newJMin = jMin;
        newJMax = jMin + colRange;
      }
      if (newJMax > jMax + 1) {
        newJMax = jMax + 1;
        newJMin = newJMax - colRange;
      }

      if (newIMin < iMin) {
        newIMin = iMin;
        newIMax = iMin + rowRange;
      }
      if (newIMax > iMax + 1) {
        newIMax = iMax + 1;
        newIMin = newIMax - rowRange;
      }

      setViewJMin(newJMin);
      setViewJMax(newJMax);
      setViewIMin(newIMin);
      setViewIMax(newIMax);
    };

    const handleWindowMouseUp = (e: MouseEvent) => {
      setIsDragging(false);
      
      if (dragStartRef.current) {
        const canvas = canvasRef.current;
        if (canvas) {
          const rect = canvas.getBoundingClientRect();
          const rawX = e.clientX - rect.left;
          const rawY = e.clientY - rect.top;

          const dx = rawX - dragStartRef.current.startX;
          const dy = rawY - dragStartRef.current.startY;
          const distance = Math.sqrt(dx * dx + dy * dy);
          const timeElapsed = Date.now() - dragStartRef.current.time;

          // Single-click detected if mouse moved less than 5px and click was short (e.g. <300ms)
          if (distance < 5 && timeElapsed < 300) {
            const paddingLeft = 52;
            const paddingBottom = 44;
            const paddingTop = 26;
            const paddingRight = 85;

            const plotWidth = dimensions.width - paddingLeft - paddingRight;
            const plotHeight = dimensions.height - paddingTop - paddingBottom;

            if (
              rawX >= paddingLeft &&
              rawX <= dimensions.width - paddingRight &&
              rawY >= paddingTop &&
              rawY <= dimensions.height - paddingBottom
            ) {
              const colRange = viewJMax - viewJMin;
              const rowRange = viewIMax - viewIMin;

              const fx = (rawX - paddingLeft) / plotWidth;
              const fy = (paddingTop + plotHeight - rawY) / plotHeight;

              const jValF = viewJMin + fx * colRange;
              const iValF = viewIMin + fy * rowRange;

              const iVal = Math.floor(iValF);
              const jVal = Math.floor(jValF);

              if (iVal >= iMin && iVal <= iMax && jVal >= jMin && jVal <= jMax) {
                const point = matrixLookup.get(`${iVal},${jVal}`);
                if (point) {
                  onCellDoubleClick(iVal, jVal);
                }
              }
            }
          }
        }
      }

      dragStartRef.current = null;
    };

    window.addEventListener("mousemove", handleWindowMouseMove);
    window.addEventListener("mouseup", handleWindowMouseUp);

    return () => {
      window.removeEventListener("mousemove", handleWindowMouseMove);
      window.removeEventListener("mouseup", handleWindowMouseUp);
    };
  }, [isDragging, dimensions, jMin, jMax, iMin, iMax, viewIMin, viewIMax, viewJMin, viewJMax, onCellDoubleClick, matrixLookup]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // Left-click only for dragging
    if (e.button !== 0) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const rawX = e.clientX - rect.left;
    const rawY = e.clientY - rect.top;

    const paddingLeft = 52;
    const paddingBottom = 44;
    const paddingTop = 26;
    const paddingRight = 85;

    if (
      rawX < paddingLeft ||
      rawX > dimensions.width - paddingRight ||
      rawY < paddingTop ||
      rawY > dimensions.height - paddingBottom
    ) {
      return;
    }

    setIsDragging(true);
    dragStartRef.current = {
      x: rawX,
      y: rawY,
      viewIMin,
      viewIMax,
      viewJMin,
      viewJMax,
      startX: rawX,
      startY: rawY,
      time: Date.now()
    };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging) return; // Managed by window mousemove to prevent mouse out of bounds lock up

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const rawX = e.clientX - rect.left;
    const rawY = e.clientY - rect.top;

    const paddingLeft = 52;
    const paddingBottom = 44;
    const paddingTop = 26;
    const paddingRight = 85;

    const plotWidth = dimensions.width - paddingLeft - paddingRight;
    const plotHeight = dimensions.height - paddingTop - paddingBottom;

    if (
      rawX < paddingLeft ||
      rawX > dimensions.width - paddingRight ||
      rawY < paddingTop ||
      rawY > dimensions.height - paddingBottom
    ) {
      setHoveredCell(null);
      return;
    }

    const colRange = viewJMax - viewJMin;
    const rowRange = viewIMax - viewIMin;

    const fx = (rawX - paddingLeft) / plotWidth;
    const fy = (paddingTop + plotHeight - rawY) / plotHeight;

    const jValF = viewJMin + fx * colRange;
    const iValF = viewIMin + fy * rowRange;

    const iVal = Math.floor(iValF);
    const jVal = Math.floor(jValF);

    if (iVal >= iMin && iVal <= iMax && jVal >= jMin && jVal <= jMax) {
      const point = matrixLookup.get(`${iVal},${jVal}`);
      if (point) {
        setHoveredCell({
          i: iVal,
          j: jVal,
          point,
          x: rawX + 16,
          y: rawY + 16,
        });
      } else {
        setHoveredCell(null);
      }
    } else {
      setHoveredCell(null);
    }
  };

  const handleMouseLeave = () => {
    if (!isDragging) {
      setHoveredCell(null);
    }
  };

  const handleDoubleClick = () => {
    if (hoveredCell) {
      onCellDoubleClick(hoveredCell.i, hoveredCell.j);
    }
  };

  // Static Button Layout Zoom Actions
  const zoomInStep = () => {
    const colRange = viewJMax - viewJMin;
    const rowRange = viewIMax - viewIMin;
    const curCenterX = (viewJMin + viewJMax) / 2;
    const curCenterY = (viewIMin + viewIMax) / 2;

    let newColRange = colRange * 0.75;
    let newRowRange = rowRange * 0.75;

    if (newColRange < 3) newColRange = 3;
    if (newRowRange < 3) newRowRange = 3;

    let newJMin = curCenterX - newColRange / 2;
    let newJMax = curCenterX + newColRange / 2;
    let newIMin = curCenterY - newRowRange / 2;
    let newIMax = curCenterY + newRowRange / 2;

    if (newJMin < jMin) {
      newJMin = jMin;
      newJMax = jMin + newColRange;
    }
    if (newJMax > jMax + 1) {
      newJMax = jMax + 1;
      newJMin = newJMax - newColRange;
    }
    if (newIMin < iMin) {
      newIMin = iMin;
      newIMax = iMin + newRowRange;
    }
    if (newIMax > iMax + 1) {
      newIMax = iMax + 1;
      newIMin = newIMax - newRowRange;
    }

    setViewJMin(newJMin);
    setViewJMax(newJMax);
    setViewIMin(newIMin);
    setViewIMax(newIMax);
  };

  const zoomOutStep = () => {
    const colRange = viewJMax - viewJMin;
    const rowRange = viewIMax - viewIMin;
    const curCenterX = (viewJMin + viewJMax) / 2;
    const curCenterY = (viewIMin + viewIMax) / 2;

    let newColRange = colRange * 1.33;
    let newRowRange = rowRange * 1.33;

    const maxColRange = iMax - iMin + 1;
    const maxRowRange = jMax - jMin + 1;

    if (newColRange > maxColRange) newColRange = maxColRange;
    if (newRowRange > maxRowRange) newRowRange = maxRowRange;

    let newJMin = curCenterX - newColRange / 2;
    let newJMax = curCenterX + newColRange / 2;
    let newIMin = curCenterY - newRowRange / 2;
    let newIMax = curCenterY + newRowRange / 2;

    if (newJMin < jMin) {
      newJMin = jMin;
      newJMax = jMin + newColRange;
    }
    if (newJMax > jMax + 1) {
      newJMax = jMax + 1;
      newJMin = newJMax - newColRange;
    }
    if (newIMin < iMin) {
      newIMin = iMin;
      newIMax = iMin + newRowRange;
    }
    if (newIMax > iMax + 1) {
      newIMax = iMax + 1;
      newIMin = newIMax - newRowRange;
    }

    setViewJMin(newJMin);
    setViewJMax(newJMax);
    setViewIMin(newIMin);
    setViewIMax(newIMax);
  };

  const zoomMaximumAmount = () => {
    // Zoom in as much as possible - focus to 3x3 layout of residues
    const curCenterX = (viewJMin + viewJMax) / 2;
    const curCenterY = (viewIMin + viewIMax) / 2;

    const newColRange = 3;
    const newRowRange = 3;

    let newJMin = curCenterX - 1.5;
    let newJMax = curCenterX + 1.5;
    let newIMin = curCenterY - 1.5;
    let newIMax = curCenterY + 1.5;

    if (newJMin < jMin) {
      newJMin = jMin;
      newJMax = jMin + newColRange;
    }
    if (newJMax > jMax + 1) {
      newJMax = jMax + 1;
      newJMin = newJMax - newColRange;
    }
    if (newIMin < iMin) {
      newIMin = iMin;
      newIMax = iMin + newRowRange;
    }
    if (newIMax > iMax + 1) {
      newIMax = iMax + 1;
      newIMin = newIMax - newRowRange;
    }

    setViewJMin(newJMin);
    setViewJMax(newJMax);
    setViewIMin(newIMin);
    setViewIMax(newIMax);
  };

  const resetZoom = () => {
    setViewIMin(iMin);
    setViewIMax(iMax + 1);
    setViewJMin(jMin);
    setViewJMax(jMax + 1);
  };

  const savePlotToPNG = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const link = document.createElement("a");
    const label = `${mode}_${subType}`.toLowerCase().replace(/[^a-z0-9]/g, "_");
    link.download = `biomolecule_heatmap_${label}_computed.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  };

  return (
    <div ref={containerRef} className="w-full h-full min-h-[220px] relative">
      
      {/* Absolute zoom, pan and save overlay toolbar */}
      {data.length > 0 && (
        <div className={`absolute top-2 right-2 flex items-center gap-0.5 pointer-events-auto rounded-lg p-0.5 shadow-md z-10 backdrop-blur-md transition-all border ${
          darkMode 
            ? "bg-slate-900/90 border-slate-800 text-slate-350" 
            : "bg-white/90 border-slate-200 text-slate-700"
        }`}>
          <button
            onClick={zoomInStep}
            title="Zoom In (Scroll Mouse Wheel Up)"
            className={`p-1.5 rounded cursor-pointer transition-colors ${
              darkMode ? "hover:bg-slate-800 hover:text-white" : "hover:bg-slate-100 hover:text-indigo-650"
            }`}
          >
            <ZoomIn size={13} />
          </button>
          <button
            onClick={zoomOutStep}
            title="Zoom Out (Scroll Mouse Wheel Down)"
            className={`p-1.5 rounded cursor-pointer transition-colors ${
              darkMode ? "hover:bg-slate-800 hover:text-white" : "hover:bg-slate-100 hover:text-indigo-650"
            }`}
          >
            <ZoomOut size={13} />
          </button>
          <button
            onClick={zoomMaximumAmount}
            title="Zoom Maximum Amount (Inspect detailed couples)"
            className={`p-1.5 rounded cursor-pointer transition-colors ${
              darkMode ? "hover:bg-slate-800 hover:text-indigo-400" : "hover:bg-slate-100 hover:text-indigo-600"
            }`}
          >
            <Maximize2 size={13} />
          </button>
          <button
            onClick={resetZoom}
            title="Reset Zoom / Fit to Screen"
            className={`p-1.5 rounded cursor-pointer transition-colors ${
              darkMode ? "hover:bg-slate-800 hover:text-white" : "hover:bg-slate-100 hover:text-indigo-650"
            }`}
          >
            <RotateCcw size={13} />
          </button>
          <div className={`w-px h-4 mx-0.5 ${darkMode ? "bg-slate-800" : "bg-slate-200"}`} />
          <button
            onClick={savePlotToPNG}
            title="Save Plot Image (PNG)"
            className={`p-1.5 rounded cursor-pointer transition-colors ${
              darkMode ? "hover:bg-slate-800 hover:text-emerald-400" : "hover:bg-slate-100 hover:text-emerald-600"
            }`}
          >
            <Download size={13} />
          </button>
        </div>
      )}

      {/* Tiny interaction hint */}
      {data.length > 0 && (
        <div className={`absolute bottom-3 right-24 pointer-events-none select-none text-[8px] font-mono opacity-25 ${
          darkMode ? "text-slate-500" : "text-slate-400"
        }`}>
          [Drag mouse to Pan | Wheel to Zoom]
        </div>
      )}

      <canvas
        ref={canvasRef}
        style={{ width: `${dimensions.width}px`, height: `${dimensions.height}px` }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onDoubleClick={handleDoubleClick}
        className={`block rounded transition-colors ${
          isDragging ? "cursor-grabbing" : "cursor-grab"
        }`}
      />

      {/* Tooltip */}
      {hoveredCell && (
        <div
          style={{
            position: "absolute",
            left: `${hoveredCell.x}px`,
            top: `${hoveredCell.y}px`,
            pointerEvents: "none",
            zIndex: 9999,
          }}
          className={`${
            darkMode 
              ? "bg-slate-950/95 border-slate-750 text-slate-350"
              : "bg-white/95 border-slate-200 text-slate-700 shadow-lg"
          } border backdrop-blur-md p-2.5 rounded-lg text-xs font-mono flex flex-col gap-1 max-w-[210px]`}
        >
          <div className={`font-sans font-extrabold border-b pb-1 flex justify-between gap-4 ${
            darkMode ? "text-slate-100 border-slate-800" : "text-slate-800 border-slate-100"
          }`}>
            <span>Pair ({hoveredCell.i}, {hoveredCell.j})</span>
            <span className="text-[10px] text-indigo-500 font-normal">Dbl-click to plot</span>
          </div>
          <div className="flex justify-between gap-4 mt-1">
            <span className={darkMode ? "text-slate-400" : "text-slate-500"}>Mean Corr:</span>
            <span className={hoveredCell.point.Mean_Correlation > 0 ? "text-emerald-500 font-extrabold" : "text-blue-500 font-extrabold"}>
              {hoveredCell.point.Mean_Correlation.toFixed(4)}
            </span>
          </div>
          <div className="flex justify-between gap-4">
            <span className={darkMode ? "text-slate-400" : "text-slate-500"}>Std Dev:</span>
            <span className="font-semibold">{hoveredCell.point.Std_Correlation.toFixed(4)}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className={darkMode ? "text-slate-400" : "text-slate-500"}>{"P( > 0 ):"}</span>
            <span className="text-emerald-500 font-bold">{hoveredCell.point.Positive_Probability.toFixed(3)}</span>
          </div>
          <div className="flex justify-between gap-4">
            <span className={darkMode ? "text-slate-400" : "text-slate-500"}>{"P( < 0 ):"}</span>
            <span className="text-blue-500 font-bold">{hoveredCell.point.Negative_Probability.toFixed(3)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
