import React, { useState, useRef, useEffect } from "react";
import { X, Minimize2, Maximize2, RotateCcw } from "lucide-react";

interface WindowFrameProps {
  id: string;
  title: string;
  key?: string;
  defaultPosition?: { x: number; y: number };
  defaultSize?: { width: number; height: number };
  minSize?: { width: number; height: number };
  onClose: () => void;
  onFocus: () => void;
  activeZIndex: number;
  children: React.ReactNode;
}

export default function WindowFrame({
  id,
  title,
  defaultPosition = { x: 100, y: 100 },
  defaultSize = { width: 600, height: 450 },
  minSize = { width: 350, height: 280 },
  onClose,
  onFocus,
  activeZIndex,
  children,
}: WindowFrameProps) {
  const [position, setPosition] = useState(defaultPosition);
  const [size, setSize] = useState(defaultSize);
  const [isMaximized, setIsMaximized] = useState(false);
  const [preMaxState, setPreMaxState] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  const windowRef = useRef<HTMLDivElement>(null);
  const dragStartPos = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const dragStartWindowPos = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const isDragging = useRef(false);

  const resizeStartPos = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const resizeStartSize = useRef<{ width: number; height: number }>({ width: 0, height: 0 });
  const isResizing = useRef(false);

  // Focus on instantiation
  useEffect(() => {
    onFocus();
  }, []);

  // Drag handlers
  const handleHeaderMouseDown = (e: React.MouseEvent) => {
    if (isMaximized) return;
    onFocus();
    
    // Prevent dragging if clicking buttons
    const target = e.target as HTMLElement;
    if (target.closest("button")) return;

    isDragging.current = true;
    dragStartPos.current = { x: e.clientX, y: e.clientY };
    dragStartWindowPos.current = { ...position };

    document.addEventListener("mousemove", handleHeaderMouseMove);
    document.addEventListener("mouseup", handleHeaderMouseUp);
  };

  const handleHeaderMouseMove = (e: MouseEvent) => {
    if (!isDragging.current) return;
    const dx = e.clientX - dragStartPos.current.x;
    const dy = e.clientY - dragStartPos.current.y;

    const newX = Math.max(0, Math.min(window.innerWidth - size.width, dragStartWindowPos.current.x + dx));
    const newY = Math.max(0, Math.min(window.innerHeight - 80, dragStartWindowPos.current.y + dy));
    
    setPosition({ x: newX, y: newY });
  };

  const handleHeaderMouseUp = () => {
    isDragging.current = false;
    document.removeEventListener("mousemove", handleHeaderMouseMove);
    document.removeEventListener("mouseup", handleHeaderMouseUp);
  };

  // Resize handlers
  const handleResizeMouseDown = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    onFocus();

    isResizing.current = true;
    resizeStartPos.current = { x: e.clientX, y: e.clientY };
    resizeStartSize.current = { ...size };

    document.addEventListener("mousemove", handleResizeMouseMove);
    document.addEventListener("mouseup", handleResizeMouseUp);
  };

  const handleResizeMouseMove = (e: MouseEvent) => {
    if (!isResizing.current) return;
    const dx = e.clientX - resizeStartPos.current.x;
    const dy = e.clientY - resizeStartPos.current.y;

    const newW = Math.max(minSize.width, resizeStartSize.current.width + dx);
    const newH = Math.max(minSize.height, resizeStartSize.current.height + dy);

    setSize({ width: newW, height: newH });
  };

  const handleResizeMouseUp = () => {
    isResizing.current = false;
    document.removeEventListener("mousemove", handleResizeMouseMove);
    document.removeEventListener("mouseup", handleResizeMouseUp);
  };

  const toggleMaximize = () => {
    onFocus();
    if (isMaximized) {
      if (preMaxState) {
        setPosition({ x: preMaxState.x, y: preMaxState.y });
        setSize({ width: preMaxState.w, height: preMaxState.h });
      }
      setIsMaximized(false);
    } else {
      setPreMaxState({ x: position.x, y: position.y, w: size.width, h: size.height });
      setPosition({ x: 0, y: 0 });
      setSize({ width: window.innerWidth, height: window.innerHeight - 44 });
      setIsMaximized(true);
    }
  };

  // Window geometry styling
  const style: React.CSSProperties = isMaximized
    ? {
        position: "fixed",
        top: 0,
        left: 0,
        width: "100vw",
        height: "calc(100vh - 44px)",
        zIndex: activeZIndex,
      }
    : {
        position: "absolute",
        left: `${position.x}px`,
        top: `${position.y}px`,
        width: `${size.width}px`,
        height: `${size.height}px`,
        zIndex: activeZIndex,
      };

  return (
    <div
      ref={windowRef}
      style={style}
      className="flex flex-col bg-slate-900 border border-slate-800 rounded-xl shadow-2xl overflow-hidden select-none"
      onClick={onFocus}
      id={`window-${id}`}
    >
      {/* Title bar */}
      <div
        className="flex items-center justify-between bg-slate-950 px-4 py-2.5 cursor-move border-b border-slate-850"
        onMouseDown={handleHeaderMouseDown}
        onDoubleClick={toggleMaximize}
      >
        <span className="text-[11px] font-bold text-slate-200 tracking-wide font-sans select-none truncate pr-4">
          {title}
        </span>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <button
            onClick={toggleMaximize}
            className="p-1 hover:bg-slate-800 rounded-md text-slate-400 hover:text-slate-200 transition-colors cursor-pointer"
            title={isMaximized ? "Restore Window" : "Maximize Window"}
          >
            {isMaximized ? <RotateCcw size={12} /> : <Maximize2 size={12} />}
          </button>
          <button
            onClick={onClose}
            className="p-1 hover:bg-red-500/20 hover:text-red-400 text-slate-450 rounded-md transition-colors cursor-pointer"
            title="Close Window"
          >
            <X size={12} />
          </button>
        </div>
      </div>

      {/* Content wrapper */}
      <div className="flex-1 min-h-0 bg-slate-950 text-slate-150 overflow-auto font-sans relative">
        {children}
      </div>

      {/* Resize handle */}
      {!isMaximized && (
        <div
          className="absolute bottom-0 right-0 w-4 h-4 cursor-se-resize flex items-end justify-end p-0.5 pointer-events-auto"
          onMouseDown={handleResizeMouseDown}
        >
          <svg width="8" height="8" viewBox="0 0 8 8" className="text-slate-500 fill-current">
            <path d="M6 0 L8 0 L8 8 L0 8 L0 6 L4 6 L4 4 L6 4 Z" />
          </svg>
        </div>
      )}
    </div>
  );
}
