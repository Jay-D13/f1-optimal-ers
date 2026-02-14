import React, { useEffect, useMemo, useState, useRef, useCallback } from 'react';
import { Stage, Layer, Line } from 'react-konva';
import { useTheme } from '../context/ThemeContext';

import type { TrackPoint, RacelinePoint } from '../api';

interface TrackMapProps {
    trackData: TrackPoint[];
    raceline: RacelinePoint[] | null;
    onRacelineChange?: (newRaceline: RacelinePoint[]) => void;
    editable?: boolean;
}

const TrackMap: React.FC<TrackMapProps> = ({ trackData, raceline, onRacelineChange, editable = false }) => {
    const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
    const [scale, setScale] = useState(1);
    const [offset, setOffset] = useState({ x: 0, y: 0 });
    const [localRaceline, setLocalRaceline] = useState<RacelinePoint[]>([]);
    const [draggingIdx, setDraggingIdx] = useState<number | null>(null);
    const stageRef = useRef<any>(null);

    useEffect(() => {
        if (raceline) {
            setLocalRaceline(raceline);
        }
    }, [raceline]);

    // Calculate bounding box and scale — fit to container at 1x stage scale
    useEffect(() => {
        if (trackData.length === 0) return;

        // Also include boundary widths in bounding box calc
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const p of trackData) {
            // Rough expansion by max(w_left, w_right) in all directions
            const w = Math.max(p.w_left, p.w_right);
            if (p.x - w < minX) minX = p.x - w;
            if (p.x + w > maxX) maxX = p.x + w;
            if (p.y - w < minY) minY = p.y - w;
            if (p.y + w > maxY) maxY = p.y + w;
        }

        const trackWidth = maxX - minX;
        const trackHeight = maxY - minY;

        // Fit the track into the container with some padding
        const padding = 10;
        const availWidth = dimensions.width - padding * 2;
        const availHeight = dimensions.height - padding * 2;

        const fitScale = Math.min(availWidth / trackWidth, availHeight / trackHeight);

        setScale(fitScale);
        setOffset({
            x: padding - minX * fitScale + (availWidth - trackWidth * fitScale) / 2,
            y: padding - minY * fitScale + (availHeight - trackHeight * fitScale) / 2
        });

        const handleResize = () => {
            const container = document.getElementById('track-map-container');
            if (container) {
                setDimensions({ width: container.clientWidth, height: container.clientHeight });
            }
        };

        window.addEventListener('resize', handleResize);
        handleResize();

        return () => window.removeEventListener('resize', handleResize);
    }, [trackData, dimensions.width, dimensions.height]);

    // Transform: world coords -> canvas pixel coords
    const toCanvas = useCallback((x: number, y: number) => ({
        x: x * scale + offset.x,
        y: dimensions.height - (y * scale + offset.y) // Flip Y
    }), [scale, offset, dimensions.height]);

    // Inverse: canvas pixel coords -> world coords
    const fromCanvas = useCallback((cx: number, cy: number) => ({
        x: (cx - offset.x) / scale,
        y: (dimensions.height - cy - offset.y) / scale
    }), [scale, offset, dimensions.height]);

    // Track boundary lines (computed in canvas space)
    const trackDataMemo = useMemo(() => {
        if (trackData.length < 2) return { leftLine: [], rightLine: [], centerLine: [] };

        const outerPoints: number[] = [];
        const innerPoints: number[] = [];
        const centerPoints: number[] = [];

        for (let i = 0; i < trackData.length; i++) {
            const p = trackData[i];
            const next = trackData[(i + 1) % trackData.length];
            const prev = trackData[(i - 1 + trackData.length) % trackData.length];

            const dx = next.x - prev.x;
            const dy = next.y - prev.y;
            const len = Math.sqrt(dx * dx + dy * dy);

            let nx = 0, ny = 0;
            if (len > 0) {
                nx = -dy / len;
                ny = dx / len;
            }

            const l = toCanvas(p.x + nx * p.w_left, p.y + ny * p.w_left);
            outerPoints.push(l.x, l.y);

            const r = toCanvas(p.x - nx * p.w_right, p.y - ny * p.w_right);
            innerPoints.push(r.x, r.y);

            const c = toCanvas(p.x, p.y);
            centerPoints.push(c.x, c.y);
        }

        return { leftLine: outerPoints, rightLine: innerPoints, centerLine: centerPoints };
    }, [trackData, toCanvas]);

    const { leftLine, rightLine, centerLine } = trackDataMemo;

    // Raceline in canvas space
    const racelinePoints = useMemo(() => {
        return localRaceline.flatMap(p => {
            const c = toCanvas(p.x, p.y);
            return [c.x, c.y];
        });
    }, [localRaceline, toCanvas]);

    // Constrain a world-space point to within the track boundaries
    const constrainPoint = useCallback((x: number, y: number, hintIdx: number) => {
        const trackLen = trackData.length;
        if (trackLen < 2) return { x, y };

        let best = { dSq: Infinity, closeX: 0, closeY: 0, w_left: 0, w_right: 0, nx: 0, ny: 0 };

        // Search a window around a hinted track index
        const racelineLen = localRaceline.length;
        const centerIdx = racelineLen > 0
            ? Math.floor((hintIdx / racelineLen) * trackLen)
            : 0;
        const searchRadius = Math.min(80, Math.floor(trackLen / 2));

        for (let i = -searchRadius; i <= searchRadius; i++) {
            const idx = (centerIdx + i + trackLen) % trackLen;
            const p1 = trackData[idx];
            const p2 = trackData[(idx + 1) % trackLen];

            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;
            const lenSq = dx * dx + dy * dy;
            if (lenSq < 1e-12) continue;

            let t = ((x - p1.x) * dx + (y - p1.y) * dy) / lenSq;
            t = Math.max(0, Math.min(1, t));

            const closeX = p1.x + t * dx;
            const closeY = p1.y + t * dy;
            const dSq = (x - closeX) ** 2 + (y - closeY) ** 2;

            if (dSq < best.dSq) {
                const len = Math.sqrt(lenSq);
                best = {
                    dSq,
                    closeX,
                    closeY,
                    w_left: p1.w_left + t * (p2.w_left - p1.w_left),
                    w_right: p1.w_right + t * (p2.w_right - p1.w_right),
                    nx: -dy / len,
                    ny: dx / len,
                };
            }
        }

        if (best.dSq === Infinity) return { x, y };

        // Project onto normal and clamp within track width
        let dist = (x - best.closeX) * best.nx + (y - best.closeY) * best.ny;
        dist = Math.max(-best.w_right, Math.min(best.w_left, dist));

        return {
            x: best.closeX + best.nx * dist,
            y: best.closeY + best.ny * dist,
        };
    }, [trackData, localRaceline]);

    // ---- Direct line interaction (no Circle handles) ----

    // Given a canvas-space position, find the nearest raceline point index
    const findNearestRacelineIdx = useCallback((canvasX: number, canvasY: number): number => {
        let minDist = Infinity;
        let bestIdx = 0;
        for (let i = 0; i < localRaceline.length; i++) {
            const c = toCanvas(localRaceline[i].x, localRaceline[i].y);
            const d = (canvasX - c.x) ** 2 + (canvasY - c.y) ** 2;
            if (d < minDist) {
                minDist = d;
                bestIdx = i;
            }
        }
        return bestIdx;
    }, [localRaceline, toCanvas]);

    // Convert stage pointer to canvas coords (accounts for stage pan/zoom)
    const pointerToCanvas = useCallback((stage: any): { x: number; y: number } | null => {
        const pointer = stage.getPointerPosition();
        if (!pointer) return null;
        const transform = stage.getAbsoluteTransform().copy();
        transform.invert();
        return transform.point(pointer);
    }, []);

    // Mouse down on raceline -> start drag
    const handleRacelineMouseDown = useCallback((e: any) => {
        if (!editable) return;
        e.cancelBubble = true; // prevent stage drag

        const stage = e.target.getStage();
        const canvasPos = pointerToCanvas(stage);
        if (!canvasPos) return;

        const idx = findNearestRacelineIdx(canvasPos.x, canvasPos.y);
        setDraggingIdx(idx);
        stage.container().style.cursor = 'grabbing';
    }, [editable, pointerToCanvas, findNearestRacelineIdx]);

    // Mouse move on stage while dragging
    const handleStageMouseMove = useCallback((e: any) => {
        if (draggingIdx === null) return;

        const stage = e.target.getStage();
        const canvasPos = pointerToCanvas(stage);
        if (!canvasPos) return;

        const worldPos = fromCanvas(canvasPos.x, canvasPos.y);
        const constrained = constrainPoint(worldPos.x, worldPos.y, draggingIdx);

        // Apply Gaussian-weighted neighbor blending for smooth deformation
        const newRaceline = [...localRaceline];
        const blendRadius = 3; // affect +/- 3 neighbors
        const sigma = 1.5;

        const oldX = newRaceline[draggingIdx].x;
        const oldY = newRaceline[draggingIdx].y;
        const deltaX = constrained.x - oldX;
        const deltaY = constrained.y - oldY;

        for (let di = -blendRadius; di <= blendRadius; di++) {
            const idx = (draggingIdx + di + newRaceline.length) % newRaceline.length;
            const weight = Math.exp(-(di * di) / (2 * sigma * sigma));

            const newX = newRaceline[idx].x + deltaX * weight;
            const newY = newRaceline[idx].y + deltaY * weight;

            // Re-constrain each neighbor to stay within track
            const neighborConstrained = constrainPoint(newX, newY, idx);
            newRaceline[idx] = { ...newRaceline[idx], x: neighborConstrained.x, y: neighborConstrained.y };
        }

        setLocalRaceline(newRaceline);
    }, [draggingIdx, localRaceline, fromCanvas, constrainPoint, pointerToCanvas]);

    // Mouse up -> end drag
    const handleStageMouseUp = useCallback((e: any) => {
        if (draggingIdx !== null) {
            setDraggingIdx(null);
            if (onRacelineChange) {
                onRacelineChange(localRaceline);
            }
            const stage = e.target.getStage();
            if (stage) stage.container().style.cursor = 'default';
        }
    }, [draggingIdx, localRaceline, onRacelineChange]);

    // Zoom/Pan
    const handleWheel = useCallback((e: any) => {
        e.evt.preventDefault();
        const stage = e.target.getStage();

        if (e.evt.ctrlKey || e.evt.metaKey) {
            const oldScale = stage.scaleX();
            const pointer = stage.getPointerPosition();
            const mousePointTo = {
                x: (pointer.x - stage.x()) / oldScale,
                y: (pointer.y - stage.y()) / oldScale,
            };
            const newScale = e.evt.deltaY > 0 ? oldScale * 0.95 : oldScale * 1.05;
            stage.scale({ x: newScale, y: newScale });
            stage.position({
                x: pointer.x - mousePointTo.x * newScale,
                y: pointer.y - mousePointTo.y * newScale,
            });
        } else {
            const oldPos = stage.position();
            stage.position({
                x: oldPos.x - e.evt.deltaX,
                y: oldPos.y - e.evt.deltaY,
            });
        }
    }, []);

    // Cursor feedback when hovering over the raceline
    const handleRacelineMouseEnter = useCallback((e: any) => {
        if (!editable) return;
        const stage = e.target.getStage();
        if (stage) stage.container().style.cursor = 'grab';
    }, [editable]);

    const handleRacelineMouseLeave = useCallback((e: any) => {
        if (!editable || draggingIdx !== null) return;
        const stage = e.target.getStage();
        if (stage) stage.container().style.cursor = 'default';
    }, [editable, draggingIdx]);

    const { theme } = useTheme();

    // ... (rest of the component logic)

    return (
        <div id="track-map-container" className="w-full h-full bg-retro-bg border border-retro-border rounded overflow-hidden relative group">
            {/* Scale Info */}
            <div className="absolute bottom-2 right-2 z-10 font-mono text-xs text-retro-text/50 bg-white/80 dark:bg-black/80 dark:text-white/50 p-2 rounded pointer-events-none text-right border border-retro-border/10 shadow-sm backdrop-blur">
                <div>SCALE: {scale.toFixed(2)}x</div>
                <div className="text-[10px] opacity-70">SCROLL TO PAN • CTRL+SCROLL TO ZOOM</div>
            </div>

            <Stage
                ref={stageRef}
                width={dimensions.width}
                height={dimensions.height}
                onWheel={handleWheel}
                draggable={draggingIdx === null} // Disable stage drag while editing raceline
                onMouseMove={handleStageMouseMove}
                onMouseUp={handleStageMouseUp}
                onTouchMove={handleStageMouseMove}
                onTouchEnd={handleStageMouseUp}
            >
                <Layer>
                    {/* Centerline */}
                    <Line
                        points={centerLine}
                        stroke={theme === 'dark' ? '#444' : '#ccc'}
                        strokeWidth={1}
                        dash={[6, 6]}
                        opacity={0.4}
                        strokeScaleEnabled={false}
                        tension={0.3}
                        closed
                    />

                    {/* Track Boundaries */}
                    <Line
                        points={leftLine}
                        stroke={theme === 'dark' ? '#888' : '#444'}
                        strokeWidth={1.5}
                        strokeScaleEnabled={false}
                        tension={0.3}
                        closed
                    />
                    <Line
                        points={rightLine}
                        stroke={theme === 'dark' ? '#888' : '#444'}
                        strokeWidth={1.5}
                        strokeScaleEnabled={false}
                        tension={0.3}
                        closed
                    />

                    {/* Raceline — directly interactive */}
                    {racelinePoints.length > 0 && (
                        <Line
                            points={racelinePoints}
                            stroke="#FF1801"
                            strokeWidth={3}
                            strokeScaleEnabled={false}
                            closed={true}
                            tension={0.3}
                            hitStrokeWidth={20} // Wide invisible hit zone makes it very easy to click
                            onMouseDown={handleRacelineMouseDown}
                            onTouchStart={handleRacelineMouseDown}
                            onMouseEnter={handleRacelineMouseEnter}
                            onMouseLeave={handleRacelineMouseLeave}
                        />
                    )}
                </Layer>
            </Stage>
        </div>
    );
};

export default TrackMap;
