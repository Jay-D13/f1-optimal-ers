import { useMemo } from 'react';
import { useWidth } from '../hooks/useWidth';

export interface TraceLine {
    s: number[];
    y: number[];
    color: string;
    width?: number;
    /** 'line', or 'signed' for an area filled from zero: `color` above, `negColor` below */
    kind?: 'line' | 'signed';
    negColor?: string;
}

interface TraceProps {
    title: string;
    unit: string;
    lines: TraceLine[];
    length: number;           // x domain [0, length] (m)
    yMin: number;
    yMax: number;
    ticks: number[];
    tickUnit?: string;
    height: number;
    cursor: number | null;    // lap distance (m)
    onCursor: (s: number | null) => void;
}

const LEFT = 44;
const RIGHT = 6;
const PAD = 6;

function toPath(s: number[], y: number[], sx: (v: number) => number, sy: (v: number) => number, step: number) {
    const n = Math.min(s.length, y.length);
    let d = '';
    for (let i = 0; i < n; i += step) {
        d += `${i === 0 ? 'M' : 'L'}${sx(s[i]).toFixed(1)},${sy(y[i]).toFixed(1)}`;
    }
    return d;
}

function toArea(s: number[], y: number[], sx: (v: number) => number, sy: (v: number) => number, clip: (v: number) => number) {
    const n = Math.min(s.length, y.length);
    if (n === 0) return '';
    let d = `M${sx(s[0]).toFixed(1)},${sy(0).toFixed(1)}`;
    for (let i = 0; i < n; i++) d += `L${sx(s[i]).toFixed(1)},${sy(clip(y[i])).toFixed(1)}`;
    return d + `L${sx(s[n - 1]).toFixed(1)},${sy(0).toFixed(1)}Z`;
}

/** One chart against lap distance, sharing its cursor with the others. */
export function Trace({ title, unit, lines, length, yMin, yMax, ticks, tickUnit = '', height, cursor, onCursor }: TraceProps) {
    const [ref, width] = useWidth<HTMLDivElement>();
    const x1 = Math.max(width - RIGHT, LEFT + 1);
    const sx = (v: number) => LEFT + (v / length) * (x1 - LEFT);
    const sy = (v: number) => height - PAD - ((v - yMin) / (yMax - yMin)) * (height - 2 * PAD);

    const paths = useMemo(() => {
        if (width === 0) return [];
        return lines.map((line) => {
            const step = line.s.length > 2 * width ? 2 : 1;
            if (line.kind === 'signed') {
                return {
                    key: `${line.color}-signed`,
                    pos: toArea(line.s, line.y, sx, sy, (v) => Math.max(v, 0)),
                    neg: toArea(line.s, line.y, sx, sy, (v) => Math.min(v, 0)),
                    line,
                };
            }
            return { key: line.color, d: toPath(line.s, line.y, sx, sy, step), line };
        });
        // sx/sy depend only on these
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [lines, width, height, length, yMin, yMax]);

    const handleMove = (e: React.MouseEvent<SVGSVGElement>) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const px = e.clientX - rect.left;
        if (px < LEFT || px > x1) return onCursor(null);
        onCursor(((px - LEFT) / (x1 - LEFT)) * length);
    };

    return (
        <div className="flex flex-col gap-1.5">
            <div className="flex justify-between">
                <span className="cap" style={{ color: 'var(--ink)' }}>{title}</span>
                <span className="cap">{unit}</span>
            </div>
            <div ref={ref} className="w-full">
                {width > 0 && (
                    <svg width={width} height={height} role="img" aria-label={title} onMouseMove={handleMove} onMouseLeave={() => onCursor(null)} style={{ display: 'block', cursor: 'crosshair' }}>
                        {ticks.map((t) => (
                            <g key={t}>
                                <line x1={LEFT} x2={x1} y1={sy(t)} y2={sy(t)} stroke="var(--rule)" />
                                <text x={LEFT - 8} y={sy(t) + 3.5} textAnchor="end" className="tick">{t}{tickUnit}</text>
                            </g>
                        ))}
                        {paths.map((p) => 'd' in p
                            ? <path key={p.key} d={p.d} fill="none" stroke={p.line.color} strokeWidth={p.line.width ?? 1.8} strokeLinejoin="round" />
                            : (
                                <g key={p.key}>
                                    <path d={p.pos} fill={p.line.color} />
                                    <path d={p.neg} fill={p.line.negColor ?? p.line.color} />
                                </g>
                            ))}
                        {cursor != null && (
                            <line x1={sx(cursor)} x2={sx(cursor)} y1={0} y2={height} stroke="var(--ink)" strokeOpacity={0.6} />
                        )}
                    </svg>
                )}
            </div>
        </div>
    );
}

/** Lap distance axis labels in km, to sit under a stack of traces. */
export function DistanceAxis({ length }: { length: number }) {
    const [ref, width] = useWidth<HTMLDivElement>();
    const x1 = Math.max(width - RIGHT, LEFT + 1);
    const ticks: number[] = [];
    for (let k = 0; k <= length; k += 1000) ticks.push(k);
    return (
        <div ref={ref} className="w-full" aria-hidden="true">
            {width > 0 && (
                <svg width={width} height={16} style={{ display: 'block' }}>
                    {ticks.map((k) => (
                        <text key={k} x={LEFT + (k / length) * (x1 - LEFT)} y={12} textAnchor="middle" className="tick">
                            {k / 1000}{k === ticks[ticks.length - 1] ? ' km' : ''}
                        </text>
                    ))}
                </svg>
            )}
        </div>
    );
}
