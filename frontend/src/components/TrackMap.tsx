import { useMemo } from 'react';

type Mode = 'deploy' | 'harvest' | 'neutral';

interface TrackMapProps {
    x: number[];
    y: number[];
    /** Lap distance of each point; with `power`, colours the line by ERS mode */
    s?: number[] | null;
    power?: number[] | null;      // (W) per point
    zones?: [number, number][];   // Straight Mode zones (m)
    cursorIndex?: number | null;
    width: number;
    height: number;
}

const MODE_COLOR: Record<Mode, string> = {
    deploy: 'var(--deploy)',
    harvest: 'var(--harvest)',
    neutral: 'var(--neutral)',
};
const THRESHOLD_W = 5e3;

function modeOf(p: number | undefined): Mode {
    if (p == null) return 'neutral';
    return p > THRESHOLD_W ? 'deploy' : p < -THRESHOLD_W ? 'harvest' : 'neutral';
}

/** The lap in plan view: grey before a run, coloured by what the MGU-K does once there is one. */
export function TrackMap({ x, y, s, power, zones = [], cursorIndex, width, height }: TrackMapProps) {
    const geometry = useMemo(() => {
        const n = Math.min(x.length, y.length);
        if (n < 2) return null;
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (let i = 0; i < n; i++) {
            minX = Math.min(minX, x[i]); maxX = Math.max(maxX, x[i]);
            minY = Math.min(minY, y[i]); maxY = Math.max(maxY, y[i]);
        }
        const pad = 18;
        const k = Math.min((width - 2 * pad) / (maxX - minX), (height - 2 * pad) / (maxY - minY));
        const ox = pad + (width - 2 * pad - (maxX - minX) * k) / 2;
        const oy = pad + (height - 2 * pad - (maxY - minY) * k) / 2;
        const X = Array.from({ length: n }, (_, i) => ox + (x[i] - minX) * k);
        const Y = Array.from({ length: n }, (_, i) => oy + (maxY - y[i]) * k);
        const point = (i: number) => `${X[i].toFixed(1)},${Y[i].toFixed(1)}`;

        const full = 'M' + X.map((_, i) => point(i)).join('L') + 'Z';

        // Runs of equal ERS mode
        const runs: { mode: Mode; d: string }[] = [];
        if (power) {
            let start = 0;
            for (let i = 1; i <= n; i++) {
                if (i === n || modeOf(power[i]) !== modeOf(power[start])) {
                    const end = Math.min(i, n - 1);
                    let d = 'M' + point(start);
                    for (let j = start + 1; j <= end; j++) d += 'L' + point(j);
                    runs.push({ mode: modeOf(power[start]), d });
                    start = i;
                }
            }
        }

        const zonePaths = s ? zones.map(([a, b]) => {
            // A zone may run across the timing line (a > b)
            const inside = (v: number) => (a <= b ? v >= a && v <= b : v >= a || v <= b);
            let d = '';
            let open = false;
            for (let i = 0; i < n; i++) {
                if (inside(s[i])) {
                    d += (open ? 'L' : 'M') + point(i);
                    open = true;
                } else {
                    open = false;
                }
            }
            return d;
        }).filter(Boolean) : [];

        return { X, Y, full, runs, zonePaths };
    }, [x, y, s, power, zones, width, height]);

    if (!geometry) return null;
    const { X, Y, full, runs, zonePaths } = geometry;
    const c = cursorIndex != null ? Math.min(cursorIndex, X.length - 1) : null;

    return (
        <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Track map" style={{ display: 'block' }}>
            {zonePaths.map((d, i) => (
                <path key={`z${i}`} d={d} fill="none" stroke="var(--ink)" strokeWidth={24} strokeOpacity={0.14} strokeLinecap="butt" strokeLinejoin="round" />
            ))}
            <path d={full} fill="none" stroke="var(--trackbed)" strokeWidth={16} strokeLinejoin="round" />
            {runs.length > 0
                ? runs.map((r, i) => <path key={i} d={r.d} fill="none" stroke={MODE_COLOR[r.mode]} strokeWidth={5} strokeLinejoin="round" />)
                : <path d={full} fill="none" stroke="var(--neutral)" strokeWidth={4} strokeLinejoin="round" />}
            <rect x={X[0] - 10} y={Y[0] - 2.5} width={20} height={5} fill="var(--ink)" />
            {c != null && (
                <rect x={X[c] - 7} y={Y[c] - 7} width={14} height={14} fill="var(--paper)" stroke="var(--ink)" strokeWidth={3} />
            )}
        </svg>
    );
}
