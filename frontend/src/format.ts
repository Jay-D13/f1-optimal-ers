/** 81.786 → "1:21.786" */
export function formatLap(seconds: number | null | undefined): string {
    if (seconds == null || !Number.isFinite(seconds)) return '–:––.–––';
    const minutes = Math.floor(seconds / 60);
    const rest = seconds - minutes * 60;
    return `${minutes}:${rest.toFixed(3).padStart(6, '0')}`;
}

/** -2.3527 → "−2.353", 0.1 → "+0.100" */
export function formatDelta(seconds: number | null | undefined, digits = 3): string {
    if (seconds == null || !Number.isFinite(seconds)) return '–';
    const sign = seconds < 0 ? '−' : '+';
    return `${sign}${Math.abs(seconds).toFixed(digits)}`;
}

/** "Gasly" → "GAS" */
export function driverCode(name: string): string {
    return name.slice(0, 3).toUpperCase();
}

export function formatTimestamp(iso: string | null): string {
    if (!iso) return '';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

/** Index of the value in a sorted array closest to x. */
export function nearestIndex(sorted: number[], x: number): number {
    let lo = 0;
    let hi = sorted.length - 1;
    while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (sorted[mid] < x) lo = mid;
        else hi = mid;
    }
    return Math.abs(sorted[lo] - x) <= Math.abs(sorted[hi] - x) ? lo : hi;
}

export function median(values: number[]): number | null {
    if (values.length === 0) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
