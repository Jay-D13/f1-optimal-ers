const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export type Regulations = '2025' | '2026';
export type Session = 'qualifying' | 'race';

export interface Pole {
    driver: string;
    team: string;
    time: number;
}

export interface RunInfo {
    run_id: string;
    track: string;
    timestamp: string | null;
    regulations: Regulations | null;
    session: Session | null;
    n_laps: number | null;
    lap_time: number | null;
    solver_status: string | null;
    solve_time: number | null;
    recovered_mj: number | null;
    delta_to_pole: number | null;
}

export interface Round {
    round: number;
    name: string;
    circuit: string;
    track: string;
    has_raceline: boolean;
    quali_cap_mj: number;
    race_cap_mj: number;
    superclip_kw: number;
    ramp_kw_s: number;
    power_limited_m: number;
    verified: boolean;
    straight_mode_zones: { corner: string; offset_m: number }[];
    pole: Pole | null;
    latest_run: RunInfo | null;
}

export interface RunSeries {
    s: number[];
    t: number[] | null;
    v: number[];
    soc: number[] | null;
    power: number[] | null;
    throttle: number[] | null;
    brake: number[] | null;
    x: number[] | null;
    y: number[] | null;
}

export interface PoleTrace {
    driver: string;
    lap_time: number;
    s: number[];
    v: number[];
}

export interface RunDetail {
    info: RunInfo;
    summary: {
        track_info?: { total_length?: number };
        energy?: { total_deployed_MJ?: number; total_recovered_MJ?: number };
        velocity_stats?: { max_speed_kmh?: number };
    };
    series: RunSeries;
    straight_mode_zones: [number, number][];
    pole_trace: PoleTrace | null;
}

export interface RunSettings {
    regulations: Regulations;
    session: Session;
    laps: number;
    ds: number;
    collocation: 'euler' | 'trapezoidal' | 'hermite_simpson';
    nlp_solver: 'auto' | 'ipopt' | 'fatrop' | 'sqpmethod';
    ipopt_hessian: 'limited-memory' | 'exact';
    initial_soc: number;
    final_soc_min: number;
    tire_model: 'scalar' | 'dynamic';
    tire_compound: 'soft' | 'medium' | 'hard';
}

export const DEFAULT_SETTINGS: RunSettings = {
    regulations: '2026',
    session: 'qualifying',
    laps: 1,
    ds: 5,
    collocation: 'trapezoidal',
    nlp_solver: 'auto',
    ipopt_hessian: 'exact',
    initial_soc: 0.5,
    final_soc_min: 0.3,
    tire_model: 'scalar',
    tire_compound: 'medium',
};

export interface Job {
    id: string;
    round: number;
    track: string;
    request: RunSettings & { round: number };
    status: 'running' | 'done' | 'failed' | 'cancelled';
    started: string;
    finished: string | null;
    run_id: string | null;
    log: string[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(`${API_URL}${path}`, init);
    if (!res.ok) {
        let detail = res.statusText;
        try {
            detail = (await res.json()).detail ?? detail;
        } catch {
            // Not JSON
        }
        throw new Error(detail);
    }
    return res.json() as Promise<T>;
}

export const api = {
    season: () => request<{ rounds: Round[] }>('/season').then((d) => d.rounds),
    roundRuns: (round: number) => request<{ runs: RunInfo[] }>(`/rounds/${round}/runs`).then((d) => d.runs),
    raceline: (round: number) => request<{ x: number[] | null; y: number[] | null }>(`/rounds/${round}/raceline`),
    run: (track: string, runId: string, round: number) =>
        request<RunDetail>(`/runs/${encodeURIComponent(track)}/${encodeURIComponent(runId)}?round=${round}`),
    jobs: () => request<{ jobs: Job[] }>('/jobs').then((d) => d.jobs),
    startJob: (round: number, settings: RunSettings) =>
        request<Job>('/jobs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ round, ...settings }),
        }),
    cancelJob: (id: string) => request<{ ok: boolean }>(`/jobs/${id}`, { method: 'DELETE' }),
};
