const API_URL = 'http://localhost:8000';

export interface TrackPoint {
    x: number;
    y: number;
    w_left: number;
    w_right: number;
}

export interface RacelinePoint {
    x: number;
    y: number;
}

export interface SimulationParams {
    track: string;
    year?: number;
    laps: number;
    regulations: '2025' | '2026';
    initial_soc: number;
    final_soc_min: number;
    per_lap_final_soc_min?: number;
    ds: number;
    collocation: 'euler' | 'trapezoidal' | 'hermite_simpson';
    nlp_solver: 'auto' | 'ipopt' | 'fatrop';
    use_tumftm: boolean;
    driver?: string;
}

export interface FastF1Event {
    name: string;
    location: string;
    round: number;
    country: string;
    official_name: string;
}

export interface FastF1Driver {
    id: string;
    code: string;
    team: string;
}

export const api = {
    getTracks: async () => {
        const res = await fetch(`${API_URL}/tracks`);
        const data = await res.json();
        return data.tracks as string[];
    },

    getTrackData: async (trackId: string) => {
        const res = await fetch(`${API_URL}/track/${trackId}`);
        const data = await res.json();
        return data.track_data as TrackPoint[];
    },

    getRaceline: async (trackId: string, type: 'tumftm' | 'fastf1' = 'tumftm') => {
        const res = await fetch(`${API_URL}/raceline/${trackId}?type=${type}`);
        const data = await res.json();
        return data.raceline as RacelinePoint[] | null;
    },

    runSimulation: async (params: SimulationParams) => {
        const res = await fetch(`${API_URL}/simulation`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        return await res.json();
    },

    getFastF1Years: async () => {
        const res = await fetch(`${API_URL}/fastf1/years`);
        const data = await res.json();
        return data.years as number[];
    },

    getFastF1Tracks: async (year: number) => {
        const res = await fetch(`${API_URL}/fastf1/tracks/${year}`);
        const data = await res.json();
        return data.tracks as FastF1Event[];
    },

    getFastF1Drivers: async (year: number, location: string) => {
        const res = await fetch(`${API_URL}/fastf1/drivers/${year}/${location}`);
        const data = await res.json();
        return data.drivers as FastF1Driver[];
    }
};
