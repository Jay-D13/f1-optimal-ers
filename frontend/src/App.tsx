import React, { useState, useEffect, useMemo } from 'react';
import { Activity, Settings, Map, Play, Loader2, Database, ChevronRight, ChevronDown, ChevronUp, AlertCircle, Download, Upload, Gauge, BatteryCharging, Wrench, Cpu, ShieldCheck, ShieldAlert } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import TrackMap from './components/TrackMap';
import { api, type TrackPoint, type RacelinePoint, type SimulationParams, type FastF1Event, type FastF1Driver } from './api';
import { ThemeToggle } from './components/ThemeToggle';
import ResultsDashboard from './components/ResultsDashboard';
import ConfigSectionCard from './components/config/ConfigSectionCard';
import SegmentedControl from './components/config/SegmentedControl';
import FieldHint from './components/config/FieldHint';
import type { FieldMeta, SegmentedOption, ValidationState } from './components/config/types';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

const DEFAULT_SIM_PARAMS: SimulationParams = {
    track: '',
    year: 2024,
    laps: 1,
    regulations: '2025',
    session: 'qualifying',
    event: undefined,
    initial_soc: 0.5,
    final_soc_min: 0.3,
    per_lap_final_soc_min: undefined,
    flying_lap: true,
    enable_tire_degradation: false,
    tire_wear_rate_per_lap: 0.012,
    tire_min_grip_scale: 0.88,
    tire_model: 'scalar',
    tire_compound: 'medium',
    ambient_temp_c: 25.0,
    track_temp_c: 35.0,
    tire_init_temp_c: 80.0,
    ds: 5.0,
    collocation: 'trapezoidal',
    nlp_solver: 'auto',
    ipopt_linear_solver: 'mumps',
    ipopt_hessian: 'exact',
    use_tumftm: true,
    driver: undefined,
};

const REGULATION_ERS_DELTA: Record<'2025' | '2026', { label: string; deployKw: number; recoveryLimitMj: number; usableBatteryMj: number }> = {
    '2025': {
        label: '2014-2025',
        deployKw: 120,
        recoveryLimitMj: 2.0,
        usableBatteryMj: 4.0,
    },
    '2026': {
        label: '2026-',
        deployKw: 350,
        recoveryLimitMj: 8.5,
        usableBatteryMj: 4.0,
    },
};

function regulationLabel(value: '2025' | '2026'): string {
    return REGULATION_ERS_DELTA[value].label;
}

function App() {
    const [activeTab, setActiveTab] = useState<'track' | 'settings' | 'results'>('track');

    // Track Selection State
    const [trackSource, setTrackSource] = useState<'local' | 'fastf1'>('local');
    const [localTracks, setLocalTracks] = useState<string[]>([]);

    // FastF1 Selection State
    const [f1Years, setF1Years] = useState<number[]>([]);
    const [selectedYear, setSelectedYear] = useState<number>(2024);
    const [f1Events, setF1Events] = useState<FastF1Event[]>([]);
    const [f1Drivers, setF1Drivers] = useState<FastF1Driver[]>([]);
    const [loadingF1, setLoadingF1] = useState(false);

    // Selected Context
    const [selectedTrackName, setSelectedTrackName] = useState<string | null>(null);
    const [selectedDriver, setSelectedDriver] = useState<string | null>(null); // For FastF1
    const [trackData, setTrackData] = useState<TrackPoint[]>([]);
    const [raceline, setRaceline] = useState<RacelinePoint[] | null>(null);
    const [loadingTrack, setLoadingTrack] = useState(false);

    // Simulation State
    const [simulating, setSimulating] = useState(false);
    const [results, setResults] = useState<any>(null);
    const [simParams, setSimParams] = useState<SimulationParams>(DEFAULT_SIM_PARAMS);

    const [openSections, setOpenSections] = useState({
        simulation: true,
        regulations: false,
        tire: false,
        advanced: false,
    });

    const lapStartOptions: SegmentedOption<'flying' | 'standing'>[] = [
        { value: 'flying', label: 'Flying Lap', description: 'Continuity at lap boundary' },
        { value: 'standing', label: 'Standing Start', description: 'No carry-over speed at start' },
    ];

    const regulationOptions: SegmentedOption<'2025' | '2026'>[] = [
        { value: '2025', label: '2014-2025', description: '120 kW MGU-K era' },
        { value: '2026', label: '2026-', description: '350 kW MGU-K era' },
    ];

    const degradationOptions: SegmentedOption<'enabled' | 'disabled'>[] = [
        { value: 'enabled', label: 'Enabled', description: 'Apply grip decay each lap' },
        { value: 'disabled', label: 'Disabled', description: 'Constant grip for all laps' },
    ];

    const collocationOptions: SegmentedOption<'euler' | 'trapezoidal' | 'hermite_simpson'>[] = [
        { value: 'euler', label: 'Euler', description: 'Fastest first-order integration' },
        { value: 'trapezoidal', label: 'Trapezoidal', description: 'Balanced second-order accuracy' },
        { value: 'hermite_simpson', label: 'Hermite-Simpson', description: 'Higher-order accuracy' },
    ];

    const solverOptions: SegmentedOption<'auto' | 'ipopt' | 'fatrop' | 'sqpmethod'>[] = [
        { value: 'auto', label: 'Auto', description: 'Backend-selected default' },
        { value: 'ipopt', label: 'IPOPT', description: 'Robust interior-point backend' },
        { value: 'fatrop', label: 'Fatrop', description: 'Fast sparse nonlinear backend' },
        { value: 'sqpmethod', label: 'SQPMethod', description: 'Sequential quadratic programming' },
    ];

    const hessianOptions: SegmentedOption<'limited-memory' | 'exact'>[] = [
        { value: 'limited-memory', label: 'Limited Memory', description: 'L-BFGS approximation' },
        { value: 'exact', label: 'Exact', description: 'Full Hessian computation' },
    ];

    const toggleSection = (section: keyof typeof openSections) => {
        setOpenSections((prev) => ({ ...prev, [section]: !prev[section] }));
    };

    const resetSimulationSection = () => {
        setSimParams((prev) => ({
            ...prev,
            laps: DEFAULT_SIM_PARAMS.laps,
            initial_soc: DEFAULT_SIM_PARAMS.initial_soc,
            final_soc_min: DEFAULT_SIM_PARAMS.final_soc_min,
            per_lap_final_soc_min: DEFAULT_SIM_PARAMS.per_lap_final_soc_min,
            flying_lap: DEFAULT_SIM_PARAMS.flying_lap,
        }));
    };

    const resetRegulationVehicleSection = () => {
        setSimParams((prev) => ({
            ...prev,
            regulations: DEFAULT_SIM_PARAMS.regulations,
        }));
    };

    const resetTireSection = () => {
        setSimParams((prev) => ({
            ...prev,
            enable_tire_degradation: DEFAULT_SIM_PARAMS.enable_tire_degradation,
            tire_wear_rate_per_lap: DEFAULT_SIM_PARAMS.tire_wear_rate_per_lap,
            tire_min_grip_scale: DEFAULT_SIM_PARAMS.tire_min_grip_scale,
        }));
    };

    const resetAdvancedSolverSection = () => {
        setSimParams((prev) => ({
            ...prev,
            ds: DEFAULT_SIM_PARAMS.ds,
            collocation: DEFAULT_SIM_PARAMS.collocation,
            nlp_solver: DEFAULT_SIM_PARAMS.nlp_solver,
            ipopt_linear_solver: DEFAULT_SIM_PARAMS.ipopt_linear_solver,
            ipopt_hessian: DEFAULT_SIM_PARAMS.ipopt_hessian,
        }));
    };

    const fieldErrors = useMemo(() => {
        const errors: Partial<Record<keyof SimulationParams, string>> = {};

        if (!Number.isFinite(simParams.laps) || simParams.laps < 1) {
            errors.laps = 'Laps must be at least 1.';
        }
        if (!Number.isFinite(simParams.initial_soc) || simParams.initial_soc < 0 || simParams.initial_soc > 1) {
            errors.initial_soc = 'Initial SOC must stay within 0 and 1.';
        }
        if (!Number.isFinite(simParams.final_soc_min) || simParams.final_soc_min < 0 || simParams.final_soc_min > 1) {
            errors.final_soc_min = 'Final SOC floor must stay within 0 and 1.';
        }
        if (
            simParams.per_lap_final_soc_min !== undefined
            && (!Number.isFinite(simParams.per_lap_final_soc_min) || simParams.per_lap_final_soc_min < 0 || simParams.per_lap_final_soc_min > 1)
        ) {
            errors.per_lap_final_soc_min = 'Per-lap SOC floor must stay within 0 and 1.';
        }
        if (!Number.isFinite(simParams.ds) || simParams.ds <= 0) {
            errors.ds = 'Spatial step must be greater than zero.';
        }
        if (!Number.isFinite(simParams.tire_wear_rate_per_lap) || simParams.tire_wear_rate_per_lap < 0) {
            errors.tire_wear_rate_per_lap = 'Wear rate must be non-negative.';
        }
        if (!Number.isFinite(simParams.tire_min_grip_scale) || simParams.tire_min_grip_scale <= 0 || simParams.tire_min_grip_scale > 1) {
            errors.tire_min_grip_scale = 'Grip floor must be in the interval (0, 1].';
        }

        return errors;
    }, [simParams]);

    const validationState = useMemo<ValidationState>(() => {
        const errors = Object.values(fieldErrors).filter((error): error is string => Boolean(error));
        const warnings: string[] = [];

        if (simParams.enable_tire_degradation && simParams.laps <= 1) {
            warnings.push('Tire degradation has no visible effect on a single-lap horizon.');
        }
        if (simParams.nlp_solver === 'ipopt' && simParams.laps > 8) {
            warnings.push('Large horizons with IPOPT may require longer solve times.');
        }

        return {
            isValid: errors.length === 0,
            errors,
            warnings,
        };
    }, [fieldErrors, simParams.enable_tire_degradation, simParams.laps, simParams.nlp_solver]);

    const canRunSimulation = Boolean(selectedTrackName) && !simulating && validationState.isValid;

    // Initial Data Load
    useEffect(() => {
        api.getTracks().then(setLocalTracks).catch(console.error);
        api.getFastF1Years().then(setF1Years).catch(console.error);
    }, []);

    // Fetch FastF1 Tracks when Year changes
    useEffect(() => {
        if (trackSource === 'fastf1') {
            setLoadingF1(true);
            api.getFastF1Tracks(selectedYear)
                .then(setF1Events)
                .catch(console.error)
                .finally(() => setLoadingF1(false));
        }
    }, [selectedYear, trackSource]);

    // Handle Track Selection (Local)
    const handleSelectLocalTrack = async (track: string) => {
        setLoadingTrack(true);
        try {
            setSelectedTrackName(track);
            setSelectedDriver(null);
            setSimParams(prev => ({ ...prev, track, use_tumftm: true })); // Default to TUMFTM for local

            const tData = await api.getTrackData(track);
            setTrackData(tData);
            const rData = await api.getRaceline(track);
            setRaceline(rData);
        } catch (e) {
            console.error(e);
        } finally {
            setLoadingTrack(false);
        }
    };

    // Handle Track Selection (FastF1)
    const handleSelectF1Track = async (event: FastF1Event) => {
        // For FastF1, we might not have local CSVs yet unless cached
        // For now, let's assume we proceed to Driver selection
        // If we pick a track here, we should probably try to load it if available locally
        // Or just set the context for the simulation
        setSelectedTrackName(event.location); // FastF1 uses location/event name

        setLoadingF1(true);
        try {
            const drivers = await api.getFastF1Drivers(selectedYear, event.location);
            setF1Drivers(drivers);

            // Try to load track visualization if exists locally (fuzzy match)
            // This part is tricky without strict mapping. 
            // We'll skip viz for purely new FastF1 tracks for now or try name match
            const match = localTracks.find(t => t.toLowerCase() === event.location.toLowerCase() || t.toLowerCase() === event.country.toLowerCase());
            if (match) {
                const tData = await api.getTrackData(match);
                setTrackData(tData);
                const rData = await api.getRaceline(match);
                setRaceline(rData);
            } else {
                setTrackData([]);
                setRaceline(null);
            }
        } catch (e) {
            console.error(e);
        } finally {
            setLoadingF1(false);
        }
    };

    const handleRunSimulation = async () => {
        if (!selectedTrackName || !validationState.isValid) return;
        setSimulating(true);
        try {
            const params = {
                ...simParams,
                track: selectedTrackName,
                year: trackSource === 'fastf1' ? selectedYear : undefined,
                driver: selectedDriver || undefined,
                use_tumftm: trackSource === 'local', // Prefer TUMFTM for local, FastF1 for remote
            };

            const res = await api.runSimulation(params);
            setResults(res);
            setActiveTab('results');
        } catch (e) {
            console.error(e);
        } finally {
            setSimulating(false);
        }
    };

    // History Management
    const [history, setHistory] = useState<RacelinePoint[][]>([]);
    const [historyIndex, setHistoryIndex] = useState(-1);

    // Initial load history init
    useEffect(() => {
        if (raceline && history.length === 0) {
            setHistory([raceline]);
            setHistoryIndex(0);
        }
    }, [raceline]); // Careful, this might reset history on every raceline change if we aren't careful? 
    // Actually, raceline changes when we drag. We need a separate way to init history.
    // Let's wrap raceline update.

    const updateRaceline = (newRaceline: RacelinePoint[], addToHistory = true) => {
        setRaceline(newRaceline);
        if (addToHistory) {
            const newHistory = history.slice(0, historyIndex + 1);
            newHistory.push(newRaceline);
            setHistory(newHistory);
            setHistoryIndex(newHistory.length - 1);
        }
    };

    const handleUndo = () => {
        if (historyIndex > 0) {
            const prevIndex = historyIndex - 1;
            setRaceline(history[prevIndex]);
            setHistoryIndex(prevIndex);
        }
    };

    const handleRedo = () => {
        if (historyIndex < history.length - 1) {
            const nextIndex = historyIndex + 1;
            setRaceline(history[nextIndex]);
            setHistoryIndex(nextIndex);
        }
    };

    const handleReset = async () => {
        if (!selectedTrackName) return;
        if (confirm("Reset raceline to default?")) {
            const rData = await api.getRaceline(selectedTrackName);
            setRaceline(rData);
            // Reset history too? Or just add to history?
            // Let's add to history so simpler
            updateRaceline(rData!, true);
        }
    };

    const handleSaveRaceline = () => {
        if (!raceline || !selectedTrackName) return;

        // Convert to CSV
        const header = "x,y\n";
        const rows = raceline.map(p => `${p.x},${p.y}`).join("\n");
        const csvContent = header + rows;

        const dataStr = "data:text/csv;charset=utf-8," + encodeURIComponent(csvContent);
        const downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", `${selectedTrackName}_raceline.csv`);
        document.body.appendChild(downloadAnchorNode); // required for firefox
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    };

    const handleLoadRaceline = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const content = e.target?.result as string;
                const lines = content.split('\n');

                // Parse CSV
                const newRaceline: RacelinePoint[] = [];
                let startIdx = 0;

                // Skip header if present
                if (lines.length > 0 && lines[0].toLowerCase().includes('x')) {
                    startIdx = 1;
                }

                for (let i = startIdx; i < lines.length; i++) {
                    const line = lines[i].trim();
                    if (!line) continue;

                    const parts = line.split(',');
                    if (parts.length >= 2) {
                        const x = parseFloat(parts[0]);
                        const y = parseFloat(parts[1]);
                        if (!isNaN(x) && !isNaN(y)) {
                            newRaceline.push({ x, y });
                        }
                    }
                }

                if (newRaceline.length > 0) {
                    updateRaceline(newRaceline, true);
                } else {
                    alert('Invalid raceline CSV file.');
                }
            } catch (error) {
                console.error("Error parsing CSV:", error);
                alert('Error parsing CSV file.');
            }
        };
        reader.readAsText(file);
        // Reset input so same file can be selected again
        event.target.value = '';
    };

    const handleTrackSelectRef = async (track: string) => {
        // ... existing logic but reset history
        setHistory([]);
        setHistoryIndex(-1);
        await handleSelectLocalTrack(track);
    };

    const sectionHeaderActions = (
        section: keyof typeof openSections,
        onReset: () => void,
        collapseAriaLabel: string,
    ) => (
        <div className="flex items-center gap-2">
            <button
                type="button"
                onClick={onReset}
                className="rounded-md border border-panel-border bg-panel-muted/60 px-2 py-1 font-mono text-[10px] uppercase tracking-wide text-retro-text/80 hover:border-f1-red/40"
            >
                Reset
            </button>
            <button
                type="button"
                onClick={() => toggleSection(section)}
                className="rounded-md border border-panel-border bg-panel-muted/60 p-1 text-retro-text/70 hover:border-f1-red/40"
                aria-label={collapseAriaLabel}
            >
                {openSections[section] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
        </div>
    );



    return (
        <div className="min-h-screen bg-retro-bg font-sans text-retro-text selection:bg-f1-red/20 transition-colors duration-200">
            {/* Top Bar */}
            <header className="fixed top-0 left-0 right-0 h-14 border-b border-retro-border bg-retro-bg/95 backdrop-blur z-50 flex items-center justify-between px-6 transition-colors duration-200">
                <div className="flex items-center gap-3">
                    <div className="w-3 h-3 bg-f1-red rounded-full animate-pulse" />
                    <h1 className="font-mono font-bold text-lg tracking-tight">
                        F1 ERS OPTIMAL CONTROL <span className="text-retro-border/50 text-xs">v1.3.0</span>
                    </h1>
                </div>
                <div className="flex items-center gap-4 text-sm font-mono">
                    <span className="flex items-center gap-2 px-3 py-1 bg-retro-text/5 rounded-full">
                        <Database size={14} />
                        <span>SOURCE: {trackSource === 'local' ? 'LOCAL DB' : 'FASTF1 API'}</span>
                    </span>
                    <span className="flex items-center gap-2 text-green-600 dark:text-green-400">
                        <Activity size={14} />
                        <span>SYSTEM: ONLINE</span>
                    </span>
                    <div className="w-px h-4 bg-retro-border/20" />
                    <ThemeToggle />
                </div>
            </header>

            <div className="pt-20 px-6 pb-6 h-screen flex gap-6 overflow-hidden">
                {/* Sidebar */}
                <nav className="w-64 flex-shrink-0 flex flex-col gap-2">
                    <NavButton active={activeTab === 'track'} onClick={() => setActiveTab('track')} icon={<Map size={18} />} label="TRACK SELECTION" desc="Select & edit track" />
                    <NavButton active={activeTab === 'settings'} onClick={() => setActiveTab('settings')} icon={<Settings size={18} />} label="CONFIGURATION" desc="Vehicle & Solver setup" />
                    <NavButton active={activeTab === 'results'} onClick={() => setActiveTab('results')} icon={<Activity size={18} />} label="SIMULATION" desc="Run & Analyze results" />

                    <div className="mt-auto p-4 border border-retro-border rounded-lg bg-white/50 dark:bg-white/5">
                        <div className="mb-4">
                            <div className="font-mono text-xs font-bold mb-2 text-retro-border">TRACK SOURCE</div>
                            <div className="flex bg-retro-border/10 p-1 rounded">
                                <button
                                    onClick={() => setTrackSource('local')}
                                    className={cn("flex-1 text-xs font-mono py-1 rounded transition-all", trackSource === 'local' ? "bg-white dark:bg-retro-border dark:text-white shadow text-black" : "text-retro-text/50")}
                                >LOCAL</button>
                                <button
                                    onClick={() => setTrackSource('fastf1')}
                                    className={cn("flex-1 text-xs font-mono py-1 rounded transition-all", trackSource === 'fastf1' ? "bg-white dark:bg-retro-border dark:text-white shadow text-black" : "text-retro-text/50")}
                                >FASTF1</button>
                            </div>
                        </div>

                        <button
                            onClick={handleRunSimulation}
                            disabled={!canRunSimulation}
                            className="w-full flex items-center justify-center gap-2 bg-f1-red hover:bg-red-600 disabled:bg-gray-400 text-white font-mono text-sm py-2 px-4 rounded transition-colors shadow-sm active:translate-y-[1px]"
                        >
                            {simulating ? <Loader2 className="animate-spin" size={16} /> : <Play size={16} />}
                            {simulating ? 'RUNNING...' : 'RUN SIMULATION'}
                        </button>
                        {!validationState.isValid && (
                            <p className="mt-2 font-mono text-[10px] uppercase tracking-wide text-amber-600 dark:text-amber-400">
                                Resolve {validationState.errors.length} config issue{validationState.errors.length === 1 ? '' : 's'} before running.
                            </p>
                        )}
                    </div>
                </nav>

                {/* Main Content */}
                <main className="flex-1 bg-white dark:bg-white/5 border border-retro-border rounded-lg shadow-sm overflow-hidden flex flex-col relative transition-colors duration-200">
                    <div className="absolute top-0 right-0 p-2 z-10"><div className="w-2 h-2 border-t border-r border-retro-border" /></div>
                    <div className="absolute bottom-0 left-0 p-2 z-10"><div className="w-2 h-2 border-b border-l border-retro-border" /></div>

                    <div className="flex-1 p-6 overflow-auto h-full box-border">
                        {activeTab === 'track' && (
                            <div className="h-full flex flex-col gap-4">
                                <div className="flex items-end justify-between border-b-2 border-retro-border pb-4 flex-shrink-0">
                                    <div>
                                        <h2 className="text-3xl font-bold font-mono uppercase">
                                            {trackSource === 'local' ? 'Local Tracks' : 'FastF1 Explorer'}
                                        </h2>
                                        <p className="text-retro-text/60 mt-1 max-w-lg">
                                            {trackSource === 'local'
                                                ? "Select a pre-processed track from the database."
                                                : "Browse real F1 sessions to use specific year/driver data."}
                                        </p>
                                    </div>
                                    {selectedTrackName && (
                                        <div className="font-mono text-sm px-3 py-1 bg-retro-text/5 rounded flex flex-col items-end">
                                            <span className="text-xs text-retro-text/40">SELECTED</span>
                                            <span className="font-bold">
                                                {selectedTrackName}
                                                {selectedDriver ? ' (' + selectedDriver + ')' : ''}
                                            </span>
                                        </div>
                                    )}
                                </div>

                                {/* Track Browsing UI */}
                                {!selectedTrackName ? (
                                    trackSource === 'local' ? (
                                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 overflow-y-auto">
                                            {localTracks.map((track) => (
                                                <div key={track} onClick={() => handleTrackSelectRef(track)}
                                                    className="aspect-video bg-retro-bg border border-retro-border rounded hover:border-f1-red transition-all cursor-pointer flex items-center justify-center group relative hover:shadow-md">
                                                    <span className="font-mono text-lg uppercase tracking-wider group-hover:text-f1-red transition-colors">{track}</span>
                                                </div>
                                            ))}
                                        </div>
                                    ) : (
                                        <div className="flex flex-col gap-4 h-full">
                                            {/* Year Selector */}
                                            <div className="flex gap-2 overflow-x-auto pb-2 border-b border-retro-border/10">
                                                {f1Years.map(year => (
                                                    <button
                                                        key={year}
                                                        onClick={() => setSelectedYear(year)}
                                                        className={cn("px-4 py-2 font-mono rounded border transition-all", selectedYear === year ? "bg-f1-black text-white border-f1-black dark:bg-white dark:text-black dark:border-white" : "bg-white dark:bg-transparent border-retro-border hover:border-f1-red")}
                                                    >
                                                        {year}
                                                    </button>
                                                ))}
                                            </div>

                                            {/* Event List */}
                                            {loadingF1 ? (
                                                <div className="flex-1 flex items-center justify-center"><Loader2 className="animate-spin text-f1-red" size={32} /></div>
                                            ) : (
                                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 overflow-y-auto pb-20">
                                                    {f1Events.map(event => (
                                                        <div key={event.round} onClick={() => handleSelectF1Track(event)}
                                                            className="p-4 bg-retro-bg border border-retro-border rounded hover:border-f1-red hover:shadow-md cursor-pointer transition-all flex flex-col justify-between group">
                                                            <div>
                                                                <div className="font-mono text-xs text-f1-red mb-1">ROUND {event.round}</div>
                                                                <div className="font-bold text-lg leading-tight group-hover:text-f1-red transition-colors">{event.name}</div>
                                                                <div className="text-sm text-retro-text/60 mt-1">{event.location}, {event.country}</div>
                                                            </div>
                                                            <div className="mt-4 flex justify-end opacity-0 group-hover:opacity-100 transition-opacity">
                                                                <ChevronRight size={18} />
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    )
                                ) : (
                                    <div className="flex-1 flex flex-col gap-4 relative">
                                        <div className="absolute top-0 left-0 right-0 z-10 flex gap-4 justify-between pointer-events-none p-4">
                                            <div className="flex gap-2 pointer-events-auto">
                                                <button onClick={() => setSelectedTrackName(null)} className="flex items-center gap-1 text-xs font-mono text-retro-text/60 hover:text-f1-red bg-white/90 dark:bg-black/50 backdrop-blur px-3 py-2 rounded border border-retro-border/20 shadow-sm transition-all">
                                                    ← BACK TO LIST
                                                </button>
                                                {trackSource === 'fastf1' && !selectedDriver && (
                                                    <div className="flex items-center gap-2 text-sm bg-f1-blue/10 text-f1-blue px-3 py-1 rounded font-mono animate-pulse">
                                                        <AlertCircle size={14} /> Please Select a Driver Below
                                                    </div>
                                                )}
                                            </div>

                                            {/* Editor Controls */}
                                            {trackSource === 'local' && (
                                                <div className="flex gap-1 pointer-events-auto bg-white/90 dark:bg-black/50 backdrop-blur rounded border border-retro-border/20 p-1 shadow-sm">
                                                    <button onClick={handleUndo} disabled={historyIndex <= 0} className="px-3 py-1 text-xs font-mono hover:bg-black/5 rounded disabled:opacity-30 disabled:hover:bg-transparent">UNDO</button>
                                                    <div className="w-px bg-retro-border/20 my-1"></div>
                                                    <button onClick={handleRedo} disabled={historyIndex >= history.length - 1} className="px-3 py-1 text-xs font-mono hover:bg-black/5 rounded disabled:opacity-30 disabled:hover:bg-transparent">REDO</button>
                                                    <div className="w-px bg-retro-border/20 my-1"></div>
                                                    <button onClick={handleReset} className="px-3 py-1 text-xs font-mono hover:bg-red-50 text-f1-red rounded">RESET</button>
                                                </div>
                                            )}
                                        </div>

                                        {/* CSV Controls - Bottom Left */}
                                        {trackSource === 'local' && selectedTrackName && (
                                            <div className="absolute bottom-2 left-2 z-10 flex gap-2 pointer-events-auto">
                                                <button
                                                    onClick={handleSaveRaceline}
                                                    disabled={!raceline}
                                                    className="p-2 bg-white/80 dark:bg-black/80 text-retro-text hover:text-f1-red border border-retro-border/10 rounded shadow-sm backdrop-blur disabled:opacity-30 transition-colors"
                                                    title="Save Raceline (CSV)"
                                                >
                                                    <Download size={16} />
                                                </button>
                                                <label
                                                    className="p-2 bg-white/80 dark:bg-black/80 text-retro-text hover:text-f1-red border border-retro-border/10 rounded shadow-sm backdrop-blur cursor-pointer transition-colors flex items-center justify-center"
                                                    title="Load Raceline (CSV)"
                                                >
                                                    <Upload size={16} />
                                                    <input type="file" accept=".csv" onChange={handleLoadRaceline} className="hidden" />
                                                </label>
                                            </div>
                                        )}

                                        {/* Driver Selection for FastF1 */}
                                        {trackSource === 'fastf1' && !selectedDriver && (
                                            <div className="absolute inset-0 z-20 bg-white/95 dark:bg-black/95 backdrop-blur flex flex-col p-10">
                                                <h3 className="font-mono font-bold text-xl mb-6">SELECT DRIVER TELEMENTRY ({selectedYear} {selectedTrackName})</h3>
                                                {loadingF1 ? (
                                                    <div className="flex items-center justify-center h-40"><Loader2 className="animate-spin" size={32} /></div>
                                                ) : (
                                                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3 overflow-y-auto">
                                                        {f1Drivers.map(d => (
                                                            <button key={d.id} onClick={() => setSelectedDriver(d.id)}
                                                                className="p-3 border border-retro-border rounded hover:bg-f1-black hover:text-white dark:hover:bg-white dark:hover:text-black transition-all text-left group">
                                                                <div className="font-mono font-bold text-lg">{d.code}</div>
                                                                <div className="text-xs text-retro-text/60 group-hover:text-white/60 truncate">{d.team}</div>
                                                            </button>
                                                        ))}
                                                    </div>
                                                )}
                                                <div className="mt-auto">
                                                    <button onClick={() => setSelectedTrackName(null)} className="text-sm hover:underline">Cancel</button>
                                                </div>
                                            </div>
                                        )}

                                        <div className="flex-1 border border-retro-border/50 rounded overflow-hidden relative bg-retro-bg">
                                            {trackData.length > 0 ? (
                                                <TrackMap
                                                    trackData={trackData}
                                                    raceline={raceline}
                                                    onRacelineChange={(nr) => updateRaceline(nr)}
                                                    editable={true}
                                                />
                                            ) : (
                                                <div className="w-full h-full flex flex-col items-center justify-center text-retro-text/40 font-mono p-8 text-center">
                                                    {loadingTrack ? <Loader2 className="animate-spin mb-2" size={32} /> : <Map size={48} className="mb-4 opacity-20" />}
                                                    <div className="max-w-md">
                                                        {loadingTrack ? "LOADING TRACK DATA..." : "NO PRE-PROCESSED VISUALIZATION AVAILABLE FOR THIS TRACK."}
                                                    </div>
                                                    {!loadingTrack && <div className="mt-2 text-xs">You can still run the simulation. FastF1 data will be downloaded by the backend.</div>}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {activeTab === 'settings' && (
                            <div className="config-canvas pb-20">
                                <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(260px,1fr)]">
                                    <div className="space-y-4">
                                        <div className="mb-1 flex items-center justify-between border-b border-panel-border/70 pb-2">
                                            <h2 className="font-mono text-2xl font-bold uppercase tracking-tight">Configuration</h2>
                                            <span className={cn(
                                                'rounded-full border px-3 py-1 font-mono text-xs uppercase tracking-wide',
                                                validationState.isValid
                                                    ? 'border-emerald-500/35 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
                                                    : 'border-amber-500/35 bg-amber-500/10 text-amber-700 dark:text-amber-300',
                                            )}>
                                                {validationState.isValid ? 'Run-Ready' : `${validationState.errors.length} issue${validationState.errors.length === 1 ? '' : 's'}`}
                                            </span>
                                        </div>

                                        <ConfigSectionCard
                                            icon={<Gauge size={16} />}
                                            title="Simulation Parameters"
                                            accent={fieldErrors.laps || fieldErrors.initial_soc || fieldErrors.final_soc_min ? 'warn' : 'neutral'}
                                            headerRight={sectionHeaderActions(
                                                'simulation',
                                                resetSimulationSection,
                                                openSections.simulation ? 'Collapse simulation parameters' : 'Expand simulation parameters',
                                            )}
                                        >
                                            {openSections.simulation && (
                                                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                                                    <ConfigInput
                                                        meta={{ id: 'laps', label: 'LAPS', hint: 'Number of laps in optimization horizon.', error: fieldErrors.laps } satisfies FieldMeta}
                                                        type="number"
                                                        value={simParams.laps}
                                                        onValueChange={(v) => setSimParams({ ...simParams, laps: Number(v) })}
                                                        min={1}
                                                    />
                                                    <ConfigInput
                                                        meta={{ id: 'initial_soc', label: 'INITIAL SOC', hint: 'Battery state at start (0 to 1).', error: fieldErrors.initial_soc } satisfies FieldMeta}
                                                        type="number"
                                                        value={simParams.initial_soc}
                                                        onValueChange={(v) => setSimParams({ ...simParams, initial_soc: Number(v) })}
                                                        min={0}
                                                        max={1}
                                                        step={0.05}
                                                    />
                                                    <ConfigInput
                                                        meta={{ id: 'final_soc_min', label: 'MIN FINAL SOC', hint: 'Minimum SOC at end of horizon.', error: fieldErrors.final_soc_min } satisfies FieldMeta}
                                                        type="number"
                                                        value={simParams.final_soc_min}
                                                        onValueChange={(v) => setSimParams({ ...simParams, final_soc_min: Number(v) })}
                                                        min={0}
                                                        max={1}
                                                        step={0.05}
                                                    />
                                                    <ConfigInput
                                                        meta={{ id: 'per_lap_final_soc_min', label: 'PER-LAP MIN SOC', hint: 'Optional floor at each lap boundary.', error: fieldErrors.per_lap_final_soc_min } satisfies FieldMeta}
                                                        type="number"
                                                        value={simParams.per_lap_final_soc_min ?? ''}
                                                        onValueChange={(v) => setSimParams({ ...simParams, per_lap_final_soc_min: v === '' ? undefined : Number(v) })}
                                                        onClearValue={() => setSimParams({ ...simParams, per_lap_final_soc_min: undefined })}
                                                        min={0}
                                                        max={1}
                                                        step={0.01}
                                                    />
                                                    <div className="md:col-span-2">
                                                        <label className="font-mono text-xs font-bold uppercase tracking-wide text-retro-text/70">LAP START MODE</label>
                                                        <SegmentedControl
                                                            value={simParams.flying_lap ? 'flying' : 'standing'}
                                                            options={lapStartOptions}
                                                            onChange={(next) => setSimParams({ ...simParams, flying_lap: next === 'flying' })}
                                                        />
                                                    </div>
                                                </div>
                                            )}
                                        </ConfigSectionCard>

                                        <ConfigSectionCard
                                            icon={<BatteryCharging size={16} />}
                                            title="Regulations & Vehicle"
                                            accent="neutral"
                                            headerRight={sectionHeaderActions(
                                                'regulations',
                                                resetRegulationVehicleSection,
                                                openSections.regulations ? 'Collapse regulations settings' : 'Expand regulations settings',
                                            )}
                                        >
                                            {openSections.regulations && (
                                                <div className="space-y-4">
                                                    <div>
                                                        <label className="font-mono text-xs font-bold uppercase tracking-wide text-retro-text/70">REGULATIONS</label>
                                                        <SegmentedControl
                                                            value={simParams.regulations}
                                                            options={regulationOptions}
                                                            onChange={(next) => setSimParams({ ...simParams, regulations: next })}
                                                        />
                                                    </div>
                                                </div>
                                            )}
                                        </ConfigSectionCard>

                                        <ConfigSectionCard
                                            icon={<Wrench size={16} />}
                                            title="Tire Degradation"
                                            accent={simParams.enable_tire_degradation ? 'warn' : 'neutral'}
                                            headerRight={sectionHeaderActions(
                                                'tire',
                                                resetTireSection,
                                                openSections.tire ? 'Collapse tire settings' : 'Expand tire settings',
                                            )}
                                        >
                                            {openSections.tire && (
                                                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                                                    <div className="md:col-span-2">
                                                        <label className="font-mono text-xs font-bold uppercase tracking-wide text-retro-text/70">DEGRADATION MODEL</label>
                                                        <SegmentedControl
                                                            value={simParams.enable_tire_degradation ? 'enabled' : 'disabled'}
                                                            options={degradationOptions}
                                                            onChange={(next) => setSimParams({ ...simParams, enable_tire_degradation: next === 'enabled' })}
                                                        />
                                                    </div>
                                                    <ConfigInput
                                                        meta={{ id: 'tire_wear_rate_per_lap', label: 'WEAR RATE / LAP', hint: 'Grip fraction lost each lap.', error: fieldErrors.tire_wear_rate_per_lap } satisfies FieldMeta}
                                                        type="number"
                                                        value={simParams.tire_wear_rate_per_lap}
                                                        onValueChange={(v) => setSimParams({ ...simParams, tire_wear_rate_per_lap: Number(v) })}
                                                        min={0}
                                                        step={0.001}
                                                        disabled={!simParams.enable_tire_degradation}
                                                    />
                                                    <ConfigInput
                                                        meta={{ id: 'tire_min_grip_scale', label: 'MIN GRIP SCALE', hint: 'Lower bound for lap grip multiplier.', error: fieldErrors.tire_min_grip_scale } satisfies FieldMeta}
                                                        type="number"
                                                        value={simParams.tire_min_grip_scale}
                                                        onValueChange={(v) => setSimParams({ ...simParams, tire_min_grip_scale: Number(v) })}
                                                        min={0.01}
                                                        max={1}
                                                        step={0.01}
                                                        disabled={!simParams.enable_tire_degradation}
                                                    />
                                                </div>
                                            )}
                                        </ConfigSectionCard>

                                        <ConfigSectionCard
                                            icon={<Cpu size={16} />}
                                            title="Advanced Solver Settings"
                                            accent="neutral"
                                            headerRight={sectionHeaderActions(
                                                'advanced',
                                                resetAdvancedSolverSection,
                                                openSections.advanced ? 'Collapse advanced settings' : 'Expand advanced settings',
                                            )}
                                        >
                                            {openSections.advanced && (
                                                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                                                    <ConfigInput
                                                        meta={{ id: 'ds', label: 'SPATIAL STEP (m)', hint: 'Distance discretization step for NLP.', error: fieldErrors.ds } satisfies FieldMeta}
                                                        type="number"
                                                        value={simParams.ds}
                                                        onValueChange={(v) => setSimParams({ ...simParams, ds: Number(v) })}
                                                        min={1}
                                                        step={0.5}
                                                    />
                                                    <div className="md:col-span-2">
                                                        <label className="font-mono text-xs font-bold uppercase tracking-wide text-retro-text/70">COLLOCATION METHOD</label>
                                                        <SegmentedControl
                                                            value={simParams.collocation}
                                                            options={collocationOptions}
                                                            className="sm:grid-cols-3"
                                                            onChange={(next) => setSimParams({ ...simParams, collocation: next })}
                                                        />
                                                    </div>
                                                    <div className="md:col-span-2">
                                                        <label className="font-mono text-xs font-bold uppercase tracking-wide text-retro-text/70">NLP BACKEND</label>
                                                        <SegmentedControl
                                                            value={simParams.nlp_solver}
                                                            options={solverOptions}
                                                            className="sm:grid-cols-2 lg:grid-cols-4"
                                                            onChange={(next) => setSimParams({ ...simParams, nlp_solver: next })}
                                                        />
                                                    </div>
                                                    <div className="md:col-span-2 grid grid-cols-1 gap-4 rounded-lg border border-panel-border bg-panel-muted/40 p-3 md:grid-cols-2">
                                                        <ConfigInput
                                                            meta={{
                                                                id: 'ipopt_linear_solver',
                                                                label: 'IPOPT LINEAR SOLVER',
                                                                hint: simParams.nlp_solver === 'ipopt' ? 'Examples: mumps, ma97.' : 'Select IPOPT backend to edit.',
                                                            } satisfies FieldMeta}
                                                            type="text"
                                                            value={simParams.ipopt_linear_solver}
                                                            onValueChange={(v) => setSimParams({ ...simParams, ipopt_linear_solver: v })}
                                                            disabled={simParams.nlp_solver !== 'ipopt'}
                                                        />
                                                        <div>
                                                            <label className="font-mono text-xs font-bold uppercase tracking-wide text-retro-text/70">IPOPT HESSIAN</label>
                                                            <SegmentedControl
                                                                value={simParams.ipopt_hessian}
                                                                options={hessianOptions}
                                                                onChange={(next) => setSimParams({ ...simParams, ipopt_hessian: next })}
                                                                className="sm:grid-cols-2"
                                                            />
                                                            {simParams.nlp_solver !== 'ipopt' && (
                                                                <FieldHint meta={{ id: 'ipopt_hessian', hint: 'Hessian mode is only used when IPOPT is selected.' }} />
                                                            )}
                                                        </div>
                                                    </div>
                                                </div>
                                            )}
                                        </ConfigSectionCard>
                                    </div>

                                    <aside className="space-y-4 lg:sticky lg:top-0 lg:h-fit">
                                        <section className="config-card rounded-xl border border-panel-border bg-panel-bg/95 p-4">
                                            <div className="mb-3 flex items-center gap-2 border-b border-panel-border/70 pb-3">
                                                {validationState.isValid ? <ShieldCheck size={16} className="text-emerald-500" /> : <ShieldAlert size={16} className="text-amber-500" />}
                                                <h3 className="font-mono text-sm font-bold uppercase">Run Readiness</h3>
                                            </div>
                                            <div className="space-y-2 font-mono text-xs">
                                                <ReadinessRow label="Track Source" value={trackSource === 'local' ? 'Local / TUMFTM' : 'FastF1 Session'} />
                                                <ReadinessRow label="Regulations" value={regulationLabel(simParams.regulations)} />
                                                <ReadinessRow label="Horizon" value={`${simParams.laps} lap(s)`} />
                                                <ReadinessRow label="Lap Start Mode" value={simParams.flying_lap ? 'Flying lap' : 'Standing start'} />
                                                <ReadinessRow label="Collocation" value={simParams.collocation} />
                                                <ReadinessRow label="Solver" value={simParams.nlp_solver} />
                                                <ReadinessRow label="SOC" value={`${(simParams.initial_soc * 100).toFixed(0)}% -> ${(simParams.final_soc_min * 100).toFixed(0)}%`} />
                                                <ReadinessRow label="Tire Degradation" value={simParams.enable_tire_degradation ? 'Enabled' : 'Disabled'} />
                                            </div>
                                            <div className="mt-4 rounded-lg border border-panel-border bg-panel-muted/60 p-3">
                                                <p className={cn(
                                                    'font-mono text-xs uppercase tracking-wide',
                                                    validationState.isValid ? 'text-emerald-700 dark:text-emerald-300' : 'text-amber-700 dark:text-amber-300',
                                                )}>
                                                    {validationState.isValid ? 'Ready to run simulation.' : `Blocked by ${validationState.errors.length} issue${validationState.errors.length === 1 ? '' : 's'}.`}
                                                </p>
                                                {validationState.errors.length > 0 && (
                                                    <ul className="mt-2 space-y-1 text-xs text-amber-700 dark:text-amber-300">
                                                        {validationState.errors.map((error) => (
                                                            <li key={error}>• {error}</li>
                                                        ))}
                                                    </ul>
                                                )}
                                            </div>
                                        </section>

                                        <section className="config-card rounded-xl border border-panel-border bg-panel-bg/95 p-4">
                                            <div className="mb-3 flex items-center gap-2 border-b border-panel-border/70 pb-3">
                                                <BatteryCharging size={16} className="text-f1-red" />
                                                <h3 className="font-mono text-sm font-bold uppercase">Regulation Delta</h3>
                                            </div>
                                            <div className="space-y-2 font-mono text-xs">
                                                <ReadinessRow
                                                    label="Deploy Power"
                                                    value={`${REGULATION_ERS_DELTA[simParams.regulations].deployKw} kW`}
                                                />
                                                <ReadinessRow
                                                    label="Recovery / Lap"
                                                    value={`${REGULATION_ERS_DELTA[simParams.regulations].recoveryLimitMj.toFixed(1)} MJ`}
                                                />
                                                <ReadinessRow
                                                    label="Usable Battery"
                                                    value={`${REGULATION_ERS_DELTA[simParams.regulations].usableBatteryMj.toFixed(1)} MJ`}
                                                />
                                            </div>
                                        </section>
                                    </aside>
                                </div>
                            </div>
                        )}

                        {activeTab === 'results' && (
                            <ResultsDashboard
                                results={results}
                                trackName={selectedTrackName || ''}
                                onBack={() => setActiveTab('track')}
                            />
                        )}
                    </div>
                </main>
            </div >
        </div >
    );
}

interface ConfigInputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'onChange'> {
    meta: FieldMeta;
    onValueChange: (value: string) => void;
    onClearValue?: () => void;
}

function ConfigInput({ meta, onValueChange, onClearValue, className, ...props }: ConfigInputProps) {
    const hasValue = props.value !== undefined && props.value !== null && props.value !== '';
    const canClear = Boolean(onClearValue) && hasValue;

    return (
        <div className="flex flex-col gap-1">
            <label htmlFor={meta.id} className="font-mono text-xs font-bold uppercase tracking-wide text-retro-text/70">
                {meta.label}
            </label>
            <div className="relative">
                <input
                    id={meta.id}
                    aria-invalid={Boolean(meta.error)}
                    aria-describedby={`${meta.id}-hint`}
                    className={cn(
                        'w-full rounded-lg border bg-panel-muted/70 px-3 py-2 font-mono text-sm outline-none transition-all',
                        canClear && 'pr-16',
                        meta.error
                            ? 'border-amber-500/70 focus:border-amber-500 focus:ring-2 focus:ring-amber-500/25'
                            : 'border-panel-border focus:border-f1-red focus:ring-2 focus:ring-f1-red/20',
                        props.disabled && 'cursor-not-allowed opacity-60',
                        className,
                    )}
                    onChange={(event) => onValueChange(event.target.value)}
                    {...props}
                />
                {canClear && (
                    <button
                        type="button"
                        onClick={onClearValue}
                        className="absolute right-2 top-1/2 -translate-y-1/2 rounded border border-panel-border bg-panel-bg px-2 py-0.5 font-mono text-[10px] uppercase tracking-wide text-retro-text/70 hover:border-f1-red/40"
                        aria-label={`Clear ${meta.label}`}
                    >
                        Clear
                    </button>
                )}
            </div>
            <FieldHint meta={{ id: meta.id, hint: meta.hint, error: meta.error }} />
        </div>
    );
}

function ReadinessRow({ label, value }: { label: string; value: string }) {
    return (
        <div className="flex items-center justify-between gap-2 border-b border-panel-border/40 pb-1 last:border-none">
            <span className="text-retro-text/60">{label}</span>
            <span className="font-bold uppercase tracking-wide">{value}</span>
        </div>
    );
}

function NavButton({ active, onClick, icon, label, desc }: { active: boolean, onClick: () => void, icon: React.ReactNode, label: string, desc: string }) {
    return (
        <button onClick={onClick} className={cn("flex items-start gap-3 p-3 rounded-lg text-left transition-all border border-transparent", active ? "bg-white dark:bg-white/10 border-retro-border shadow-sm ring-1 ring-black/5 dark:ring-white/5" : "hover:bg-black/5 dark:hover:bg-white/5 hover:border-black/5")}>
            <div className={cn("mt-0.5", active ? "text-f1-red" : "text-retro-text/60")}>{icon}</div>
            <div>
                <div className={cn("font-bold font-mono text-sm", active ? "text-black dark:text-white" : "text-retro-text")}>{label}</div>
                <div className="text-xs text-retro-text/50 font-medium leading-tight">{desc}</div>
            </div>
        </button>
    )
}

export default App;
