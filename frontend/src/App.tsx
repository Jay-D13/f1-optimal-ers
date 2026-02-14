import React, { useState, useEffect } from 'react';
import { Activity, Settings, Map, Play, Loader2, Database, ChevronRight, AlertCircle, Download, Upload } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import TrackMap from './components/TrackMap';
import { api, type TrackPoint, type RacelinePoint, type SimulationParams, type FastF1Event, type FastF1Driver } from './api';
import { ThemeToggle } from './components/ThemeToggle';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
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
    const [simParams, setSimParams] = useState<SimulationParams>({
        track: '',
        laps: 1,
        regulations: '2025',
        initial_soc: 0.5,
        final_soc_min: 0.3,
        ds: 5.0,
        collocation: 'euler',
        nlp_solver: 'auto',
        use_tumftm: true,
        driver: undefined
    });

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
        if (!selectedTrackName) return;
        setSimulating(true);
        try {
            const params = {
                ...simParams,
                track: selectedTrackName,
                year: trackSource === 'fastf1' ? selectedYear : undefined,
                driver: selectedDriver || undefined,
                use_tumftm: trackSource === 'local' // Prefer TUMFTM for local, FastF1 for remote
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
                            disabled={!selectedTrackName || simulating}
                            className="w-full flex items-center justify-center gap-2 bg-f1-red hover:bg-red-600 disabled:bg-gray-400 text-white font-mono text-sm py-2 px-4 rounded transition-colors shadow-sm active:translate-y-[1px]"
                        >
                            {simulating ? <Loader2 className="animate-spin" size={16} /> : <Play size={16} />}
                            {simulating ? 'RUNNING...' : 'RUN SIMULATION'}
                        </button>
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
                            <div className="max-w-2xl mx-auto pt-4 pb-20">
                                <h2 className="text-2xl font-bold font-mono uppercase mb-8 border-b border-retro-border pb-2">Configuration</h2>

                                <SettingsSection title="SIMULATION PARAMETERS">
                                    <SettingsInput label="LAPS" type="number" value={simParams.laps} onChange={(v: string) => setSimParams({ ...simParams, laps: Number(v) })} min={1} />
                                    <SettingsInput label="INITIAL SOC (0-1)" type="number" step={0.1} value={simParams.initial_soc} onChange={(v: string) => setSimParams({ ...simParams, initial_soc: Number(v) })} min={0} max={1} />
                                    <SettingsInput label="MIN FINAL SOC (0-1)" type="number" step={0.1} value={simParams.final_soc_min} onChange={(v: string) => setSimParams({ ...simParams, final_soc_min: Number(v) })} min={0} max={1} />
                                </SettingsSection>

                                <SettingsSection title="REGULATIONS & VEHICLE">
                                    <div className="flex flex-col gap-1 mb-4">
                                        <label className="font-mono text-xs font-bold text-retro-text/60">REGULATIONS YEAR</label>
                                        <div className="flex gap-2">
                                            {['2025', '2026'].map(r => (
                                                <button key={r} onClick={() => setSimParams({ ...simParams, regulations: r as any })}
                                                    className={cn("px-4 py-2 font-mono text-sm border rounded flex-1 transition-colors", simParams.regulations === r ? "bg-f1-black text-white border-f1-black dark:bg-white dark:text-black dark:border-white" : "bg-white dark:bg-transparent hover:border-black dark:hover:border-white")}>
                                                    {r}
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                </SettingsSection>

                                <SettingsSection title="SOLVER (ADVANCED)">
                                    <SettingsInput label="SPATIAL STEP (m)" type="number" value={simParams.ds} onChange={(v: string) => setSimParams({ ...simParams, ds: Number(v) })} min={1} />

                                    <div className="flex flex-col gap-1 mb-4">
                                        <label className="font-mono text-xs font-bold text-retro-text/60">COLLOCATION METHOD</label>
                                        <select className="bg-white dark:bg-white/5 border border-retro-border p-2 rounded font-mono text-sm outline-none focus:border-f1-red"
                                            value={simParams.collocation} onChange={e => setSimParams({ ...simParams, collocation: e.target.value as any })}>
                                            <option value="euler">Euler (Fastest)</option>
                                            <option value="trapezoidal">Trapezoidal (Balanced)</option>
                                            <option value="hermite_simpson">Hermite-Simpson (Precise)</option>
                                        </select>
                                    </div>

                                    <div className="flex flex-col gap-1 mb-4">
                                        <label className="font-mono text-xs font-bold text-retro-text/60">NLP SOLVER</label>
                                        <select className="bg-white dark:bg-white/5 border border-retro-border p-2 rounded font-mono text-sm outline-none focus:border-f1-red"
                                            value={simParams.nlp_solver} onChange={e => setSimParams({ ...simParams, nlp_solver: e.target.value as any })}>
                                            <option value="auto">Auto (Recommended)</option>
                                            <option value="ipopt">IPOPT</option>
                                            <option value="fatrop">Fatrop</option>
                                        </select>
                                    </div>
                                </SettingsSection>
                            </div>
                        )}

                        {activeTab === 'results' && (
                            <div className="h-full flex flex-col">
                                <h2 className="text-2xl font-bold font-mono uppercase mb-4 border-b border-retro-border pb-2 flex justify-between items-center">
                                    <span>Simulation Results</span>
                                    {results && <span className="text-xs font-normal bg-green-100 text-green-800 px-2 py-1 rounded">COMPLETED</span>}
                                </h2>
                                {results ? (
                                    <div className="font-mono text-xs bg-retro-bg p-4 border border-retro-border rounded overflow-auto flex-1 font-mono-pre">
                                        <pre>{JSON.stringify(results, null, 2)}</pre>
                                    </div>
                                ) : (
                                    <div className="flex-1 flex flex-col items-center justify-center text-retro-text/40 font-mono">
                                        <Activity size={48} className="mb-4 opacity-20" />
                                        <div>NO RESULTS AVAILABLE.</div>
                                        <button onClick={() => setActiveTab('track')} className="mt-4 text-f1-red hover:underline">Select a track to start</button>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </main>
            </div >
        </div >
    );
}

// Sub-components
const SettingsSection = ({ title, children }: { title: string, children: React.ReactNode }) => (
    <div className="mb-8">
        <h3 className="font-mono text-sm font-bold text-retro-border mb-4">{title}</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {children}
        </div>
    </div>
);

const SettingsInput = ({ label, type, value, onChange, ...props }: any) => (
    <div className="flex flex-col gap-1">
        <label className="font-mono text-xs font-bold text-retro-text/60">{label}</label>
        <input
            type={type}
            className="bg-white dark:bg-white/5 border border-retro-border p-2 rounded font-mono text-sm outline-none focus:border-f1-red focus:ring-1 focus:ring-f1-red"
            value={value}
            onChange={e => onChange(e.target.value)}
            {...props}
        />
    </div>
);

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
