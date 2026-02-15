import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceArea
} from 'recharts';
import { Play, Pause, RotateCcw, Download, Loader2, Zap, TrendingUp, Clock } from 'lucide-react';
import { api } from '../api';
import TrackMap from './TrackMap';

interface ResultsDashboardProps {
    results: any;
    trackName: string;
    onBack: () => void;
}

const ResultsDashboard: React.FC<ResultsDashboardProps> = ({ results, trackName, onBack }) => {
    const [detailedData, setDetailedData] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Animation State
    const [playing, setPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [maxTime, setMaxTime] = useState(0);
    const [playbackSpeed, setPlaybackSpeed] = useState(1);
    const playbackSpeedRef = useRef(1); // To access in animation loop
    const lastFrameTimeRef = useRef<number>(0);
    const requestRef = useRef<number>(0);

    // Track Map Data (for animation context)
    const [trackData, setTrackData] = useState<any[]>([]);
    const summary = results?.results ?? results;

    // Load detailed data
    useEffect(() => {
        const loadData = async () => {
            if (!results?.run_id) return;
            setLoading(true);
            try {
                // Fetch detailed time-series
                const data = await api.getSimulationData(trackName, results.run_id);
                setDetailedData(data);

                // Also fetch track data for map if we don't have it
                try {
                    const tData = await api.getTrackData(trackName);
                    setTrackData(tData);
                } catch (e) {
                    console.warn("Could not load track map data for viz", e);
                }
            } catch (err) {
                console.error(err);
                setError("Failed to load detailed telemetry for visualization.");
            } finally {
                setLoading(false);
            }
        };

        loadData();
    }, [results, trackName]);

    // Prepare chart data
    const chartData = useMemo(() => {
        if (!detailedData) return [];
        return detailedData.s.map((s: number, i: number) => ({
            distance: s,
            velocity: detailedData.v[i] * 3.6, // km/h
            soc: detailedData.soc[i] * 100, // %
            power: detailedData.power[i] / 1000, // kW
            throttle: detailedData.throttle[i] * 100,
            brake: detailedData.brake[i] * 100,
        }));
    }, [detailedData]);

    // Initialize max time
    useEffect(() => {
        if (detailedData?.t && detailedData.t.length > 0) {
            setMaxTime(detailedData.t[detailedData.t.length - 1]);
        }
    }, [detailedData]);

    // Animation Logic
    const animate = (time: number) => {
        if (playing) {
            if (lastFrameTimeRef.current !== undefined) {
                const deltaTime = (time - lastFrameTimeRef.current) / 1000;
                setCurrentTime(prev => {
                    let next = prev + (deltaTime * playbackSpeedRef.current);
                    if (next >= maxTime) {
                        setPlaying(false);
                        return maxTime; // Stop at end or loop? Let's stop.
                    }
                    return next;
                });
            }
            lastFrameTimeRef.current = time;
            requestRef.current = requestAnimationFrame(animate);
        }
    };

    useEffect(() => {
        if (playing) {
            lastFrameTimeRef.current = performance.now();
            requestRef.current = requestAnimationFrame(animate);
        } else {
            if (requestRef.current) cancelAnimationFrame(requestRef.current);
        }
        return () => {
            if (requestRef.current) cancelAnimationFrame(requestRef.current);
        };
    }, [playing, playbackSpeed, maxTime]);

    // Derived current state from time
    const currentIndex = useMemo(() => {
        if (!detailedData?.t) return 0;
        // Find index where t >= currentTime. Simple linear search or binary search.
        // Given typically < 1000 points, findIndex is fine or we can optimize if needed.
        // Using findIndex might be jittery if we are between steps? 
        // Let's find the closest index.
        const idx = detailedData.t.findIndex((t: number) => t >= currentTime);
        return idx === -1 ? detailedData.t.length - 1 : idx;
    }, [currentTime, detailedData]);

    const telemetryAtCursor = chartData[currentIndex] || {};

    // Downloads
    const handleDownloadJSON = () => {
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(summary, null, 2));
        downloadFile(dataStr, `${trackName}_results.json`);
    };

    const handleDownloadCSV = () => {
        if (!chartData.length) return;
        const headers = Object.keys(chartData[0]).join(",");
        const rows = chartData.map((row: any) => Object.values(row).join(",")).join("\n");
        const csvContent = "data:text/csv;charset=utf-8," + encodeURIComponent(headers + "\n" + rows);
        downloadFile(csvContent, `${trackName}_telemetry.csv`);
    };

    const downloadFile = (uri: string, filename: string) => {
        const link = document.createElement('a');
        link.setAttribute("href", uri);
        link.setAttribute("download", filename);
        document.body.appendChild(link);
        link.click();
        link.remove();
    };

    if (loading) return <div className="flex h-full items-center justify-center space-x-2"><Loader2 className="animate-spin text-f1-red" /><span>Loading Telemetry...</span></div>;
    if (error) return <div className="p-8 text-red-500">{error}</div>;

    return (
        <div className="flex flex-col h-full bg-retro-bg overflow-hidden relative">
            {/* Header / Toolbar */}
            <div className="flex items-center justify-between p-4 border-b border-retro-border bg-white dark:bg-white/5 flex-shrink-0">
                <div className="flex items-center gap-4">
                    <button onClick={onBack} className="text-xs font-mono hover:text-f1-red">← BACK</button>
                    <h2 className="text-lg font-bold font-mono text-f1-red">RESULTS DASHBOARD</h2>
                </div>
                <div className="flex gap-2">
                    <button onClick={handleDownloadJSON} className="flex items-center gap-2 px-3 py-1.5 text-xs font-mono border border-retro-border rounded hover:bg-black/5 dark:hover:bg-white/10">
                        <Download size={14} /> SUMMARY JSON
                    </button>
                    <button onClick={handleDownloadCSV} className="flex items-center gap-2 px-3 py-1.5 text-xs font-mono border border-retro-border rounded hover:bg-black/5 dark:hover:bg-white/10">
                        <Download size={14} /> DATA CSV
                    </button>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {/* Top Row: KPI Cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <KPICard label="LAP TIME" value={summary?.performance?.lap_time?.toFixed(3)} unit="s" icon={<Clock size={16} />} />
                    <KPICard label="IMPROVEMENT" value={summary?.performance?.time_improvement?.toFixed(3)} unit="s" sub={summary?.performance?.lap_time_no_ers ? `${((summary.performance.time_improvement / summary.performance.lap_time_no_ers) * 100).toFixed(2)}%` : ''} icon={<TrendingUp size={16} />} />
                    <KPICard label="ENERGY USED" value={summary?.energy?.net_energy_MJ?.toFixed(3)} unit="MJ" icon={<Zap size={16} />} />
                    <KPICard label="AVG SPEED" value={summary?.velocity_stats?.avg_speed_kmh?.toFixed(0)} unit="km/h" />
                </div>

                {/* Main Viz Area */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[500px]">
                    {/* Left: Map & Animation */}
                    <div className="lg:col-span-1 border border-retro-border rounded-lg bg-white dark:bg-white/5 flex flex-col overflow-hidden relative">
                        <div className="flex-1 relative">
                            {trackData.length > 0 ? (
                                <TrackMap
                                    trackData={trackData}
                                    raceline={null} // We could overlay raceline if we wanted
                                // We need to pass a marker position... TrackMap might need update or we overlay here
                                // For now, let's just use TrackMap as background and maybe overlays
                                // Actually TrackMap doesn't support external marker props easily yet based on previous check
                                // So we might just show static map for now, or minimal update
                                />
                            ) : (
                                <div className="w-full h-full flex items-center justify-center text-xs font-mono opacity-50">Map Data Unavailable</div>
                            )}

                            {/* Overlay HUD */}
                            <div className="absolute top-4 left-4 bg-black/80 text-white p-3 rounded font-mono text-xs space-y-1 backdrop-blur-md border border-white/20">
                                <div className="text-f1-red font-bold text-lg">{telemetryAtCursor.velocity?.toFixed(0) || 0} <span className="text-xs text-gray-400">KM/H</span></div>
                                <div className="flex justify-between gap-4">
                                    <span>SOC:</span>
                                    <span className={telemetryAtCursor.soc < 30 ? "text-red-400" : "text-green-400"}>{telemetryAtCursor.soc?.toFixed(1)}%</span>
                                </div>
                                <div className="flex justify-between gap-4">
                                    <span>ERS:</span>
                                    <span className={telemetryAtCursor.power > 0 ? "text-blue-400" : "text-yellow-400"}>{telemetryAtCursor.power?.toFixed(0)} kW</span>
                                </div>
                                <div className="w-full bg-gray-700 h-1 mt-2 rounded-full overflow-hidden">
                                    <div className="bg-white h-full" style={{ width: `${telemetryAtCursor.throttle}%` }} />
                                </div>
                                <div className="w-full bg-gray-700 h-1 mt-1 rounded-full overflow-hidden">
                                    <div className="bg-red-500 h-full" style={{ width: `${telemetryAtCursor.brake}%` }} />
                                </div>
                            </div>
                        </div>

                        {/* Controls */}
                        <div className="p-4 border-t border-retro-border bg-retro-bg/50 backdrop-blur">
                            <div className="flex justify-between text-[10px] font-mono mb-1 text-retro-text/60">
                                <span>{currentTime.toFixed(2)}s</span>
                                <span>{maxTime.toFixed(2)}s</span>
                            </div>
                            <input
                                type="range"
                                min="0"
                                max={maxTime || 100}
                                step="0.01"
                                value={currentTime}
                                onChange={(e) => { setCurrentTime(parseFloat(e.target.value)); setPlaying(false); }}
                                className="w-full mb-3 accent-f1-red h-1 bg-gray-300 rounded-lg appearance-none cursor-pointer"
                            />
                            <div className="flex justify-between items-center">
                                <div className="flex gap-2">
                                    <button onClick={() => setPlaying(!playing)} className="p-2 rounded-full bg-f1-black text-white hover:bg-f1-red transition-colors">
                                        {playing ? <Pause size={16} /> : <Play size={16} />}
                                    </button>
                                    <button onClick={() => { setCurrentTime(0); setPlaying(true); }} className="p-2 rounded-full hover:bg-black/10">
                                        <RotateCcw size={16} />
                                    </button>
                                </div>
                                <select
                                    className="bg-transparent text-xs font-mono border border-retro-border rounded p-1"
                                    value={playbackSpeed}
                                    onChange={(e) => {
                                        const val = Number(e.target.value);
                                        setPlaybackSpeed(val);
                                        playbackSpeedRef.current = val;
                                    }}
                                >
                                    <option value={0.5}>0.5x</option>
                                    <option value={1}>1.0x</option>
                                    <option value={2}>2.0x</option>
                                    <option value={5}>5.0x</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    {/* Right: Charts */}
                    <div className="lg:col-span-2 flex flex-col gap-4">
                        {/* Velocity Chart */}
                        <ChartCard title="SPEED PROFILE (KM/H)">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                                    <XAxis dataKey="distance" type="number" unit="m" tick={{ fontSize: 10 }} />
                                    <YAxis domain={[0, 360]} tick={{ fontSize: 10 }} />
                                    <Tooltip contentStyle={{ fontSize: '12px', fontFamily: 'monospace' }} />
                                    <Line type="monotone" dataKey="velocity" stroke="#e10600" dot={false} strokeWidth={2} isAnimationActive={false} />
                                    {/* Sync cursor */}
                                    <ReferenceArea x1={telemetryAtCursor.distance} x2={telemetryAtCursor.distance} stroke="black" strokeOpacity={0.3} />
                                </LineChart>
                            </ResponsiveContainer>
                        </ChartCard>

                        {/* SOC & Power Chart */}
                        <ChartCard title="ENERGY MANAGEMENT">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                                    <XAxis dataKey="distance" type="number" unit="m" tick={{ fontSize: 10 }} />
                                    <YAxis yAxisId="left" domain={[0, 100]} unit="%" tick={{ fontSize: 10 }} stroke="#10b981" />
                                    <YAxis yAxisId="right" tick={{ fontSize: 10 }} unit="kW" stroke="#3b82f6" />
                                    <Tooltip contentStyle={{ fontSize: '12px', fontFamily: 'monospace' }} />
                                    <Legend />
                                    <Line yAxisId="left" type="monotone" dataKey="soc" name="SOC %" stroke="#10b981" dot={false} strokeWidth={2} isAnimationActive={false} />
                                    <Line yAxisId="right" type="monotone" dataKey="power" name="ERS Power (kW)" stroke="#3b82f6" dot={false} strokeWidth={1} isAnimationActive={false} />
                                    <ReferenceArea yAxisId="left" x1={telemetryAtCursor.distance} x2={telemetryAtCursor.distance} stroke="black" strokeOpacity={0.3} />
                                </LineChart>
                            </ResponsiveContainer>
                        </ChartCard>
                    </div>
                </div>
            </div>
        </div>
    );
};

const KPICard = ({ label, value, unit, sub, icon }: any) => (
    <div className="bg-white dark:bg-white/5 border border-retro-border p-4 rounded-lg flex items-center justify-between">
        <div>
            <div className="text-xs font-mono text-retro-text/60 mb-1">{label}</div>
            <div className="text-2xl font-bold font-mono tracking-tighter">
                {value ?? '-'} <span className="text-sm font-normal text-retro-text/40">{unit}</span>
            </div>
            {sub && <div className="text-xs font-mono text-green-600 mt-1">{sub}</div>}
        </div>
        {icon && <div className="text-retro-text/20">{icon}</div>}
    </div>
);

const ChartCard = ({ title, children }: any) => (
    <div className="flex-1 bg-white dark:bg-white/5 border border-retro-border p-4 rounded-lg flex flex-col min-h-[200px]">
        <h4 className="font-mono text-xs font-bold text-retro-text/60 mb-2">{title}</h4>
        <div className="flex-1 w-full min-h-0">
            {children}
        </div>
    </div>
);

export default ResultsDashboard;
