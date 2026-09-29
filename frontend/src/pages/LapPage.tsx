import { useEffect, useMemo, useState } from 'react';
import { Play, Settings, X } from 'lucide-react';
import { api, type Job, type Round, type RunDetail, type RunInfo, type RunSettings } from '../api';
import { Legend, Stripes, ThemeToggle } from '../components/Chrome';
import { SettingsDrawer } from '../components/SettingsDrawer';
import { DistanceAxis, Trace } from '../components/Trace';
import { TrackMap } from '../components/TrackMap';
import { formatDelta, formatLap, formatTimestamp, nearestIndex } from '../format';

interface LapPageProps {
    round: Round;
    settings: RunSettings;
    onSettingsChange: (next: RunSettings) => void;
    jobs: Job[];
    running: Job | null;
    version: number;
    onStart: (round: number, settings: RunSettings) => Promise<Job>;
    onCancel: (id: string) => Promise<void>;
    onBack: () => void;
}

const MAP_W = 400;
const MAP_H = 540;

function Figure({ label, value, sub, color, first = false }: { label: string; value: string; sub: string; color?: string; first?: boolean }) {
    return (
        <div className={`flex flex-col gap-1 ${first ? 'pr-7' : 'border-l-2 border-rule px-7'}`}>
            <span className="cap">{label}</span>
            <span className="display text-5xl" style={color ? { color } : undefined}>{value}</span>
            <span className="num text-xs text-mute">{sub}</span>
        </div>
    );
}

function Toggle<T extends string>({ label, value, options, onChange }: { label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void }) {
    return (
        <div className="toggle" role="group" aria-label={label}>
            {options.map((o) => (
                <button key={o.value} type="button" aria-pressed={value === o.value} onClick={() => onChange(o.value)}>{o.label}</button>
            ))}
        </div>
    );
}

function JobPanel({ job, onCancel, onDismiss }: { job: Job; onCancel: () => void; onDismiss: () => void }) {
    const running = job.status === 'running';
    return (
        <div className="box flex flex-col gap-2 p-4" role="status">
            <div className="flex items-center justify-between gap-4">
                <span className="display text-2xl">{running ? <span className="pulse">Solving…</span> : job.status === 'cancelled' ? 'Run cancelled' : 'Run failed'}</span>
                {running
                    ? <button type="button" className="btn" onClick={onCancel}>Cancel</button>
                    : <button type="button" className="btn" style={{ padding: 8 }} onClick={onDismiss} aria-label="Dismiss"><X size={14} /></button>}
            </div>
            {/* column-reverse keeps the newest line in view */}
            <div className="flex max-h-48 flex-col-reverse overflow-auto">
                <pre className="num whitespace-pre-wrap text-xs leading-relaxed text-mute">{job.log.slice(-40).join('\n')}</pre>
            </div>
        </div>
    );
}

export function LapPage({ round, settings, onSettingsChange, jobs, running, version, onStart, onCancel, onBack }: LapPageProps) {
    const [runs, setRuns] = useState<RunInfo[] | null>(null);
    const [picked, setPicked] = useState<string | null>(null);
    const [detail, setDetail] = useState<RunDetail | null>(null);
    const [preview, setPreview] = useState<{ x: number[]; y: number[] } | null>(null);
    const [cursor, setCursor] = useState<number | null>(null);
    const [settingsOpen, setSettingsOpen] = useState(false);
    const [startError, setStartError] = useState<string | null>(null);
    const [dismissed, setDismissed] = useState<string | null>(null);

    const roundJob = jobs.find((j) => j.round === round.round) ?? null;

    useEffect(() => {
        let live = true;
        api.roundRuns(round.round).then((r) => live && setRuns(r)).catch(() => live && setRuns([]));
        return () => { live = false; };
    }, [round.round, version]);

    const matching = useMemo(
        () => (runs ?? []).filter((r) => r.session === settings.session && r.regulations === settings.regulations),
        [runs, settings.session, settings.regulations],
    );
    // The picked run, else the one this round's last job made, else the newest
    const lastJobRun = roundJob?.status === 'done' ? roundJob.run_id : null;
    const selected = matching.find((r) => r.run_id === (picked ?? lastJobRun)) ?? matching[0] ?? null;

    useEffect(() => {
        if (!selected) return;
        let live = true;
        api.run(round.track, selected.run_id, round.round).then((d) => live && setDetail(d)).catch(() => live && setDetail(null));
        return () => { live = false; };
    }, [round.track, round.round, selected?.run_id]); // eslint-disable-line react-hooks/exhaustive-deps

    useEffect(() => {
        if (!round.has_raceline || preview) return;
        api.raceline(round.round).then((r) => r.x && r.y && setPreview({ x: r.x, y: r.y })).catch(() => undefined);
    }, [round.round, round.has_raceline, preview]);

    const shown = selected && detail?.info.run_id === selected.run_id ? detail : null;
    const series = shown?.series ?? null;
    const length = shown?.summary.track_info?.total_length ?? series?.s.at(-1) ?? 1;
    const laps = shown?.info.n_laps ?? 1;
    const xLength = laps > 1 && series ? series.s[series.s.length - 1] : length;
    const index = series && cursor != null ? nearestIndex(series.s, cursor) : null;

    const traces = useMemo(() => {
        if (!series) return null;
        const kmh = series.v.map((v) => v * 3.6);
        const pole = shown?.pole_trace && laps === 1 && shown.info.session === 'qualifying' ? shown.pole_trace : null;
        return {
            speed: [
                ...(pole ? [{ s: pole.s, y: pole.v.map((v) => v * 3.6), color: 'var(--pole)' }] : []),
                { s: series.s, y: kmh, color: 'var(--model)', width: 2 },
            ],
            power: series.power ? [{ s: series.s, y: series.power.map((p) => p / 1e3), color: 'var(--deploy)', negColor: 'var(--harvest)', kind: 'signed' as const }] : [],
            soc: series.soc ? [{ s: series.s, y: series.soc.map((v) => v * 100), color: 'var(--ink)' }] : [],
        };
    }, [series, shown?.pole_trace, shown?.info.session, laps]);

    const cap = settings.regulations === '2025' ? 2.0 : settings.session === 'qualifying' ? round.quali_cap_mj : round.race_cap_mj;
    const recovered = shown?.summary.energy?.total_recovered_MJ;
    const lapTime = shown?.info.lap_time ?? null;
    const delta = shown?.info.delta_to_pole ?? null;
    const quali = settings.session === 'qualifying';

    const startRun = async () => {
        setStartError(null);
        setDismissed(null);
        setPicked(null);
        try {
            await onStart(round.round, quali ? { ...settings, laps: 1 } : settings);
        } catch (e) {
            setStartError((e as Error).message);
        }
    };

    const showJob = roundJob && roundJob.status !== 'done' && roundJob.id !== dismissed;
    const runDisabledReason = !round.has_raceline ? 'No racing line for the current layout yet'
        : running ? (running.round === round.round ? 'Running…' : `A run is in progress (R${running.round})`) : null;

    return (
        <div className="min-h-screen">
            <Stripes />
            <div className="mx-auto flex max-w-[1440px] flex-col gap-6 px-6 pb-8 pt-5 lg:px-12">
                <header className="flex flex-wrap items-center gap-4 border-b-2 border-ink pb-3.5">
                    <button type="button" className="link num text-sm" onClick={onBack}>← Season</button>
                    <h1 className="display text-4xl">R{String(round.round).padStart(2, '0')} {round.circuit}</h1>
                    <span className="text-sm text-mute">{round.name}</span>
                    <div className="flex-1" />
                    <Toggle label="Session" value={settings.session} onChange={(v) => onSettingsChange({ ...settings, session: v })}
                        options={[{ value: 'qualifying', label: 'Qualifying' }, { value: 'race', label: 'Race' }]} />
                    <Toggle label="Regulations" value={settings.regulations} onChange={(v) => onSettingsChange({ ...settings, regulations: v })}
                        options={[{ value: '2025', label: '2025' }, { value: '2026', label: '2026' }]} />
                    <button type="button" className="btn" onClick={() => setSettingsOpen(true)}><Settings size={14} /> Settings</button>
                    <button type="button" className="btn btn-go" onClick={startRun} disabled={runDisabledReason != null} title={runDisabledReason ?? undefined}>
                        <Play size={12} fill="currentColor" /> {quali ? 'Run lap' : 'Run race'}
                    </button>
                    <ThemeToggle />
                </header>

                {startError && <div className="box p-3 text-sm" role="alert">Couldn't start the run: {startError}</div>}

                <div className="flex flex-wrap items-end justify-between gap-6">
                    <div className="flex flex-wrap items-end gap-y-4">
                        <Figure first label={laps > 1 ? `model, ${laps} laps` : 'model lap'} value={formatLap(lapTime)} color={lapTime == null ? 'var(--mute)' : 'var(--model)'}
                            sub={shown ? `optimal ERS · Ipopt ${shown.info.solve_time?.toFixed(1) ?? '–'} s` : 'no run yet'} />
                        <Figure label="real pole" value={formatLap(round.pole?.time)} color="var(--pole)"
                            sub={round.pole ? `${round.pole.driver} · ${round.pole.team}` : ''} />
                        <Figure label="delta" value={formatDelta(delta)}
                            sub={delta == null ? (shown && !quali ? 'race run: no pole to compare' : '–') : `${formatDelta((delta / (round.pole?.time ?? 1)) * 100, 1)}% · model too ${delta < 0 ? 'fast' : 'slow'}`} />
                        <Figure label="recharged" value={recovered == null ? '–' : `${recovered.toFixed(1)}/${(cap * laps).toFixed(1)}`}
                            sub={`MJ · ${settings.regulations === '2026' ? 'event cap' : '2025 limit'}${laps > 1 ? ` × ${laps}` : ''}`} />
                    </div>
                    {matching.length > 1 && (
                        <div className="field">
                            <label htmlFor="run-pick">Run</label>
                            <select id="run-pick" value={selected?.run_id} onChange={(e) => setPicked(e.target.value)}>
                                {matching.map((r) => (
                                    <option key={r.run_id} value={r.run_id}>{formatTimestamp(r.timestamp)} · {formatLap(r.lap_time)}</option>
                                ))}
                            </select>
                        </div>
                    )}
                </div>

                <div className="flex flex-col gap-7 lg:flex-row">
                    <section className="box flex shrink-0 flex-col gap-2 self-start p-2.5" style={{ width: MAP_W + 24 }} aria-label="Track map">
                        {series?.x && series.y ? (
                            <TrackMap x={series.x} y={series.y} s={series.s} power={series.power} zones={shown?.straight_mode_zones}
                                cursorIndex={index} width={MAP_W} height={MAP_H} />
                        ) : preview ? (
                            <TrackMap x={preview.x} y={preview.y} width={MAP_W} height={MAP_H} />
                        ) : (
                            <div className="flex items-center justify-center text-sm text-mute" style={{ width: MAP_W, height: MAP_H }}>
                                {round.has_raceline ? 'Loading the track…' : 'No racing line for this layout yet.'}
                            </div>
                        )}
                        <div className="px-1.5 pb-1">
                            <Legend items={series?.power
                                ? [{ label: 'deploy', color: 'var(--deploy)' }, { label: 'harvest', color: 'var(--harvest)' }, { label: 'engine only', color: 'var(--neutral)' }, { label: 'Straight Mode', color: 'color-mix(in srgb, var(--ink) 20%, transparent)' }]
                                : [{ label: 'racing line', color: 'var(--neutral)' }]} />
                        </div>
                    </section>

                    <section className="flex min-w-0 flex-1 flex-col gap-3">
                        {showJob && roundJob && (
                            <JobPanel job={roundJob} onCancel={() => onCancel(roundJob.id)} onDismiss={() => setDismissed(roundJob.id)} />
                        )}
                        {traces && series ? (
                            <>
                                <div className="flex flex-wrap items-center gap-4">
                                    <Legend items={[{ label: 'model', color: 'var(--model)' }, ...(traces.speed.length > 1 ? [{ label: `pole lap · ${shown?.pole_trace?.driver}`, color: 'var(--pole)' }] : [])]} />
                                    <div className="flex-1" />
                                    <span className="num text-xs">
                                        {index != null
                                            ? `${series.s[index].toFixed(0)} m · ${(series.v[index] * 3.6).toFixed(0)} km/h`
                                              + (series.power?.[index] != null ? ` · ${formatDelta(series.power[index] / 1e3, 0)} kW` : '')
                                              + (series.soc?.[index] != null ? ` · charge ${(series.soc[index] * 100).toFixed(0)}%` : '')
                                            : 'Point at a chart to read values'}
                                    </span>
                                </div>
                                <Trace title="Speed" unit="km/h" lines={traces.speed} length={xLength} yMin={50} yMax={360} ticks={[100, 200, 300]} height={220} cursor={cursor} onCursor={setCursor} />
                                {traces.power.length > 0 && (
                                    <Trace title="MGU-K power" unit="kW · + deploy / − harvest" lines={traces.power} length={xLength} yMin={-400} yMax={400} ticks={[-350, 0, 350]} height={130} cursor={cursor} onCursor={setCursor} />
                                )}
                                {traces.soc.length > 0 && (
                                    <Trace title="State of charge" unit={settings.regulations === '2026' ? '% · 4 MJ window' : '%'} lines={traces.soc} length={xLength} yMin={0} yMax={100} ticks={[0, 50, 100]} tickUnit="%" height={96} cursor={cursor} onCursor={setCursor} />
                                )}
                                <DistanceAxis length={xLength} />
                            </>
                        ) : showJob && roundJob?.status === 'running' ? null : runs == null ? (
                            <div className="text-sm text-mute">Loading runs…</div>
                        ) : (
                            <div className="box flex flex-col gap-2 p-6">
                                <span className="display text-3xl">No {settings.session} run with {settings.regulations} rules yet</span>
                                <p className="text-sm">
                                    {round.has_raceline
                                        ? `Press ${quali ? 'Run lap' : 'Run race'}. A qualifying lap takes a few minutes: two reference laps, then the ERS optimisation.`
                                        : 'This round has no racing line for its current layout, so the model can’t run it yet.'}
                                </p>
                            </div>
                        )}
                    </section>
                </div>
            </div>

            <SettingsDrawer open={settingsOpen} onClose={() => setSettingsOpen(false)} round={round} settings={settings} onChange={onSettingsChange} />
        </div>
    );
}
